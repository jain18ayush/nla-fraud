"""
Phase 3b: Generate (activation, summary) pairs for NLA warm-start SFT.

Uses OpenRouter (OpenAI-compatible API) to write 3-5 bullet summaries per
transaction row. Progress is written line-by-line to data/summaries_progress.jsonl
so any crash or credit-limit stop is fully resumable — just re-run.

The JSONL stores only text data (row_idx, summary, features_json, fraud_score,
label). The final parquet joins back to the activation parquet by row_idx to
attach activation vectors, giving a clean 1:1 mapping:

  parquet row_idx  <->  activation_vector  <->  summary

Output: data/summaries.parquet
  columns: row_idx, activation_vector, activation_vector_normed,
           summary, features_json, fraud_score, label

Env vars required:
  OPENROUTER_API_KEY

Optional:
  OPENROUTER_MODEL  (overrides config)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "summary_cache"
PROGRESS_FILE = DATA_DIR / "summaries_progress.jsonl"
REPORTS_DIR = ROOT / "reports"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat"


def load_config() -> dict:
    with open(ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are interpreting the hidden-layer activations of a fraud-detection neural network. "
    "Your output is training data for a language model that must reconstruct those activations "
    "from your text — so every word must carry signal. Be terse and causal: say WHY each "
    "feature drives the score up or down. Output ONLY the bullet list. No preamble, no "
    "reasoning steps, no closing sentence. Start your response with `-` immediately."
)

_USER_TMPL = """\
Network fraud score: {fraud_score:.3f}

Transaction:
{serialized}

Gradient×input attributions (signed, higher magnitude = more influential):
{attributions}

Write 3-5 bullets. Each bullet: [exact feature value] → [why it pushes the score \
toward fraud or legitimacy]. Order by influence on the score. Max 140 tokens total.\
"""


def _attribution_text(row: dict, feature_cols: list[str]) -> str:
    items = []
    for c in feature_cols:
        v = row.get(c + "_attr")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            items.append((abs(float(v)), float(v), c))
    items.sort(reverse=True)
    lines = [f"  {c}: {v:+.3f}" for _, v, c in items[:8]]
    return "\n".join(lines) if lines else "(attributions not available)"


def build_prompt(row: dict, serialized: str, feature_cols: list[str]) -> tuple[str, str]:
    user = _USER_TMPL.format(
        fraud_score=float(row.get("fraud_score", 0.0)),
        serialized=serialized,
        attributions=_attribution_text(row, feature_cols),
    )
    return _SYSTEM, user


# ── JSONL progress file ───────────────────────────────────────────────────────

def load_progress() -> dict[int, dict]:
    """Returns {row_idx: record} for all completed rows."""
    done: dict[int, dict] = {}
    if not PROGRESS_FILE.exists():
        return done
    with open(PROGRESS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[int(rec["row_idx"])] = rec
            except Exception:
                continue
    return done


def _append_progress(record: dict) -> None:
    with open(PROGRESS_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ── Per-call disk cache ───────────────────────────────────────────────────────

def _cache_key(row_idx: int, model: str) -> str:
    # row_idx is the stable parquet row number, so this key is stable across runs
    return hashlib.md5(f"{row_idx}:{model}".encode()).hexdigest()[:16]


def _load_cache(key: str) -> Optional[str]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["summary"]
        except Exception:
            return None
    return None


def _save_cache(key: str, summary: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"summary": summary}))


# ── OpenRouter call ───────────────────────────────────────────────────────────

async def _call_openrouter(
    client,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
            msg = response.choices[0].message
            text = msg.content or ""
            if not text.strip() and hasattr(msg, "reasoning") and msg.reasoning:
                text = msg.reasoning
            text = text.strip()
            # If model leaked preamble instead of starting with a bullet, the
            # response is unusable — signal caller to retry with stricter prompt
            if text and not text[0] in ("-", "•", "*"):
                raise ValueError(f"bad_format: response did not start with a bullet")
            return text
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credit" in err_str.lower() or "insufficient" in err_str.lower():
                raise  # propagate immediately; caller will stop the run
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {e}; sleeping {wait}s …", flush=True)
            await asyncio.sleep(wait)
    return ""


# ── Core async generator ──────────────────────────────────────────────────────

async def _generate_all(
    rows: list[dict],
    serialized_texts: list[str],
    feature_cols: list[str],
    feat_jsons: list[str],
    model: str,
    max_tokens: int,
    concurrency: int,
    api_key: str,
    already_done: dict[int, dict],
    write_callback: Callable[[dict], None],
) -> tuple[int, int]:
    """
    Calls write_callback(record) immediately after each row completes.
    Returns (n_completed_this_run, n_failed).
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("[ERROR] openai not installed. Run: uv add openai", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/nla-fraud",
            "X-Title": "nla-fraud",
        },
    )
    semaphore = asyncio.Semaphore(concurrency)

    # Separate rows into: skip (progress), cache hit, needs API call
    pending = []
    cache_hits = 0
    for i, (row, txt, feat_json) in enumerate(zip(rows, serialized_texts, feat_jsons)):
        row_idx = int(row["_row_idx"])
        if row_idx in already_done:
            continue
        key = _cache_key(row_idx, model)
        cached = _load_cache(key)
        if cached is not None:
            rec = {
                "row_idx": row_idx,
                "summary": cached,
                "features_json": feat_json,
                "fraud_score": float(row.get("fraud_score", 0.0)),
                "label": int(row.get("label", 0)),
            }
            write_callback(rec)
            cache_hits += 1
        else:
            system, user = build_prompt(row, txt, feature_cols)
            pending.append((row_idx, key, system, user, feat_json,
                            float(row.get("fraud_score", 0.0)), int(row.get("label", 0))))

    print(f"  Already done (progress file): {len(already_done):,}")
    print(f"  Cache hits:                   {cache_hits:,}")
    print(f"  Needs API call:               {len(pending):,}")

    if not pending:
        await client.close()
        return cache_hits, 0

    from tqdm import tqdm
    completed = 0
    failed = 0
    credit_exhausted = False
    lock = asyncio.Lock()
    pbar = tqdm(total=len(pending), desc="Generating", unit="row", dynamic_ncols=True)

    async def _worker(row_idx, key, system, user, feat_json, fraud_score, label):
        nonlocal completed, failed, credit_exhausted
        if credit_exhausted:
            return
        try:
            summary = await _call_openrouter(client, model, system, user, max_tokens, semaphore)
            _save_cache(key, summary)
            write_callback({
                "row_idx": row_idx,
                "summary": summary,
                "features_json": feat_json,
                "fraud_score": fraud_score,
                "label": label,
            })
            async with lock:
                completed += 1
                pbar.update(1)
                pbar.set_postfix(done=completed, failed=failed)
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credit" in err_str.lower() or "insufficient" in err_str.lower():
                async with lock:
                    if not credit_exhausted:
                        credit_exhausted = True
                        pbar.write("\n[STOP] Credit limit hit — progress saved. Re-run to continue.")
            else:
                async with lock:
                    failed += 1
                    pbar.update(1)
                    pbar.set_postfix(done=completed, failed=failed)
                    pbar.write(f"[FAIL] row_idx={row_idx}: {e}")

    await asyncio.gather(*[_worker(*args) for args in pending])
    pbar.close()
    await client.close()
    return completed + cache_hits, failed


# ── Build final parquet from JSONL + activation parquet ──────────────────────

def build_parquet(layer: str = "l2") -> pd.DataFrame:
    """
    Join summaries_progress.jsonl with activations parquet by row_idx.
    Returns the merged dataframe ready to write as summaries.parquet.
    """
    # Load progress
    done = load_progress()
    if not done:
        print("[WARN] Progress file is empty.", file=sys.stderr)
        return pd.DataFrame()

    progress_df = pd.DataFrame(done.values())
    progress_df["row_idx"] = progress_df["row_idx"].astype(int)
    progress_df = progress_df[progress_df["summary"].str.strip().ne("")]

    # Load activation parquet
    act_path = DATA_DIR / f"activations_{layer}.parquet"
    act_df = pd.read_parquet(act_path, columns=["activation_vector", "activation_vector_normed"])
    act_df = act_df.reset_index().rename(columns={"index": "row_idx"})
    act_df["row_idx"] = act_df["row_idx"].astype(int)

    merged = progress_df.merge(act_df, on="row_idx", how="left")
    # Deduplicate (keep last in case of re-runs)
    merged = merged.drop_duplicates(subset="row_idx", keep="last")
    return merged[["row_idx", "activation_vector", "activation_vector_normed",
                   "summary", "features_json", "fraud_score", "label"]]


# ── Attributions (optional) ───────────────────────────────────────────────────

def compute_attributions(mlp, prep, df_sub, device) -> pd.DataFrame:
    import torch
    from target_model import make_tensors
    # make_tensors expects is_fraud; activation parquet uses label
    _df = df_sub.copy()
    if "is_fraud" not in _df.columns and "label" in _df.columns:
        _df["is_fraud"] = _df["label"]
    cats_t, nums_t, _ = make_tensors(_df, prep)
    mlp.eval()
    all_grads = []
    for start in range(0, len(df_sub), 256):
        end = min(start + 256, len(df_sub))
        nums_b = nums_t[start:end].float().to(device).requires_grad_(True)
        cats_b = cats_t[start:end].to(device)
        mlp(cats_b, nums_b).sum().backward()
        all_grads.append(nums_b.grad.detach().cpu().numpy())
    attrs = np.concatenate(all_grads) * nums_t.numpy()
    num_cols = [c for c in prep.num_cols if c in df_sub.columns]
    attr_df = pd.DataFrame(attrs, columns=[c + "_attr" for c in num_cols], index=df_sub.index)
    return pd.concat([df_sub, attr_df], axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    sum_cfg = cfg["summaries"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=sum_cfg["n_validate"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--layer", default=cfg["activations"]["hook_layer"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=sum_cfg["concurrency"])
    parser.add_argument("--no-attributions", action="store_true")
    parser.add_argument("--build-parquet-only", action="store_true",
                        help="Skip generation; convert progress JSONL -> parquet and exit")
    args = parser.parse_args()

    model = args.model or os.environ.get("OPENROUTER_MODEL") or sum_cfg.get("openrouter_model", DEFAULT_MODEL)
    max_tokens = sum_cfg["max_tokens"]
    n_target = sum_cfg["n_generate"] if args.full else args.n

    if args.build_parquet_only:
        df = build_parquet(args.layer)
        out = DATA_DIR / "summaries.parquet"
        df.to_parquet(out, index=False)
        print(f"Written {out}  ({len(df):,} rows)")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:       {model}")
    print(f"Target:      {n_target:,} rows")
    print(f"Max tokens:  {max_tokens}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Progress:    {PROGRESS_FILE}")

    # ── Load progress ─────────────────────────────────────────────────────────
    already_done = load_progress()
    print(f"\nRows already complete: {len(already_done):,} / {n_target:,}")

    # ── Load activations ──────────────────────────────────────────────────────
    act_path = DATA_DIR / f"activations_{args.layer}.parquet"
    if not act_path.exists():
        print(f"[ERROR] {act_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Load only feature columns (skip the large activation_vector arrays)
    act_df = pd.read_parquet(act_path)
    if "train_split" in act_df.columns:
        act_df = act_df[act_df["train_split"] == "train"].copy()

    n = min(n_target, len(act_df))
    df_sub = act_df.sample(n=n, random_state=cfg["seed"])

    # Stable row_idx = original parquet row number (before reset)
    df_sub = df_sub.copy()
    df_sub["_row_idx"] = df_sub.index   # index = position in full parquet file
    df_sub = df_sub.reset_index(drop=True)

    remaining = n - len([i for i in df_sub["_row_idx"] if i in already_done])
    print(f"Sampled {n:,} rows  ({remaining:,} remaining to generate)")

    # ── Dataset source + preprocessor ────────────────────────────────────────
    dataset_source = "ieee-fraud-detection"
    prep = None

    try:
        import torch as _torch
        from target_model import load_model_and_artifacts
        _, prep, _art = load_model_and_artifacts(_torch.device("cpu"))
        dataset_source = _art.get("dataset_source", dataset_source)
        print(f"Dataset source: {dataset_source}  (preprocessor loaded)")
    except Exception as e:
        json_art_path = DATA_DIR / "mlp_artifacts.json"
        if json_art_path.exists():
            with open(json_art_path) as f:
                dataset_source = json.load(f).get("dataset_source", dataset_source)
        print(f"Dataset source: {dataset_source}  (no preprocessor: {e}; skipping attributions)")

    # ── Attributions ──────────────────────────────────────────────────────────
    if not args.no_attributions and prep is not None:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlp, _, _ = load_model_and_artifacts(device)
            df_sub = compute_attributions(mlp, prep, df_sub, device)
            print("Attributions computed.")
        except Exception as e:
            print(f"  [WARN] Attributions failed: {e}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    from serialize import serialize_row, verbalizability_table

    rows_dict = df_sub.to_dict(orient="records")
    serialized_texts = [serialize_row(r, dataset_source) for r in rows_dict]

    skip_cols = {"activation_vector", "activation_vector_normed", "label",
                 "fraud_score", "train_split", "_row_idx"}
    feature_cols = [c for c in df_sub.columns if c not in skip_cols]
    feat_jsons = [
        json.dumps(
            {k: v for k, v in r.items() if k in feature_cols and not k.endswith("_attr")},
            default=str,
        )
        for r in rows_dict
    ]

    print(f"\nExample serialization:\n  {serialized_texts[0]}\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"Generating via OpenRouter ({model}) …")
    completed, failed = asyncio.run(
        _generate_all(
            rows=rows_dict,
            serialized_texts=serialized_texts,
            feature_cols=feature_cols,
            feat_jsons=feat_jsons,
            model=model,
            max_tokens=max_tokens,
            concurrency=args.concurrency,
            api_key=api_key,
            already_done=already_done,
            write_callback=_append_progress,
        )
    )

    # ── Status ────────────────────────────────────────────────────────────────
    final_done = load_progress()
    remaining = max(0, n_target - len(final_done))
    print(f"\n{'='*50}")
    print(f"  Completed this run:  {completed:,}")
    print(f"  Failed this run:     {failed:,}")
    print(f"  Total in progress:   {len(final_done):,} / {n_target:,}")
    print(f"  Still remaining:     {remaining:,}")
    print(f"{'='*50}")

    # ── Build parquet ─────────────────────────────────────────────────────────
    df_out = build_parquet(args.layer)
    out_path = DATA_DIR / "summaries.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\nParquet → {out_path}  ({len(df_out):,} rows)")
    if remaining > 0:
        print(f"Re-run to continue: python src/gen_summaries.py --model {model}")

    # ── Sample output ─────────────────────────────────────────────────────────
    if len(df_out) > 0:
        sample = df_out.sample(n=min(5, len(df_out)), random_state=cfg["seed"])
        print("\n=== Sample summaries ===")
        for _, row in sample.iterrows():
            feat = json.loads(row["features_json"])
            print(f"\n  [row {int(row['row_idx'])}] fraud={row['fraud_score']:.3f}  label={int(row['label'])}")
            print(f"  {serialize_row(feat, dataset_source)[:120]}…")
            print(f"  {row['summary']}")

    # Save verbalizability table
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "verbalizability_table.json", "w") as f:
        json.dump(verbalizability_table(dataset_source), f, indent=2)


if __name__ == "__main__":
    main()
