"""Phase 4 exit criteria — AV -> AR round-trip on held-out activations.

Generates an explanation from each held-out activation with the SFT'd AV, feeds
it to the SFT'd AR, and reports FVE / cos against the original vector. Dumps 20
sample explanations for the eyeball check the plan requires before RL.

This is the same measurement the RL reward uses in Phase 5 (`r = -mse_nrm`), so
the number here is the baseline RL has to beat.

Usage:
    python src/roundtrip_eval.py --config configs/experiment.yaml --limit 1000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_common import (
    FraudAR,
    InjectionAdapter,
    NLAMeta,
    _from_pretrained,
    activations_tensor,
    assert_tokenizer_matches_sidecar,
    build_ar_inputs,
    build_av_prompt_ids,
    build_injected_embeds,
    cosine,
    ensure_local,
    find_injection_pos,
    fve_nrm,
    mse_nrm,
    parse_explanation,
    target_mse_scale,
    write_report,
)


def sattolo_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random single-cycle permutation of range(n) — guaranteed to
    have zero fixed points (a derangement), via Sattolo's algorithm.

    Used for the Phase 7 activation-swap control: index i must map to some
    j != i for every row, deterministically from the run seed.
    """
    assert n > 1, "cannot derange fewer than 2 rows"
    idx = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i)  # note: upper bound EXCLUDES i (unlike Fisher-Yates)
        idx[i], idx[j] = idx[j], idx[i]
    assert (idx != np.arange(n)).all(), "derangement produced a fixed point"
    return idx


def token_jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity — the text-level diagnostic for the swap
    control. If the AV emits near-identical explanations regardless of which
    activation it was fed, this stays high across unrelated row pairs.
    """
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_av(sft, dev):
    av_dir = ensure_local(sft["av_model_id"])
    meta = NLAMeta.from_yaml(Path(av_dir) / "nla_meta.yaml")
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    assert_tokenizer_matches_sidecar(tok, meta)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = _from_pretrained(AutoModelForCausalLM, av_dir, torch.bfloat16,
                             trust_remote_code=True)
    ckpt = Path(sft["_av_ckpt"])
    model = PeftModel.from_pretrained(model, ckpt / "lora").to(dev).eval()
    blob = torch.load(ckpt / "adapter.pt", map_location="cpu")
    adapter = InjectionAdapter(blob["d_target"], meta.d_model, blob["injection_scale"])
    adapter.load_state_dict(blob["state_dict"])
    return model.merge_and_unload(), adapter.to(dev).eval(), tok, meta


def load_ar(sft, dev):
    ar_dir = ensure_local(sft["ar_model_id"])
    meta = NLAMeta.from_yaml(Path(ar_dir) / "nla_meta.yaml")
    tok = AutoTokenizer.from_pretrained(ar_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ckpt = Path(sft["_ar_ckpt"])
    blob = torch.load(ckpt / "head.pt", map_location="cpu")
    ar = FraudAR(ar_dir, meta, blob["d_target"], head_mode=blob["head_mode"])
    ar.backbone = PeftModel.from_pretrained(ar.backbone, ckpt / "lora").merge_and_unload()
    ar.out.load_state_dict(blob["out"])
    if blob["value_head"] is not None:
        ar.value_head.load_state_dict(blob["value_head"])
    return ar.to(dev).eval(), tok, meta


@torch.no_grad()
def generate_explanations(model, adapter, tok, meta, acts, prompt_ids, inj_pos,
                          dev, max_new_tokens, temperature, batch_size):
    embed_layer = model.get_input_embeddings()
    embed_scale = (math.sqrt(meta.d_model)
                   if getattr(model.config, "model_type", "").startswith("gemma")
                   else 1.0)
    outs = []
    for i in range(0, len(acts), batch_size):
        chunk = acts[i:i + batch_size].to(dev)
        v = adapter(chunk)
        embeds = build_injected_embeds(embed_layer, prompt_ids, inj_pos, v,
                                       None, embed_scale)
        gen = model.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=dev),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.pad_token_id,
        )
        # With inputs_embeds and no input_ids, generate() returns ONLY the new
        # tokens — no prompt slice to strip.
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(acts))}/{len(acts)}", end="\r")
    print()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--av-ckpt", default="checkpoints/av_sft")
    ap.add_argument("--ar-ckpt", default="checkpoints/ar_sft")
    ap.add_argument("--holdout", default="checkpoints/av_sft/holdout.parquet")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. RL samples at 1.0; greedy is the fair "
                         "point estimate for a checkpoint comparison.")
    ap.add_argument("--tag", default="sft")
    ap.add_argument("--swap", action="store_true",
                    help="Phase 7 baseline #1: activation-swap validity control. "
                         "Runs real round-trip, activation-swap round-trip, and "
                         "a mean-activation floor on the SAME rows/seed, plus a "
                         "text-similarity diagnostic between the real and swap "
                         "explanations. Writes reports/phase7_swap_control.json "
                         "instead of the normal phase4 report.")
    ap.add_argument("--gen-batch", type=int, default=None,
                    help="Override cfg sft.gen_batch (A100-80GB is idle at 16; "
                         "raise it — default None keeps the config value).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sft = dict(cfg["sft"], _av_ckpt=args.av_ckpt, _ar_ckpt=args.ar_ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    df = pd.read_parquet(args.holdout).head(args.limit).reset_index(drop=True)
    acts = activations_tensor(df)
    d_target = acts.shape[1]
    scale = target_mse_scale(d_target)
    print(f"[eval] {len(df)} held-out rows, fraud={df.label.mean():.1%}, d={d_target}")

    gen_batch = args.gen_batch or cfg["sft"].get("gen_batch", 16)

    t0 = time.time()
    av, adapter, av_tok, av_meta = load_av(sft, dev)
    prompt_ids_list = build_av_prompt_ids(av_tok, av_meta)
    prompt_ids = torch.tensor(prompt_ids_list, device=dev)
    inj_pos = find_injection_pos(prompt_ids_list, av_meta)

    raw = generate_explanations(
        av, adapter, av_tok, av_meta, acts, prompt_ids, inj_pos, dev,
        cfg["rl"]["max_new_tokens"], args.temperature, gen_batch,
    )
    explanations = [parse_explanation(r) for r in raw]

    if args.swap:
        # Build a derangement i -> j (j != i for every row) so the AV sees a
        # DIFFERENT row's activation. It generates z_j from h_j; that gets fed
        # to the AR; the reconstruction is graded against h_i (this row's own
        # activation, NOT h_j) below. That grading choice is the entire point
        # of the control — see module docstring / task spec.
        rng = np.random.default_rng(cfg["seed"])
        perm = sattolo_derangement(len(acts), rng)
        acts_swapped = acts[perm]

        raw_swap = generate_explanations(
            av, adapter, av_tok, av_meta, acts_swapped, prompt_ids, inj_pos, dev,
            cfg["rl"]["max_new_tokens"], args.temperature, gen_batch,
        )
        explanations_swap = [parse_explanation(r) for r in raw_swap]

    del av, adapter
    torch.cuda.empty_cache()

    ar, ar_tok, ar_meta = load_ar(sft, dev)

    def run_ar(explanations_, bs=16):
        preds = []
        with torch.no_grad():
            for i in range(0, len(explanations_), bs):
                ids, mask = build_ar_inputs(ar_tok, ar_meta, explanations_[i:i + bs],
                                            sft.get("ar_max_len", 320), dev)
                preds.append(ar(ids, mask).float().cpu())
        return torch.cat(preds)

    pred = run_ar(explanations)

    if not args.swap:
        cos = cosine(pred, acts)
        metrics = {
            "tag": args.tag,
            "n": len(df),
            "fve_nrm": fve_nrm(pred, acts, scale),
            "mse_nrm": float(mse_nrm(pred, acts, scale)),
            "cos_mean": float(cos.mean()),
            "cos_median": float(cos.median()),
            "cos_p10": float(cos.quantile(0.10)),
            "cos_p90": float(cos.quantile(0.90)),
            # The fraud slice is what we actually care about and it is tiny — report
            # it separately so a good aggregate number can't hide a bad one there.
            "fve_fraud_slice": (fve_nrm(pred[df.label.values == 1],
                                        acts[df.label.values == 1], scale)
                                if (df.label.values == 1).sum() > 10 else None),
            "n_fraud": int((df.label.values == 1).sum()),
            "mean_explanation_chars": float(pd.Series(explanations).str.len().mean()),
            "pct_missing_close_tag": float(
                100 * sum("</explanation>" not in r for r in raw) / len(raw)
            ),
            "wall_clock_s": round(time.time() - t0, 1),
        }
        print(f"\n[eval] {metrics}")

        samples = [
            {"row": int(i), "fraud_score": float(df.fraud_score.iloc[i]),
             "label": int(df.label.iloc[i]), "cos": float(cos[i]),
             "explanation": explanations[i], "gold_summary": df.summary.iloc[i]}
            for i in range(min(20, len(df)))
        ]
        write_report(f"reports/phase4_roundtrip_{args.tag}.json",
                     {"metrics": metrics, "samples": samples})

        print("\n" + "=" * 78)
        for s in samples[:5]:
            print(f"[score={s['fraud_score']:.2f} label={s['label']} cos={s['cos']:.3f}]")
            print(s["explanation"][:400])
            print("-" * 78)
        return

    # ── Phase 7 activation-swap control ─────────────────────────────────────
    pred_swap = run_ar(explanations_swap)

    # Mean-activation floor: predict the L2-normalized holdout mean for every
    # row. This is what a fully uninformative predictor scores; the swap
    # condition should collapse toward it if the AV is prior-driven.
    # Deliberately NOT re-normalized after averaging (fve_nrm's own footgun
    # note: re-projecting the mean of unit vectors onto the sphere inflates
    # the comparison) — fed through the identical fve_nrm/mse_nrm pipeline as
    # every other condition so all three numbers are computed the same way.
    mu_unit = torch.nn.functional.normalize(acts, dim=-1).mean(dim=0)
    floor_pred = mu_unit.unsqueeze(0).expand_as(acts).contiguous()

    real_fve = fve_nrm(pred, acts, scale)
    real_mse = float(mse_nrm(pred, acts, scale))
    swap_fve = fve_nrm(pred_swap, acts, scale)  # graded against h_i, not h_j!
    swap_mse = float(mse_nrm(pred_swap, acts, scale))
    floor_fve = fve_nrm(floor_pred, acts, scale)
    floor_mse = float(mse_nrm(floor_pred, acts, scale))

    cos_real = cosine(pred, acts)
    cos_swap = cosine(pred_swap, acts)
    cos_floor = cosine(floor_pred, acts)

    jaccards = [token_jaccard(explanations[i], explanations_swap[i])
                for i in range(len(explanations))]
    jaccard_mean = float(np.mean(jaccards))
    jaccard_median = float(np.median(jaccards))

    gap_real_swap = real_fve - swap_fve
    gap_swap_floor = swap_fve - floor_fve
    # Judgment call (documented, not hidden): PASS requires the swap condition
    # to be AT OR BELOW the mean-activation floor (gap_swap_floor <= +0.05 —
    # i.e. no residual "prior credit" propping swap FVE up above what a fully
    # uninformative constant predictor gets) AND a real-vs-swap gap of at least
    # 0.15 abs FVE. Swap landing BELOW the floor is not a caveat on the pass —
    # it is the stronger version of it: a prior-driven AV would produce
    # near-floor (~0) FVE when scored against an unrelated row, because a
    # generic explanation reconstructs to ~the mean regardless of target. Swap
    # FVE undershooting the floor means the AR is confidently reconstructing
    # the SWAPPED PARTNER's real (and now systematically wrong-for-this-row)
    # direction — which requires the AV to have encoded real, row-specific
    # information about h_j into z_j. FAIL requires the gap to be under 0.05
    # abs FVE (swap indistinguishable from real). Anything else is reported as
    # AMBIGUOUS rather than forced.
    if gap_swap_floor <= 0.05 and gap_real_swap >= 0.15:
        verdict = "PASS"
    elif gap_real_swap < 0.05:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"

    metrics = {
        "tag": args.tag,
        "n": len(df),
        "seed": cfg["seed"],
        "temperature": args.temperature,
        "max_new_tokens": cfg["rl"]["max_new_tokens"],
        "gen_batch": gen_batch,
        "real": {
            "fve_nrm": real_fve, "mse_nrm": real_mse,
            "cos_mean": float(cos_real.mean()), "cos_median": float(cos_real.median()),
        },
        "swap": {
            "fve_nrm": swap_fve, "mse_nrm": swap_mse,
            "cos_mean": float(cos_swap.mean()), "cos_median": float(cos_swap.median()),
            "note": "pred_swap = AR(AV(h_j)) scored against h_i (i != j, derangement)",
        },
        "mean_floor": {
            "fve_nrm": floor_fve, "mse_nrm": floor_mse,
            "cos_mean": float(cos_floor.mean()), "cos_median": float(cos_floor.median()),
        },
        "text_similarity_jaccard": {
            "mean": jaccard_mean, "median": jaccard_median,
            "note": "token-set Jaccard between explanation(h_i) and explanation(h_j) "
                    "for the same row i; high values regardless of (i,j) mean the AV "
                    "text is prior-driven, not vector-driven",
        },
        "gap_real_minus_swap_fve": gap_real_swap,
        "gap_swap_minus_floor_fve": gap_swap_floor,
        "verdict": verdict,
        "pct_missing_close_tag_real": float(
            100 * sum("</explanation>" not in r for r in raw) / len(raw)
        ),
        "pct_missing_close_tag_swap": float(
            100 * sum("</explanation>" not in r for r in raw_swap) / len(raw_swap)
        ),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    print(f"\n[phase7-swap] {metrics}")

    samples = [
        {"row": int(i), "swap_partner_row": int(perm[i]),
         "fraud_score": float(df.fraud_score.iloc[i]), "label": int(df.label.iloc[i]),
         "cos_real": float(cos_real[i]), "cos_swap": float(cos_swap[i]),
         "jaccard": float(jaccards[i]),
         "explanation_real": explanations[i],
         "explanation_swap": explanations_swap[i],
         "gold_summary": df.summary.iloc[i]}
        for i in range(min(20, len(df)))
    ]
    write_report("reports/phase7_swap_control.json",
                 {"metrics": metrics, "samples": samples})

    print("\n" + "=" * 78)
    print(f"real FVE={real_fve:.4f}  swap FVE={swap_fve:.4f}  floor FVE={floor_fve:.4f}")
    print(f"gap(real-swap)={gap_real_swap:.4f}  gap(swap-floor)={gap_swap_floor:.4f}")
    print(f"jaccard(real,swap) mean={jaccard_mean:.4f} median={jaccard_median:.4f}")
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    for s in samples[:5]:
        print(f"[row={s['row']} partner={s['swap_partner_row']} "
              f"cos_real={s['cos_real']:.3f} cos_swap={s['cos_swap']:.3f} "
              f"jaccard={s['jaccard']:.3f}]")
        print("  REAL:", s["explanation_real"][:200])
        print("  SWAP:", s["explanation_swap"][:200])
        print("-" * 78)


if __name__ == "__main__":
    main()
