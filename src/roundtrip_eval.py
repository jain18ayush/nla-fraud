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

    t0 = time.time()
    av, adapter, av_tok, av_meta = load_av(sft, dev)
    prompt_ids_list = build_av_prompt_ids(av_tok, av_meta)
    prompt_ids = torch.tensor(prompt_ids_list, device=dev)
    inj_pos = find_injection_pos(prompt_ids_list, av_meta)

    raw = generate_explanations(
        av, adapter, av_tok, av_meta, acts, prompt_ids, inj_pos, dev,
        cfg["rl"]["max_new_tokens"], args.temperature, cfg["sft"].get("gen_batch", 16),
    )
    explanations = [parse_explanation(r) for r in raw]
    del av, adapter
    torch.cuda.empty_cache()

    ar, ar_tok, ar_meta = load_ar(sft, dev)
    preds = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(explanations), bs):
            ids, mask = build_ar_inputs(ar_tok, ar_meta, explanations[i:i + bs],
                                        sft.get("ar_max_len", 320), dev)
            preds.append(ar(ids, mask).float().cpu())
    pred = torch.cat(preds)

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


if __name__ == "__main__":
    main()
