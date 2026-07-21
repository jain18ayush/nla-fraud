"""Phase 4 — SFT the Activation Reconstructor into the fraud MLP's 128-d space.

Truncated (K+1)-layer LM + pretrained value_head + a new Linear(3584 -> 128).
Trained with normalized MSE against the target activations, LoRA on the trunk,
head at full precision.

Loss is `mse_nrm = 2*(1-cos)` — see nla_common.target_mse_scale for why the
scale constant is sqrt(128) here and not the sidecar's sqrt(3584).

Usage:
    python src/sft_ar.py --config configs/experiment.yaml
    python src/sft_ar.py --config configs/experiment.yaml --limit 512 --smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from nla_common import (
    FraudAR,
    NLAMeta,
    activations_tensor,
    build_ar_inputs,
    cosine,
    ensure_local,
    fve_nrm,
    load_pairs,
    mse_nrm,
    target_mse_scale,
    wrap_explanation,
    write_report,
)


class ARDataset(Dataset):
    """(explanation text, gold activation). During SFT the "explanation" is the
    ground-truth warm-start summary; in Phase 5 it is replaced by the AV's own
    sampled generations."""

    def __init__(self, df):
        self.acts = activations_tensor(df)
        # Same wrapping as the AV targets, then stripped back to the inner text
        # by the AR template — keeps SFT and RL inputs identical in form.
        self.texts = [wrap_explanation(s) for s in df["summary"].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.acts[i]


def collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


@torch.no_grad()
def evaluate(ar, tok, meta, df, scale, dev, max_len, bs=16):
    ar.eval()
    ds = ARDataset(df)
    dl = DataLoader(ds, batch_size=bs, collate_fn=collate)
    preds, golds = [], []
    for texts, acts in dl:
        ids, mask = build_ar_inputs(tok, meta, texts, max_len, dev)
        preds.append(ar(ids, mask).float().cpu())
        golds.append(acts)
    ar.train()
    pred, gold = torch.cat(preds), torch.cat(golds)
    return {
        "fve_nrm": fve_nrm(pred, gold, scale),
        "mse_nrm": float(mse_nrm(pred, gold, scale)),
        "cos_mean": float(cosine(pred, gold).mean()),
        "cos_median": float(cosine(pred, gold).median()),
        "n": len(gold),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--pairs", default="data/summaries.parquet")
    ap.add_argument("--out", default="checkpoints/ar_sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sft, seed = cfg["sft"], cfg["seed"]
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ar_dir = ensure_local(sft["ar_model_id"])
    meta = NLAMeta.from_yaml(Path(ar_dir) / "nla_meta.yaml")
    assert meta.role in ("ar", "critic"), f"expected AR sidecar, got {meta.role!r}"
    tok = AutoTokenizer.from_pretrained(ar_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_train = args.limit if args.limit is not None else sft.get("n_pairs")
    train_df, hold_df = load_pairs(
        args.pairs, n_train, sft.get("fraud_fraction"), seed,
        holdout=sft.get("n_holdout", 1000),
    )
    d_target = len(train_df["activation_vector"].iloc[0])
    scale = target_mse_scale(d_target)
    print(f"[ar] d_target={d_target} mse_scale={scale:.3f} "
          f"(sidecar's own was {meta.mse_scale:.3f} for d_model={meta.d_model})")
    print(f"[data] train={len(train_df)} holdout={len(hold_df)}")

    ds = ARDataset(train_df)
    bs = 2 if args.smoke else sft["batch_size"]
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    collate_fn=collate)

    ar = FraudAR(ar_dir, meta, d_target,
                 head_mode=sft.get("ar_head_mode", "stack")).to(dev)
    ar.backbone.gradient_checkpointing_enable()
    ar.backbone.config.use_cache = False

    lora = LoraConfig(
        r=sft["lora_rank"], lora_alpha=sft["lora_alpha"], lora_dropout=0.05,
        target_modules=sft["lora_target_modules"], bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    ar.backbone = get_peft_model(ar.backbone, lora)
    trunk = [p for p in ar.backbone.parameters() if p.requires_grad]
    head = ar.trainable_head_parameters()
    for p in head:
        p.requires_grad_(True)
    print(f"[ar] trainable: trunk(LoRA)={sum(p.numel() for p in trunk):,} "
          f"head={sum(p.numel() for p in head):,}")

    opt = torch.optim.AdamW([
        {"params": head, "lr": sft["lr_ar_head"]},
        {"params": trunk, "lr": sft["lr_lora"]},
    ], weight_decay=0.0)

    epochs = 1 if args.smoke else sft["sft_epochs"]
    accum = sft["grad_accum"]
    max_len = sft.get("ar_max_len", 320)
    total_steps = max(1, (len(dl) // accum) * epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps) + 1, total_steps)

    log, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        for i, (texts, acts) in enumerate(dl):
            acts = acts.to(dev)
            ids, mask = build_ar_inputs(tok, meta, texts, max_len, dev)
            pred = ar(ids, mask)
            loss = mse_nrm(pred, acts, scale)
            (loss / accum).backward()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trunk + head, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or args.smoke:
                    with torch.no_grad():
                        c = float(cosine(pred.detach().float(), acts).mean())
                    rec = {"step": step, "epoch": ep, "mse_nrm": float(loss),
                           "cos": c, "elapsed_s": round(time.time() - t0, 1)}
                    log.append(rec)
                    print(f"  step {step}/{total_steps} mse_nrm={loss:.4f} cos={c:.3f}")
                if args.smoke and step >= 2:
                    break
        if args.smoke:
            break

    metrics = evaluate(ar, tok, meta, hold_df.head(64 if args.smoke else len(hold_df)),
                       scale, dev, max_len)
    print(f"[ar] holdout (teacher-forced summaries): {metrics}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ar.backbone.save_pretrained(out_dir / "lora")
    torch.save({"out": ar.out.state_dict(),
                "value_head": (ar.value_head.state_dict()
                               if isinstance(ar.value_head, torch.nn.Linear) else None),
                "d_target": d_target, "head_mode": sft.get("ar_head_mode", "stack")},
               out_dir / "head.pt")
    write_report("reports/phase4_ar_sft.json", {
        "ar_model_id": sft["ar_model_id"], "n_train": len(train_df),
        "d_target": d_target, "mse_scale": scale, "epochs": epochs,
        "total_steps": total_steps, "log": log,
        "holdout_teacher_forced": metrics,
        "wall_clock_s": round(time.time() - t0, 1),
    })
    print(f"[ar] saved to {out_dir}")


if __name__ == "__main__":
    main()
