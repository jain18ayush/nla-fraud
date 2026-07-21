"""Phase 4 — SFT the Activation Verbalizer on fraud-MLP activations.

Adds a trainable Linear(128 -> 3584) injection adapter to the released AV and
fine-tunes with LoRA on next-token loss over the warm-start summaries.

The adapter is the only architectural change: the repo README notes a learned
affine `W·v + b` at the injection slot is a trainer-side-only modification, so
the checkpoint stays a stock HF causal LM.

Usage:
    python src/sft_av.py --config configs/experiment.yaml
    python src/sft_av.py --config configs/experiment.yaml --limit 512 --smoke
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from nla_common import (
    InjectionAdapter,
    NLAMeta,
    _from_pretrained,
    activations_tensor,
    assert_tokenizer_matches_sidecar,
    build_av_prompt_ids,
    build_injected_embeds,
    ensure_local,
    find_injection_pos,
    load_pairs,
    wrap_explanation,
    write_report,
)
from transformers import AutoModelForCausalLM


class AVDataset(Dataset):
    """(activation, tokenized target) pairs. The prompt is constant across rows,
    so it is built once in the training loop and only targets are per-row."""

    def __init__(self, df, tokenizer, max_target_tokens: int):
        self.acts = activations_tensor(df)  # raw, not normalized: the adapter
        # renormalizes its own output to injection_scale, so input scale is free.
        self.targets = []
        eos = tokenizer.eos_token_id
        for s in df["summary"].tolist():
            ids = tokenizer(
                wrap_explanation(s), add_special_tokens=False
            )["input_ids"][: max_target_tokens - 1]
            self.targets.append(ids + [eos])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.acts[i], self.targets[i]


def collate(batch, pad_id: int):
    acts = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    L = max(len(t) for t in tgts)
    ids = torch.full((len(tgts), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(tgts), L), dtype=torch.long)
    for i, t in enumerate(tgts):
        ids[i, : len(t)] = torch.tensor(t)
        mask[i, : len(t)] = 1
    return acts, ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--pairs", default="data/summaries.parquet")
    ap.add_argument("--out", default="checkpoints/av_sft")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training pairs (small default for smoke runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 steps, tiny batch — verifies the whole path runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sft, seed = cfg["sft"], cfg["seed"]
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── checkpoint + sidecar ──────────────────────────────────────────────────
    av_dir = ensure_local(sft["av_model_id"])
    meta = NLAMeta.from_yaml(Path(av_dir) / "nla_meta.yaml")
    assert meta.role == "av", f"expected AV sidecar, got {meta.role!r}"
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    assert_tokenizer_matches_sidecar(tok, meta)  # catches template/BOS drift

    prompt_ids_list = build_av_prompt_ids(tok, meta)
    inj_pos = find_injection_pos(prompt_ids_list, meta)
    prompt_ids = torch.tensor(prompt_ids_list, device=dev)
    P = len(prompt_ids_list)
    print(f"[av] prompt={P} tokens, injection at {inj_pos}, "
          f"injection_scale={meta.injection_scale}, d_model={meta.d_model}")

    # ── data ──────────────────────────────────────────────────────────────────
    n_train = args.limit if args.limit is not None else sft.get("n_pairs")
    train_df, hold_df = load_pairs(
        args.pairs, n_train, sft.get("fraud_fraction"), seed,
        holdout=sft.get("n_holdout", 1000),
    )
    d_target = len(train_df["activation_vector"].iloc[0])
    print(f"[data] train={len(train_df)} (fraud={train_df.label.mean():.1%}) "
          f"holdout={len(hold_df)} (fraud={hold_df.label.mean():.1%}) d={d_target}")

    ds = AVDataset(train_df, tok, cfg["summaries"]["max_tokens"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bs = 2 if args.smoke else sft["batch_size"]
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, pad_id))

    # ── model ─────────────────────────────────────────────────────────────────
    model = _from_pretrained(AutoModelForCausalLM, av_dir, torch.bfloat16,
                             trust_remote_code=True).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed: we feed inputs_embeds
    model.config.use_cache = False

    lora = LoraConfig(
        r=sft["lora_rank"], lora_alpha=sft["lora_alpha"], lora_dropout=0.05,
        target_modules=sft["lora_target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    adapter = InjectionAdapter(d_target, meta.d_model, meta.injection_scale).to(dev)
    embed_layer = model.get_input_embeddings()

    # Gemma-3 needs sqrt(hidden_size) after a raw embedding lookup; Qwen is 1.0.
    embed_scale = (math.sqrt(meta.d_model)
                   if getattr(model.config, "model_type", "").startswith("gemma")
                   else 1.0)

    # ── optimizer: adapter at a high LR, LoRA at a low one ────────────────────
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": adapter.parameters(), "lr": sft["lr_adapter"]},
        {"params": lora_params, "lr": sft["lr_lora"]},
    ], weight_decay=0.0)

    epochs = 1 if args.smoke else sft["sft_epochs"]
    accum = sft["grad_accum"]
    total_steps = max(1, (len(dl) // accum) * epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps) + 1, total_steps)

    warmup_epochs = sft.get("adapter_warmup_epochs", 1)
    log = []
    step = 0
    t0 = time.time()

    for ep in range(epochs):
        # Adapter-only warm-up: the adapter starts random, so early LoRA updates
        # would be chasing noise at the injection slot. Freeze LoRA for the first
        # epoch and let the adapter find the right region of embedding space.
        adapter_only = ep < warmup_epochs and not args.smoke
        for p in lora_params:
            p.requires_grad_(not adapter_only)
        print(f"[epoch {ep}] {'adapter-only warm-up' if adapter_only else 'adapter + LoRA'}")

        for i, (acts, tgt_ids, tgt_mask) in enumerate(dl):
            acts, tgt_ids, tgt_mask = acts.to(dev), tgt_ids.to(dev), tgt_mask.to(dev)
            B, L = tgt_ids.shape

            v = adapter(acts)  # [B, d_model] fp32, L2 == injection_scale
            embeds = build_injected_embeds(
                embed_layer, prompt_ids, inj_pos, v, tgt_ids, embed_scale
            )
            attn = torch.cat(
                [torch.ones(B, P, dtype=torch.long, device=dev), tgt_mask], dim=1
            )
            # Supervise the target tokens only; -100 masks the prompt and padding.
            labels = torch.full((B, P + L), -100, dtype=torch.long, device=dev)
            labels[:, P:] = tgt_ids.masked_fill(tgt_mask == 0, -100)

            out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)
            (out.loss / accum).backward()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(adapter.parameters()) + lora_params, 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or args.smoke:
                    rec = {"step": step, "epoch": ep, "loss": float(out.loss),
                           "ppl": float(torch.exp(out.loss.detach().float())),
                           "elapsed_s": round(time.time() - t0, 1)}
                    log.append(rec)
                    print(f"  step {step}/{total_steps} loss={rec['loss']:.4f} "
                          f"ppl={rec['ppl']:.2f}")
                if args.smoke and step >= 2:
                    break
        if args.smoke:
            break

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    torch.save({"state_dict": adapter.state_dict(), "d_target": d_target,
                "injection_scale": meta.injection_scale}, out_dir / "adapter.pt")
    hold_df.to_parquet(out_dir / "holdout.parquet")
    write_report("reports/phase4_av_sft.json", {
        "av_model_id": sft["av_model_id"], "n_train": len(train_df),
        "fraud_fraction": sft.get("fraud_fraction"),
        "d_target": d_target, "prompt_tokens": P, "injection_pos": inj_pos,
        "injection_scale": meta.injection_scale, "epochs": epochs,
        "total_steps": total_steps, "log": log,
        "final_loss": log[-1]["loss"] if log else None,
        "wall_clock_s": round(time.time() - t0, 1),
    })
    print(f"[av] saved to {out_dir}")


if __name__ == "__main__":
    main()
