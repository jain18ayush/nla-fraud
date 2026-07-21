"""Phase 5 — GRPO on the AV with a simultaneous supervised AR step.

Recipe from the paper/repo, adapted to a single GPU:

  each step:
    1. sample a batch of activations h
    2. generate G explanations per activation from the AV at temperature 1.0
    3. reward r = -mse_nrm(AR(z), h)          (mse_nrm = 2*(1-cos), so r in [-4, 0])
    4. AR step:  one supervised MSE gradient step on all (z, h) pairs
    5. AV step:  GRPO with group-normalized advantages + KL penalty toward the
                 frozen SFT init

The reference policy for the KL term is the SFT checkpoint, NOT the pretrained
base. We get it without a second copy of the 7B by loading the SFT LoRA twice
under two PEFT adapter names ("policy", trainable; "ref", frozen) over shared
base weights, and switching with set_adapter(). The injection adapter is
likewise kept in two copies — it is a Linear(128, 3584), so the duplicate is
free.

Usage:
    python src/rl_grpo.py --config configs/experiment.yaml
    python src/rl_grpo.py --config configs/experiment.yaml --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    ensure_local,
    find_injection_pos,
    fve_nrm,
    normalize_to,
    parse_explanation,
    target_mse_scale,
    write_report,
)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_av_trainable(sft, dev):
    """AV with a trainable 'policy' LoRA and a frozen 'ref' LoRA over one base.

    roundtrip_eval.load_av calls merge_and_unload() because it only does
    inference. We must NOT merge here: the LoRA has to stay separable, both to
    receive gradients and to be switched off in favour of the reference.
    """
    av_dir = ensure_local(sft["av_model_id"])
    meta = NLAMeta.from_yaml(Path(av_dir) / "nla_meta.yaml")
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    assert_tokenizer_matches_sidecar(tok, meta)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = _from_pretrained(AutoModelForCausalLM, av_dir, torch.bfloat16,
                            trust_remote_code=True)
    ckpt = Path(sft["_av_ckpt"])
    model = PeftModel.from_pretrained(base, ckpt / "lora", adapter_name="policy",
                                      is_trainable=True)
    model.load_adapter(ckpt / "lora", adapter_name="ref", is_trainable=False)
    model = model.to(dev)
    model.config.use_cache = True  # generation; disabled per-forward below

    blob = torch.load(ckpt / "adapter.pt", map_location="cpu")
    adapter = InjectionAdapter(blob["d_target"], meta.d_model,
                               blob["injection_scale"])
    adapter.load_state_dict(blob["state_dict"])
    adapter = adapter.to(dev)

    ref_adapter = copy.deepcopy(adapter).eval()
    for p in ref_adapter.parameters():
        p.requires_grad_(False)

    return model, adapter, ref_adapter, tok, meta, blob["d_target"]


def load_ar_trainable(sft, dev):
    """AR kept unmerged so its LoRA + head can take the supervised step."""
    ar_dir = ensure_local(sft["ar_model_id"])
    meta = NLAMeta.from_yaml(Path(ar_dir) / "nla_meta.yaml")
    tok = AutoTokenizer.from_pretrained(ar_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ckpt = Path(sft["_ar_ckpt"])
    blob = torch.load(ckpt / "head.pt", map_location="cpu")
    ar = FraudAR(ar_dir, meta, blob["d_target"], head_mode=blob["head_mode"])
    ar.backbone = PeftModel.from_pretrained(ar.backbone, ckpt / "lora",
                                            is_trainable=True)
    ar.out.load_state_dict(blob["out"])
    if blob["value_head"] is not None:
        ar.value_head.load_state_dict(blob["value_head"])
    return ar.to(dev), tok, meta


# ──────────────────────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────────────────────


def per_row_mse_nrm(pred, gold, scale):
    """mse_nrm per row instead of nla_common's batch scalar.

    Both sides are renormalized to L2 == sqrt(d), so the per-element mean is
    exactly 2*(1-cos) — same convention as nla_common.mse_nrm, just unreduced,
    because GRPO needs one reward per rollout.
    """
    return ((normalize_to(pred, scale) - normalize_to(gold, scale)) ** 2).mean(dim=-1)


@torch.no_grad()
def score_explanations(ar, ar_tok, ar_meta, texts, gold, scale, dev, max_len,
                       batch_size):
    ar.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        ids, mask = build_ar_inputs(ar_tok, ar_meta, texts[i:i + batch_size],
                                    max_len=max_len, device=dev)
        outs.append(ar(ids, mask).float())
    pred = torch.cat(outs)
    return per_row_mse_nrm(pred, gold.to(dev), scale)


# ──────────────────────────────────────────────────────────────────────────────
# Rollouts
# ──────────────────────────────────────────────────────────────────────────────


def completion_mask(ids, eos_id, pad_id):
    """1 for real completion tokens up to and including the first EOS, else 0.

    generate() right-pads with pad_token_id, and on Qwen pad == eos, so a plain
    `ids != pad_id` test would also zero out the legitimate terminal EOS. Build
    the mask from the first-EOS position instead.
    """
    B, L = ids.shape
    is_eos = ids == eos_id
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(has_eos, is_eos.float().argmax(dim=1), torch.full_like(has_eos, L, dtype=torch.long))
    pos = torch.arange(L, device=ids.device)[None].expand(B, -1)
    return (pos <= first_eos[:, None]).long()


@torch.no_grad()
def rollout(model, adapter, tok, meta, acts, prompt_ids, inj_pos, dev,
            max_new_tokens, temperature, batch_size, embed_scale):
    """Generate one explanation per row of `acts` (already G-expanded)."""
    model.eval()
    model.set_adapter("policy")
    embed_layer = model.get_input_embeddings()
    seqs = []
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
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
        # inputs_embeds with no input_ids => generate returns ONLY new tokens.
        seqs.append(gen)
    L = max(s.shape[1] for s in seqs)
    padded = [F.pad(s, (0, L - s.shape[1]), value=tok.pad_token_id) for s in seqs]
    return torch.cat(padded, dim=0)


def sequence_logprobs(model, adapter, prompt_ids, inj_pos, acts, gen_ids,
                      gen_mask, embed_scale, adapter_name):
    """Mean per-token logprob of `gen_ids` under the named adapter.

    Rebuilds the same embedding sequence the rollout saw (prompt + injected
    vector + generated tokens) and reads the logits that predict each generated
    token. Position P-1 predicts gen_ids[0], so the slice starts there.
    """
    model.set_adapter(adapter_name)
    embed_layer = model.get_input_embeddings()
    P = prompt_ids.shape[0]
    B, L = gen_ids.shape

    v = adapter(acts)
    embeds = build_injected_embeds(embed_layer, prompt_ids, inj_pos, v,
                                   gen_ids, embed_scale)
    attn = torch.cat([
        torch.ones(B, P, dtype=torch.long, device=gen_ids.device), gen_mask
    ], dim=1)

    logits = model(inputs_embeds=embeds, attention_mask=attn,
                   use_cache=False).logits[:, P - 1:-1]  # [B, L, V]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, gen_ids[..., None]).squeeze(-1)  # [B, L]
    return tok_logp, gen_mask


def masked_mean(x, mask, dim=-1):
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


# ──────────────────────────────────────────────────────────────────────────────
# Eval
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, adapter, av_tok, av_meta, ar, ar_tok, ar_meta, acts,
             prompt_ids, inj_pos, dev, cfg, scale, embed_scale):
    """Greedy round-trip FVE on a fixed held-out set."""
    rl, sft = cfg["rl"], cfg["sft"]
    gen = rollout(model, adapter, av_tok, av_meta, acts, prompt_ids, inj_pos,
                  dev, rl["max_new_tokens"], 0.0, sft.get("gen_batch", 16),
                  embed_scale)
    texts = av_tok.batch_decode(gen, skip_special_tokens=True)
    parsed = [parse_explanation(t) for t in texts]

    ar.eval()
    preds = []
    for i in range(0, len(parsed), sft.get("gen_batch", 16)):
        ids, mask = build_ar_inputs(ar_tok, ar_meta,
                                    parsed[i:i + sft.get("gen_batch", 16)],
                                    max_len=sft.get("ar_max_len", 320), device=dev)
        preds.append(ar(ids, mask).float())
    pred = torch.cat(preds)
    fve = fve_nrm(pred, acts.to(dev), scale)
    return fve, parsed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--pairs", default="data/summaries.parquet")
    ap.add_argument("--av-ckpt", default="checkpoints/av_sft")
    ap.add_argument("--ar-ckpt", default="checkpoints/ar_sft")
    ap.add_argument("--out", default="checkpoints/rl")
    ap.add_argument("--steps", type=int, default=None, help="override rl.total_steps")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="rollouts per fwd/bwd micro-step (memory knob)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 steps, tiny batch — verifies the whole path runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rl, sft, seed = cfg["rl"], cfg["sft"], cfg["seed"]
    sft = dict(sft, _av_ckpt=args.av_ckpt, _ar_ckpt=args.ar_ckpt)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("[warn] no CUDA — this will be unusably slow. Run on the GPU box.")

    B = 2 if args.smoke else rl["batch_size"]
    G = 2 if args.smoke else rl["group_size"]
    total_steps = 3 if args.smoke else (args.steps or rl["total_steps"])
    micro = args.micro_batch

    # ── models ────────────────────────────────────────────────────────────────
    print("[load] AV (policy + ref adapters) …")
    model, adapter, ref_adapter, av_tok, av_meta, d_target = load_av_trainable(sft, dev)
    print("[load] AR …")
    ar, ar_tok, ar_meta = load_ar_trainable(sft, dev)

    prompt_ids_list = build_av_prompt_ids(av_tok, av_meta)
    inj_pos = find_injection_pos(prompt_ids_list, av_meta)
    prompt_ids = torch.tensor(prompt_ids_list, device=dev)
    embed_scale = (math.sqrt(av_meta.d_model)
                   if getattr(model.config, "model_type", "").startswith("gemma")
                   else 1.0)
    scale = target_mse_scale(d_target)
    print(f"[rl] prompt={len(prompt_ids_list)} tok, inj@{inj_pos}, d={d_target}, "
          f"B={B} G={G} rollouts/step={B*G} micro={micro} steps={total_steps}")

    # ── data ──────────────────────────────────────────────────────────────────
    import pandas as pd
    train_df = pd.read_parquet(args.pairs).reset_index(drop=True)
    hold_path = Path(args.av_ckpt) / "holdout.parquet"
    eval_df = pd.read_parquet(hold_path).head(rl["eval_size"]).reset_index(drop=True)
    train_acts = activations_tensor(train_df)
    eval_acts = activations_tensor(eval_df)
    print(f"[data] train={len(train_acts)} eval={len(eval_acts)} "
          f"(eval fraud={eval_df.label.mean():.1%})")

    # ── optimizers ────────────────────────────────────────────────────────────
    model.set_adapter("policy")
    av_lora = [p for n, p in model.named_parameters()
               if p.requires_grad and "policy" in n]
    assert av_lora, "no trainable policy-LoRA params — check adapter names"
    opt_av = torch.optim.AdamW(
        [{"params": adapter.parameters(), "lr": sft["lr_adapter"] * 0.1},
         {"params": av_lora, "lr": rl["lr_av"]}], weight_decay=0.0)

    ar_lora = [p for p in ar.backbone.parameters() if p.requires_grad]
    opt_ar = torch.optim.AdamW(
        [{"params": ar.trainable_head_parameters(), "lr": rl["lr_ar"]},
         {"params": ar_lora, "lr": rl["lr_ar"] * 0.1}], weight_decay=0.0)

    beta = rl["kl_beta"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = Path("reports/phase5_samples.jsonl")
    samples_path.parent.mkdir(exist_ok=True)
    log, t0 = [], time.time()
    rng = np.random.default_rng(seed)

    for step in range(1, total_steps + 1):
        # ── 1. sample activations, G-expand ───────────────────────────────────
        idx = rng.choice(len(train_acts), size=B, replace=False)
        h = train_acts[idx]                                   # [B, d]
        h_rep = h.repeat_interleave(G, dim=0)                 # [B*G, d]

        # ── 2. rollout ────────────────────────────────────────────────────────
        gen_ids = rollout(model, adapter, av_tok, av_meta, h_rep, prompt_ids,
                          inj_pos, dev, rl["max_new_tokens"], rl["temperature"],
                          sft.get("gen_batch", 16), embed_scale)
        gen_mask = completion_mask(gen_ids, av_tok.eos_token_id, av_tok.pad_token_id)
        texts = av_tok.batch_decode(gen_ids, skip_special_tokens=True)
        parsed = [parse_explanation(t) for t in texts]

        # ── 3. reward ─────────────────────────────────────────────────────────
        mse = score_explanations(ar, ar_tok, ar_meta, parsed, h_rep, scale, dev,
                                 sft.get("ar_max_len", 320), sft.get("gen_batch", 16))
        reward = -mse                                          # [B*G]

        # group-normalized advantages
        r_g = reward.view(B, G)
        adv = ((r_g - r_g.mean(dim=1, keepdim=True))
               / (r_g.std(dim=1, keepdim=True) + 1e-4)).view(-1).detach()

        # ── 4. AR supervised step on the sampled (z, h) pairs ─────────────────
        ar.train()
        opt_ar.zero_grad(set_to_none=True)
        n_micro = math.ceil(len(parsed) / micro)
        ar_loss_sum = 0.0
        for i in range(0, len(parsed), micro):
            ids, mask = build_ar_inputs(ar_tok, ar_meta, parsed[i:i + micro],
                                        max_len=sft.get("ar_max_len", 320), device=dev)
            pred = ar(ids, mask)
            loss = per_row_mse_nrm(pred, h_rep[i:i + micro].to(dev), scale).mean()
            (loss / n_micro).backward()
            ar_loss_sum += float(loss) / n_micro
        torch.nn.utils.clip_grad_norm_(
            ar.trainable_head_parameters() + ar_lora, 1.0)
        opt_ar.step()

        # ── 5. AV GRPO step ───────────────────────────────────────────────────
        model.train()
        opt_av.zero_grad(set_to_none=True)
        pg_sum = kl_sum = 0.0
        for i in range(0, len(gen_ids), micro):
            sl = slice(i, i + micro)
            a_mb = h_rep[sl].to(dev)
            ids_mb, mask_mb = gen_ids[sl], gen_mask[sl]

            with torch.no_grad():
                ref_logp, _ = sequence_logprobs(model, ref_adapter, prompt_ids,
                                                inj_pos, a_mb, ids_mb, mask_mb,
                                                embed_scale, "ref")
            pol_logp, _ = sequence_logprobs(model, adapter, prompt_ids, inj_pos,
                                            a_mb, ids_mb, mask_mb, embed_scale,
                                            "policy")

            # k3 estimator (Schulman): unbiased, non-negative, low variance.
            d = ref_logp - pol_logp
            kl_tok = torch.exp(d) - d - 1.0

            # Length-normalized so the policy is not rewarded for padding out
            # tokens; mean explanation length is logged to catch collapse.
            seq_logp = masked_mean(pol_logp, mask_mb)
            kl = masked_mean(kl_tok, mask_mb)

            pg = -(adv[sl] * seq_logp).mean()
            loss = pg + beta * kl.mean()
            (loss * (len(ids_mb) / len(gen_ids))).backward()
            pg_sum += float(pg) * len(ids_mb) / len(gen_ids)
            kl_sum += float(kl.mean()) * len(ids_mb) / len(gen_ids)

        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + av_lora, 1.0)
        opt_av.step()
        model.set_adapter("policy")

        # ── logging ───────────────────────────────────────────────────────────
        gen_len = float(gen_mask.sum(1).float().mean())
        rec = {"step": step, "reward": float(reward.mean()),
               "mse_nrm": float(mse.mean()), "ar_loss": ar_loss_sum,
               "pg": pg_sum, "kl": kl_sum, "gen_len": round(gen_len, 1),
               "elapsed_s": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"[{step}/{total_steps}] r={rec['reward']:+.4f} "
              f"mse={rec['mse_nrm']:.4f} ar={rec['ar_loss']:.4f} "
              f"kl={rec['kl']:.4f} len={rec['gen_len']:.0f} "
              f"({rec['elapsed_s']:.0f}s)")

        # Dump explanations continuously — this is the artifact worth reading.
        with samples_path.open("a") as f:
            for k in range(min(3, len(parsed))):
                f.write(json.dumps({
                    "step": step, "reward": float(reward[k]),
                    "mse_nrm": float(mse[k]), "explanation": parsed[k],
                }) + "\n")
        if step % 10 == 0 or args.smoke:
            print(f"    sample: {parsed[0][:220]!r}")

        # ── periodic held-out eval ────────────────────────────────────────────
        if step % rl["eval_every"] == 0 or step == total_steps:
            fve, ex = evaluate(model, adapter, av_tok, av_meta, ar, ar_tok,
                               ar_meta, eval_acts, prompt_ids, inj_pos, dev,
                               cfg, scale, embed_scale)
            rec["heldout_fve"] = fve
            print(f"    >> held-out FVE = {fve:.4f}")
            for e in ex[:3]:
                print(f"    >> {e[:200]!r}")

        if step % rl["checkpoint_every"] == 0 or step == total_steps:
            ck = out_dir / f"step{step}"
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck / "av_lora", selected_adapters=["policy"])
            torch.save({"state_dict": adapter.state_dict(), "d_target": d_target,
                        "injection_scale": av_meta.injection_scale},
                       ck / "adapter.pt")
            torch.save({"out": ar.out.state_dict(),
                        "value_head": (ar.value_head.state_dict()
                                       if hasattr(ar.value_head, "weight") else None),
                        "d_target": d_target, "head_mode": ar.head_mode},
                       ck / "ar_head.pt")
            ar.backbone.save_pretrained(ck / "ar_lora")
            print(f"    >> checkpointed to {ck}")

        write_report("reports/phase5_rl.json", {
            "steps_done": step, "total_steps": total_steps,
            "batch": B, "group": G, "kl_beta": beta,
            "log": log, "wall_clock_s": round(time.time() - t0, 1),
        })

    print(f"[rl] done in {time.time() - t0:.0f}s — samples in {samples_path}")


if __name__ == "__main__":
    main()
