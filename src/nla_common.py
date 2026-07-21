"""Shared plumbing for Phase 4/5: sidecar loading, injection, normalization, FVE.

Every NLA-specific convention here is lifted from the released checkpoints and
the reference repo rather than invented. Sources cited per-function:
  - kitft/natural_language_autoencoders docs/inference.md ("The inference recipe",
    "Optional: the AR", "Computing FVE — two classic footguns")
  - nla_inference.py::NLACritic (AR loading: strip lm_head + final LN)
  - the nla_meta.yaml sidecar shipped in each HF checkpoint

NEVER hardcode token ids, prompt templates, or scale factors. They come from
the sidecar. See load_nla_meta().
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

# docs/inference.md: "Final LayerNorm replaced with Identity" — the value head
# sees raw residual-stream output of block K. Attr name varies by architecture.
_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


def _from_pretrained(cls, path, dtype, **kw):
    """transformers renamed torch_dtype -> dtype in 4.56. Support both."""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class NLAMeta:
    """Parsed nla_meta.yaml. The contract with the released checkpoint."""

    role: str
    d_model: int
    injection_scale: float | None
    mse_scale: float | None
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    av_template: str
    ar_template: str
    extraction_layer_index: int
    raw: dict

    @classmethod
    def from_hub(cls, model_id: str) -> "NLAMeta":
        path = hf_hub_download(model_id, "nla_meta.yaml")
        return cls.from_yaml(Path(path))

    @classmethod
    def from_yaml(cls, path: Path) -> "NLAMeta":
        m = yaml.safe_load(Path(path).read_text())
        tok, ex, pt = m["tokens"], m["extraction"], m["prompt_templates"]
        return cls(
            role=m["role"],
            d_model=int(m["d_model"]),
            injection_scale=(None if ex.get("injection_scale") is None
                             else float(ex["injection_scale"])),
            mse_scale=(None if ex.get("mse_scale") is None
                       else float(ex["mse_scale"])),
            injection_char=tok["injection_char"],
            injection_token_id=int(tok["injection_token_id"]),
            injection_left_neighbor_id=int(tok["injection_left_neighbor_id"]),
            injection_right_neighbor_id=int(tok["injection_right_neighbor_id"]),
            av_template=pt["av"],
            ar_template=pt.get("ar") or pt["critic"],
            extraction_layer_index=int(m["extraction_layer_index"]),
            raw=m,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────
#
# docs/inference.md §"mse_scale vs injection_scale": mse_scale = sqrt(d) makes
# `.mean()` return 2*(1-cos), d-agnostic, range [0,4].
#
#     per-element MSE = 2*s^2*(1-cos)/d   ->   s = sqrt(d)  =>  s^2/d = 1
#
# The released sidecar's mse_scale is sqrt(3584)=59.87, correct for the *LLM's*
# 3584-dim residual stream. Our reconstruction target lives in the fraud MLP's
# 128-dim activation space, so the d-agnostic constant for OUR loss is
# sqrt(128)=11.31. Using 59.87 on 128-dim vectors is not "wrong" (the scale
# cancels in cos; it is only a gradient-magnitude knob) but it would make the
# reported MSE 56*(1-cos) instead of 2*(1-cos) and break comparability with the
# paper's numbers. We therefore derive mse_scale from the TARGET dim.


def target_mse_scale(d_target: int) -> float:
    return math.sqrt(d_target)


def normalize_to(x: torch.Tensor, scale: float, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along the last dim, then rescale to `scale`."""
    return x / (x.norm(dim=-1, keepdim=True) + eps) * scale


def mse_nrm(pred: torch.Tensor, gold: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalized MSE = 2*(1-cos). docs/inference.md §"Computing MSE"."""
    return ((normalize_to(pred, scale) - normalize_to(gold, scale)) ** 2).mean()


def fve_nrm(pred: torch.Tensor, gold: torch.Tensor, scale: float) -> float:
    """Fraction of variance explained, matching the released `fve_nrm` metric.

    docs/inference.md §"Computing FVE — two classic footguns":
      1. normalize the AR's raw output by hand;
      2. do NOT normalize mu — the mean of unit-sphere vectors lies *inside*
         the sphere, and re-projecting it inflates the denominator and thus FVE.
    """
    gold_n = normalize_to(gold, scale)
    pred_n = normalize_to(pred, scale)
    mu = gold_n.mean(dim=0)  # deliberately NOT normalized
    num = ((pred_n - gold_n) ** 2).mean()
    den = ((gold_n - mu) ** 2).mean()
    return float(1.0 - num / den)


def cosine(pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(pred.float(), gold.float(), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# AV: prompt construction + injection
# ──────────────────────────────────────────────────────────────────────────────


def build_av_prompt_ids(tokenizer, meta: NLAMeta) -> list[int]:
    """Tokenize the sidecar's AV template. docs/inference.md step 1.

    One-step `tokenize=True` — the two-step path re-adds BOS on Gemma/Llama and
    shifts every position by one. Qwen has no BOS so it happens to work there,
    which is exactly what makes the bug easy to miss.
    """
    content = meta.av_template.format(injection_char=meta.injection_char)
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # transformers <5 returns a flat list of ids; >=5 returns a BatchEncoding
    # whose input_ids may be batched. Normalize to a flat list of ints.
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def find_injection_pos(prompt_ids: list[int], meta: NLAMeta) -> int:
    """Scan for the injection token, verify neighbors. docs/inference.md step 3.

    The neighbor check is mandatory: the injection char is rare but not unique,
    and the `<concept>`/`</concept>` angle brackets are pinned in the sidecar.
    """
    for p, t in enumerate(prompt_ids):
        if t != meta.injection_token_id:
            continue
        if p == 0 or p == len(prompt_ids) - 1:
            continue
        if (prompt_ids[p - 1] == meta.injection_left_neighbor_id
                and prompt_ids[p + 1] == meta.injection_right_neighbor_id):
            return p
    raise AssertionError(
        f"injection token {meta.injection_token_id} with neighbors "
        f"({meta.injection_left_neighbor_id}, {meta.injection_right_neighbor_id}) "
        f"not found in prompt of {len(prompt_ids)} tokens — template drift?"
    )


def assert_tokenizer_matches_sidecar(tokenizer, meta: NLAMeta) -> None:
    """Catch template drift / double-BOS before the first forward pass.

    Mirrors nla_inference.py::load_nla_config, which asserts against the live
    tokenizer at startup.
    """
    # Encode, don't convert_tokens_to_ids: the injection char is a plain BPE
    # character, not an added token, so convert_tokens_to_ids returns None for
    # the raw glyph (it wants the byte-level surface form).
    got = tokenizer.encode(meta.injection_char, add_special_tokens=False)
    assert got == [meta.injection_token_id], (
        f"tokenizer encodes {meta.injection_char!r} -> {got}, sidecar says "
        f"[{meta.injection_token_id}]. Wrong tokenizer for this checkpoint."
    )
    ids = build_av_prompt_ids(tokenizer, meta)
    find_injection_pos(ids, meta)  # raises on drift


class InjectionAdapter(nn.Module):
    """Linear(d_target -> d_model), output renormalized to injection_scale.

    The released AV injects the raw vector scaled to a fixed L2 norm. The repo
    README notes a learned affine `W·v + b` is a trainer-side-only change — that
    adapter is the only new parameter the AV needs to accept a new activation
    domain. Renormalizing the OUTPUT to injection_scale (rather than trusting
    the linear map to land there) keeps the injected vector inside the norm band
    the AV was trained on for the whole of training, not just at init.

    docs/inference.md: "injection_scale is mandatory. The model was trained with
    vectors at this exact L2-norm. Raw-magnitude vectors are out-of-distribution
    and output degrades badly."
    """

    def __init__(self, d_target: int, d_model: int, injection_scale: float,
                 init_std: float = 0.02):
        super().__init__()
        assert injection_scale is not None, "AV sidecar must carry injection_scale"
        self.proj = nn.Linear(d_target, d_model)
        nn.init.normal_(self.proj.weight, std=init_std)
        nn.init.zeros_(self.proj.bias)
        self.injection_scale = float(injection_scale)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 regardless of the base model's dtype; the caller casts.
        return normalize_to(self.proj(v.float()), self.injection_scale)


def build_injected_embeds(
    embed_layer: nn.Module,
    prompt_ids: torch.Tensor,   # [T] — identical for every row (fixed template)
    inj_pos: int,
    vectors: torch.Tensor,      # [B, d_model] — already scaled by the adapter
    target_ids: torch.Tensor | None = None,  # [B, L] right-padded, or None
    embed_scale: float = 1.0,
) -> torch.Tensor:
    """Splice the activation into the prompt's embedding sequence.

    Built with torch.cat rather than in-place index assignment so autograd flows
    cleanly back into the adapter.

    embed_scale is 1.0 for Qwen/Llama/Mistral and sqrt(hidden_size) for Gemma-3
    (whose ScaledWordEmbedding.forward multiplies by sqrt(d); a raw weight lookup
    bypasses it). docs/inference.md step 2.
    """
    B = vectors.shape[0]
    prompt_embeds = embed_layer(prompt_ids)[None].expand(B, -1, -1) * embed_scale
    v = vectors.to(prompt_embeds.dtype)[:, None, :]  # [B, 1, d]

    parts = [prompt_embeds[:, :inj_pos], v, prompt_embeds[:, inj_pos + 1:]]
    if target_ids is not None:
        parts.append(embed_layer(target_ids) * embed_scale)
    return torch.cat(parts, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# AR: truncated backbone + value head + new output map
# ──────────────────────────────────────────────────────────────────────────────


class FraudAR(nn.Module):
    """AR reconstructor retargeted from the LLM's 3584-dim space to our 128-dim.

    Architecture (docs/inference.md §"Optional: the AR"):
      - first K+1 layers only — config.json already ships truncated
        (num_hidden_layers=21 for K=20), no on-the-fly truncation needed;
      - lm_head -> Identity (the AR never emits logits);
      - final LayerNorm -> Identity (value head sees raw block-K output);
      - value_head = Linear(d_model, d_model, bias=False), loaded from
        value_head.safetensors;
      - extraction at the LAST token (template ends in a fixed `</text> <summary>`
        suffix, so tokens[-1] is stable — no scan required).

    We keep the pretrained value_head and stack a new Linear(d_model, d_target)
    on top of it (`head_mode="stack"`), so the pretrained residual->activation
    mapping is retained and only the final projection is learned from scratch.
    `head_mode="replace"` drops value_head and maps d_model -> d_target directly;
    exposed as a config switch because the plan permits either.
    """

    def __init__(self, checkpoint_dir: str, meta: NLAMeta, d_target: int,
                 head_mode: str = "stack", dtype=torch.bfloat16):
        super().__init__()
        assert meta.role in ("ar", "critic"), f"expected AR sidecar, got {meta.role!r}"
        assert head_mode in ("stack", "replace")
        self.head_mode = head_mode
        self.d_target = d_target

        backbone = _from_pretrained(
            AutoModelForCausalLM, checkpoint_dir, dtype, trust_remote_code=True
        )
        backbone.lm_head = nn.Identity()
        inner = backbone.model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attr among {_FINAL_LN_ATTRS} on {type(inner).__name__}"
            )
        self.backbone = backbone

        if head_mode == "stack":
            vh = nn.Linear(meta.d_model, meta.d_model, bias=False)
            sd = load_safetensors(
                hf_hub_download(checkpoint_dir, "value_head.safetensors")
                if not Path(checkpoint_dir).exists()
                else str(Path(checkpoint_dir) / "value_head.safetensors")
            )
            key = "weight" if "weight" in sd else next(iter(sd))
            vh.weight.data = sd[key].float()
            self.value_head = vh
            self.out = nn.Linear(meta.d_model, d_target)
        else:
            self.value_head = nn.Identity()
            self.out = nn.Linear(meta.d_model, d_target)
        # Small init so the head starts near zero and the MSE loss does not
        # explode on step 0.
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> torch.Tensor:
        """-> [B, d_target] raw (unnormalized) reconstruction."""
        # Ask for hidden_states rather than reaching through `.model`: once PEFT
        # wraps the backbone, `.model` resolves to the LoRA-wrapped CausalLM and
        # returns logits, not a last_hidden_state. hidden_states[-1] is the
        # post-final-norm output — and we replaced that norm with Identity, so
        # it IS the raw block-K residual stream the value head expects.
        hs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]  # [B, T, d_model]
        # Extract at the last *real* token. build_ar_inputs LEFT-pads, so real
        # tokens occupy the tail and index -1 is correct for every row. Assert
        # it rather than deriving from the mask: with left padding
        # `mask.sum(1)-1` points at a PAD position for every short row, which
        # silently extracts garbage instead of the `</text> <summary>` suffix.
        assert attention_mask[:, -1].all(), (
            "AR batch is not left-padded: last column contains padding. "
            "build_ar_inputs must set padding_side='left'."
        )
        h = hs[:, -1].float()  # [B, d_model]
        return self.out(self.value_head(h))

    def trainable_head_parameters(self):
        ps = list(self.out.parameters())
        if isinstance(self.value_head, nn.Linear):
            ps += list(self.value_head.parameters())
        return ps


def build_ar_inputs(tokenizer, meta: NLAMeta, explanations: list[str],
                    max_len: int = 320, device="cuda"):
    """Tokenize the AR template around each explanation.

    add_special_tokens=True: nla_inference.py::NLACritic notes the AR was trained
    WITH the BOS prefix (Gemma/Llama); Qwen has bos_token=None so it is a no-op.
    Dropping it shifts position 0 and degrades every reconstruction.

    LEFT padding so the final token of each row sits at index -1, matching the
    "extract at last token, no scan" convention.
    """
    prompts = [meta.ar_template.format(explanation=e) for e in explanations]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts, add_special_tokens=True, return_tensors="pt",
        padding=True, truncation=True, max_length=max_len,
    )
    tokenizer.padding_side = old_side
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────


EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"


def wrap_explanation(summary: str) -> str:
    """The released AV emits `<explanation>...</explanation>` and the AR template
    is applied to the *parsed* contents. Training targets keep the tags so the
    AV stays in-distribution and Phase 5/6 parsing works unchanged.
    """
    s = summary.strip()
    if s.startswith(EXPLANATION_OPEN):
        return s
    return f"{EXPLANATION_OPEN}\n{s}\n{EXPLANATION_CLOSE}"


def parse_explanation(text: str) -> str:
    """Inverse of wrap_explanation, tolerant of a missing close tag (truncation)."""
    import re
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    if m:
        return m.group(1)
    if EXPLANATION_OPEN in text:
        return text.split(EXPLANATION_OPEN, 1)[1].strip()
    return text.strip()


def stratified_indices(df, n: int, fraud_fraction: float | None, seed: int,
                       score_col: str = "fraud_score", label_col: str = "label"
                       ) -> np.ndarray:
    """Rebalance the natural ~3.5% positive rate for SFT sampling.

    The corpus is ~365 fraud : ~9.6k legit. Sampling at the natural rate gives
    the AV ~25x less gradient signal on risk-flavored activations than on benign
    ones, so it learns to write fluent "nothing unusual here" text and stays
    vague on exactly the cases we care about. We oversample the positive slice.

    fraud_fraction=None keeps the natural rate (set it that way for held-out
    eval sets you want to be distribution-faithful).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    if fraud_fraction is None:
        return rng.permutation(idx)[:n]

    pos = idx[df[label_col].values == 1]
    neg = idx[df[label_col].values == 0]
    n_pos = min(int(round(n * fraud_fraction)), len(pos) * 8)  # cap resampling
    n_neg = n - n_pos
    # replace=True on the positive slice: there are only ~365 of them and we
    # want a 30% share, so they necessarily repeat. Duplicates are acceptable
    # here (they are distinct activation/summary pairs seen more often), but the
    # repeat factor is capped above to avoid degenerate memorization.
    take_pos = rng.choice(pos, size=min(n_pos, len(pos) * 8), replace=n_pos > len(pos))
    take_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    out = np.concatenate([take_pos, take_neg])
    return rng.permutation(out)


def load_pairs(parquet_path: str, n: int | None, fraud_fraction: float | None,
               seed: int, holdout: int = 0):
    """Load (activation, summary) pairs, returning (train_df, holdout_df).

    The holdout is carved at the NATURAL class rate and disjoint from train, so
    Phase-4 FVE is reported on a distribution-faithful sample.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    hold_idx = perm[:holdout]
    pool_idx = perm[holdout:]
    pool = df.iloc[pool_idx].reset_index(drop=True)

    n = len(pool) if n is None else min(n, len(pool) * 4)
    take = stratified_indices(pool, n, fraud_fraction, seed)
    return pool.iloc[take].reset_index(drop=True), df.iloc[hold_idx].reset_index(drop=True)


def activations_tensor(df, col: str = "activation_vector") -> torch.Tensor:
    return torch.tensor(np.stack(df[col].values), dtype=torch.float32)


def ensure_local(model_id: str) -> str:
    """Resolve a HF id to a local snapshot dir (no-op for existing paths)."""
    if Path(model_id).exists():
        return model_id
    return snapshot_download(model_id)


def write_report(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
    print(f"[report] wrote {path}")
