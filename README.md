# NLA-Fraud

Applying Anthropic's [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) (NLA) to a tabular payments-fraud DNN.

The NLA translates a shallow MLP's 128-dim hidden activations into readable English explanations and back, using fine-tuned Qwen2.5-7B checkpoints (`kitft/nla-models`).

> **Start here.**
> - **[docs/RUNBOOK.md](docs/RUNBOOK.md)** — how to actually run this. The work
>   happens on a remote A100, *not* on the laptop; read §1 before running
>   anything.
> - **[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)** — current findings, what is
>   broken, and what to fix next, with evidence.
>
> Quick orientation: `./scripts/vast.sh status`

## Setup

```bash
uv sync
```

Set credentials as needed:

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
export ANTHROPIC_API_KEY=...
```

## Running each phase

### Phase 1 — Data

```bash
# Real IEEE-CIS fraud (requires accepting competition rules in browser once)
uv run python src/get_data.py --dataset ieee-fraud-detection

# Fully public Sparkov synthetic (no rules acceptance needed — good for pipeline debug)
uv run python src/get_data.py --dataset kartik2112/fraud-detection

# Offline synthetic (no Kaggle account needed)
uv run python src/get_data.py --dataset synthetic --limit 100000
```

Outputs: `data/transactions.parquet`, `reports/phase1_report.json`

### Phase 2 — Target model + activation corpus

```bash
uv run python src/target_model.py
uv run python src/collect_activations.py
```

Outputs: `data/fraud_mlp.pt`, `data/activations_l2.parquet`, `reports/activation_report.json`

### Phase 3 — Warm-start summaries

```bash
uv run python src/serialize.py        # smoke-test serialization
uv run python src/gen_summaries.py --limit 10000   # validate pipeline
uv run python src/gen_summaries.py    # full 100k run
```

### Phase 4 — SFT

Needs a GPU (see "Compute" below). Smoke-test the whole path first — it runs in
seconds and catches template/injection/padding errors before you burn GPU hours:

```bash
uv run python src/sft_av.py --limit 64 --smoke
uv run python src/sft_ar.py --limit 64 --smoke
```

Then the real runs, AR first (the round-trip eval needs both):

```bash
uv run python src/sft_ar.py                        # -> checkpoints/ar_sft
uv run python src/sft_av.py                        # -> checkpoints/av_sft
uv run python src/roundtrip_eval.py --limit 1000   # Phase 4 exit criteria
```

`roundtrip_eval.py` writes `reports/phase4_roundtrip_sft.json`: FVE, cosine
distribution, the fraud-slice FVE broken out separately, and 20 sample
explanations for the eyeball check the plan requires before RL.

**Conventions taken from the released checkpoints** (`src/nla_common.py`, each
cited in-code): prompt template, injection token id + neighbour check, and
`injection_scale=150` all come from `nla_meta.yaml` — nothing is hardcoded. The
AR is the truncated 21-layer backbone with `lm_head` and the final LayerNorm
replaced by `Identity`, extracting at the last token, per `docs/inference.md`.

One deliberate deviation: `mse_scale`. The sidecar ships `sqrt(3584)=59.87`,
correct for the LLM's residual stream, but our reconstruction target is the
fraud MLP's 128-d space, so we use `sqrt(128)=11.31`. That is what makes
`mse_nrm == 2*(1-cos)` and keeps FVE comparable to the paper's numbers.

### Phase 5 — RL / GRPO

```bash
uv run python src/rl_grpo.py
```

### Phase 6 — Evaluations

```bash
uv run python src/evals.py
uv run python src/steering.py
```

### Phase 7 — Baselines

```bash
uv run python src/baselines.py
```

### Inspect a single transaction

```bash
uv run python src/inspect_nla.py --row 42
uv run python src/inspect_nla.py --json '{"amt": 1.02, "category": "misc_net", ...}'
```

## Config

All hyperparameters live in `configs/experiment.yaml`. Every script reads it.

## Key design choices

- MLP trained only on **verbalizable** features (no V-columns from IEEE-CIS).
- Causal per-card velocity/z-score features computed in time order to prevent leakage.
- NLA checkpoints initialized from `kitft/nla-qwen2.5-7b-L20-{av,ar}` (Qwen2.5-7B, layer 20, d_model=3584).
- A learned `nn.Linear(128, 3584)` adapter bridges the MLP's activation space to the NLA's embedding space.
- Single-GPU training via HF `transformers` + `peft` LoRA; no distributed stack required.
