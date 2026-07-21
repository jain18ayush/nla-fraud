# NLA-Fraud Runbook

How to operate this pipeline end to end. Read this before running anything.

For *what to fix and why*, see [IMPROVEMENTS.md](IMPROVEMENTS.md). For the
original research spec, see [../plan.md](../plan.md).

---

## 1. The two-machine model — read this first

This project runs on **two machines**, and confusing them has already cost real
time:

| | Laptop (`~/Development/Research/NLA/nla-fraud`) | Vast.ai box (`vast:/root/nla-fraud`) |
|---|---|---|
| role | edit code, read reports | **run everything** |
| GPU | none (Apple Silicon, MPS only) | **A100-SXM4-80GB** |
| git | *not* a git repo | **the git repo — source of truth** |
| base models | not present | cached in `$HF_HOME=/workspace/.hf_home` |
| `data/` | full: transactions + activations (~1.8 GB) | only `summaries.parquet` |
| `checkpoints/` | stale copies | **authoritative** |

**Never try to run the 7B models locally.** The base weights are not on the
laptop, and downloading them (~26 GB) will nearly fill the disk. A previous
session did exactly this; the download was killed.

Because `data/`, `reports/`, `checkpoints/` and `logs/` are all gitignored,
**git does not sync artifacts between the two machines.** Code moves by
`vast.sh sync`; results come back by `vast.sh pull`.

---

## 2. Everything goes through `scripts/vast.sh`

```bash
./scripts/vast.sh status            # GPU, running jobs, disk, git state
./scripts/vast.sh sync              # push src/ configs/ scripts/ to the box
./scripts/vast.sh run rl_grpo.py --steps 300    # nohup; survives disconnect
./scripts/vast.sh fg roundtrip_eval.py --limit 8   # foreground, for quick checks
./scripts/vast.sh watch rl_grpo     # tail that job's log
./scripts/vast.sh logs              # list logs, newest first
./scripts/vast.sh pull              # bring reports/*.json back to the laptop
./scripts/vast.sh stop rl_grpo      # kill a job
./scripts/vast.sh shell             # interactive ssh
```

The wrapper exists to hide two things that will otherwise bite you:

1. **The box prints a port-forward warning on every SSH connection**
   (`bind [127.0.0.1]:8080: Address already in use`). It is harmless. `vast.sh`
   filters it but does not silence real errors.
2. **There is no bare `python` on the box** — only `/root/nla-fraud/.venv/bin/python`.
   Any command you write by hand must use that path or `uv run`.

`scripts/run_phase5.sh` is a Phase-5-specific convenience with one extra mode,
`samples`, which pretty-prints generated explanations as they stream in.

---

## 3. Current state

| phase | script | status |
|---|---|---|
| 1 — data | `get_data.py` | done (IEEE-CIS, 590 540 rows, 3.5% fraud) |
| 2 — target MLP + activations | `target_model.py`, `collect_activations.py` | done |
| 3 — warm-start summaries | `serialize.py`, `gen_summaries.py` | **10 k of a planned 100 k** |
| 4 — SFT | `sft_ar.py`, `sft_av.py` | done; round-trip FVE **0.429** |
| 4 — round-trip eval | `roundtrip_eval.py` | done |
| 5 — RL / GRPO | `rl_grpo.py` | **written, never run** |
| 6 — evals | `evals.py`, `steering.py` | **not written** |
| 7 — baselines | `baselines.py` | **not written** (swap control done, lives in `roundtrip_eval.py --swap`) |
| — | `inspect_nla.py` | **not written** |

Validity gates passed so far:

- **Activation-swap control: PASS.** real FVE +0.431, swap −0.867, floor −0.340,
  text Jaccard 0.184. The AV genuinely reads the activation.
- **Injection-adapter geometry: partial concern.** Geometry preserved
  (Spearman 0.943) but effective rank drops 8.13 → 3.00. See IMPROVEMENTS.md.

---

## 4. Phase-by-phase

Phases 1–3 have already produced their artifacts. You only need to re-run them
if you are changing the data or scaling the summary corpus.

### Phase 1 — data (laptop or box)

```bash
uv run python src/get_data.py --dataset ieee-fraud-detection
```

Needs `KAGGLE_USERNAME`/`KAGGLE_KEY`, and one-time acceptance of the competition
rules in a browser (the script detects the 403 and tells you). Writes
`data/transactions.parquet` + `reports/phase1_report.json`.

### Phase 2 — target MLP + activation corpus

```bash
./scripts/vast.sh run target_model.py
./scripts/vast.sh run collect_activations.py
```

Writes `data/fraud_mlp.pt`, `data/activations_l{2,3}.parquet`,
`reports/activation_report.json`. **The activation parquets are ~900 MB each and
currently exist only on the laptop.** Getting them onto the box is a
prerequisite for regenerating summaries there — plan for it (29 GB free).

### Phase 3 — warm-start summaries

```bash
./scripts/vast.sh run gen_summaries.py --limit 10000     # validate
./scripts/vast.sh run gen_summaries.py --full            # 100k
```

Needs `OPENROUTER_API_KEY`. Fully resumable — progress is journalled to JSONL,
so a crash or credit stop loses nothing; just re-run.

`--balance stratified` (the default) spreads the API budget across
(label × fraud-score-bin) cells instead of sampling iid, and attaches a
`sample_weight` column so any metric can be reweighted back to the natural
distribution. At the natural 3.5% rate an iid sample spends ~96% of the budget
on benign rows and yields almost nothing in the model-error cells that Phase 7
depends on. Use `--balance natural` to opt out.

> The existing `data/summaries.parquet` (10 k rows, 3.51% fraud) predates this
> and has **no** `sample_weight` column. Regenerating will add it.

### Phase 4 — SFT

Always smoke-test first; it catches template/injection/padding errors in seconds
rather than GPU-hours.

```bash
./scripts/vast.sh fg  sft_ar.py --limit 64 --smoke
./scripts/vast.sh fg  sft_av.py --limit 64 --smoke
./scripts/vast.sh run sft_ar.py          # AR first — round-trip eval needs both
./scripts/vast.sh run sft_av.py
./scripts/vast.sh run roundtrip_eval.py --limit 1000
```

### Phase 4b — validity gates

```bash
./scripts/vast.sh run roundtrip_eval.py --swap --limit 500   # swap control
./scripts/vast.sh fg  diagnose_adapter.py                    # adapter geometry (CPU)
```

The swap control is a **hard gate**: if swap FVE sits near real FVE, the
explanations are prompt-prior artifacts and every downstream number is
meaningless. Re-run it after any change to the AV, the adapter, or the corpus.

### Phase 5 — RL / GRPO

```bash
./scripts/run_phase5.sh smoke        # 3 steps, B=2 G=2
./scripts/run_phase5.sh run          # full, from configs/experiment.yaml
./scripts/run_phase5.sh run 300      # capped at 300 steps
./scripts/run_phase5.sh samples      # stream explanations as they generate
./scripts/run_phase5.sh watch        # tail the training log
```

Explanations stream continuously to `reports/phase5_samples.jsonl`; per-step
metrics go to `reports/phase5_rl.json`; checkpoints land in `checkpoints/rl/stepN/`.

Design notes that matter if you modify it:

- The **KL reference is the SFT checkpoint, not the pretrained base.** It is
  obtained by loading the SFT LoRA twice over shared base weights as PEFT
  adapters `policy` (trainable) and `ref` (frozen), switched with
  `set_adapter()`. This avoids a second 15 GB copy of the 7B.
- Reward is **per-row** `-mse_nrm`; `nla_common.mse_nrm` reduces to a batch
  scalar, which GRPO cannot use.
- Log-probs are **length-normalized**, so the policy is not rewarded for padding.
  `gen_len` is logged every step — watch it for collapse in either direction.
- `--micro-batch` is the memory knob. `B × G` rollouts are generated per step
  and back-propagated in micro-batches.

---

## 5. Config

Everything lives in `configs/experiment.yaml` and every script reads it. Keys
worth knowing:

| key | meaning |
|---|---|
| `activations.hook_layer` | `l2` (default) or `l3`. l2 is further from the logit and carries richer structure. |
| `summaries.natural_frac` | fraction of the summary budget sampled iid vs cell-balanced |
| `sft.fraud_fraction` | batch-level fraud oversampling during SFT (0.30) |
| `sft.n_holdout` | held-out rows carved at the **natural** class rate |
| `sft.gen_batch` | AV generation batch size |
| `rl.kl_beta` | KL penalty toward the SFT init |
| `rl.group_size` | G — explanations sampled per activation for GRPO |

Two scale constants that are easy to get wrong and are documented in
`nla_common.py`:

- **`injection_scale = 150`** — a property of the AV's 3584-d embedding space,
  applied *after* the adapter projects up. It is **correct as-is** and does not
  need adjusting for our 128-d activations.
- **`mse_scale = sqrt(128)`**, not the sidecar's `sqrt(3584)` — because the
  reconstruction target is the fraud MLP's space. This is what makes
  `mse_nrm == 2(1−cos)` and keeps FVE comparable to the paper.

---

## 6. Gotchas

- **`max_new_tokens` must be ≥ 180.** At 120, 75% of explanations were truncated
  before the closing tag, costing ~0.02 FVE.
- **`pad_token_id == eos_token_id` on Qwen.** A naive `ids != pad` completion
  mask silently zeroes the legitimate terminal EOS. `rl_grpo.completion_mask`
  builds the mask from the first-EOS position instead; it is unit-tested.
- **The AR requires left-padding.** `FraudAR.forward` asserts it — with right
  padding, index −1 lands on PAD for short rows and extracts garbage.
- **`generate()` with `inputs_embeds` returns only new tokens**, not the prompt.
  Do not strip a prompt slice.
- **Don't merge LoRA when you need gradients.** `roundtrip_eval.load_av` calls
  `merge_and_unload()` because it is inference-only; `rl_grpo` deliberately does
  not.
- **Long runs need `nohup`** or they die with the SSH session. `vast.sh run`
  handles this; hand-rolled commands do not.
