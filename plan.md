# Claude Code Prompt: Natural Language Autoencoders for a Tabular Fraud DNN

Copy everything below the line into Claude Code as the task prompt.

---

## Goal

Recreate Anthropic's Natural Language Autoencoders (NLA) method (paper: https://transformer-circuits.pub/2026/nla/index.html) on a **DNN instead of an LLM** — specifically a small tabular payments-fraud model. The target model is a shallow MLP (128-dim hidden layer) trained on open, labeled payments-fraud transaction data pulled from Kaggle. The NLA (activation verbalizer + activation reconstructor) translates the MLP's 128-dim hidden activations into readable English explanations and back. Any openly available dataset is fair game; prefer real payments/transaction fraud over other fraud types.

We are NOT reimplementing NLAs from scratch. Reuse Anthropic's open-source release wherever possible:

- Full training repo (reference for method, prompts, configs, sidecar format): `https://github.com/kitft/natural_language_autoencoders`
- Lightweight inference package: `https://github.com/kitft/nla-inference`
- Released checkpoints (HF collection `kitft/nla-models`). We will initialize from **`kitft/nla-qwen2.5-7b-L20-av`** and **`kitft/nla-qwen2.5-7b-L20-ar`** (Qwen2.5-7B-Instruct, layer 20, d_model=3584). These already know how to verbalize activation vectors and reconstruct vectors from text — we only need to teach them a new activation domain via a small adapter + fine-tuning.

Key facts about the released NLA design you must respect (from the repo README — read it and `docs/inference.md` + `docs/design.md` before writing code):

- **AV**: injects the activation as a single token embedding into a fixed prompt (`input_embeds`, not `input_ids`), then autoregresses an explanation. The embed sequence is built trainer-side: look up the prompt tokens in the AV's embedding table, splice the (scaled) vector in at the injection slot. The release injects the raw vector with a fixed scalar `injection_scale`; the README explicitly notes a learned affine adapter `W·v + b` is a trainer-side-only change. **That adapter is exactly what we add**: `nn.Linear(128, 3584)`.
- **AR**: a truncated (K+1)-layer LM + `Linear(d, d)` head extracted at the final token. We replace/augment the head with a `Linear(3584, 128)` output map to land in the MLP's activation space.
- Vectors are **L2-normalized** before comparison; round-trip `MSE(recon, orig) = 2(1 − cos)`, so the loss measures direction agreement.
- Every released checkpoint ships an **`nla_meta.yaml` sidecar** with prompt template, injection token IDs, and scale factors. Load these; never hardcode.
- Reward in RL is `-mse_nrm`; RL is GRPO on the AV with a simultaneous supervised MSE step on the AR, plus a KL penalty toward the AV's SFT init.

**Compute assumption**: one machine with a single 24–80 GB GPU. Use LoRA (or QLoRA if VRAM-constrained) on the 7B AV/AR rather than full fine-tuning, and train the injection adapter + AR output head at full precision. Do NOT set up the Miles + SGLang distributed stack from the repo — it targets multi-node runs. Instead write a self-contained single-GPU training loop with HF `transformers` + `peft` (custom GRPO loop is fine; TRL's GRPOTrainer may be used only if you can make it work with `inputs_embeds` injection — if that fights you, write the loop by hand, it's ~200 lines). Use the repo as the source of truth for prompt templates, scale-factor handling, and the reward/loss definitions.

If any of this conflicts with what you find in the repo docs, the repo docs win — read them first.

---

## Repository layout to create

```
nla-fraud/
  README.md                  # what this is, how to run each phase
  pyproject.toml             # deps: torch, transformers, peft, pandas, pyarrow,
                             # scikit-learn, anthropic, matplotlib, pyyaml, faker
  configs/
    experiment.yaml          # all hyperparameters in one place
  data/                      # gitignored; generated artifacts land here
  src/
    get_data.py              # Phase 1: Kaggle download + feature prep (+ synthetic fallback)
    target_model.py          # Phase 2: FraudMLP + training + activation hooks
    collect_activations.py   # Phase 2: build activation parquet + PCA report
    serialize.py             # Phase 3: feature vector -> English rendering
    gen_summaries.py         # Phase 3: Anthropic API warm-start data generation
    sft_ar.py                # Phase 4
    sft_av.py                # Phase 4
    rl_grpo.py               # Phase 5
    evals.py                 # Phase 6: FVE, claim verification, grader evals
    steering.py              # Phase 6: edit-explanation -> patch -> score shift
    inspect_nla.py           # CLI: pass a transaction, print explanation + recon cos
  reports/                   # gitignored; metrics json, plots, sample explanations
```

`configs/experiment.yaml` holds: dataset size, MLP dims, hook layer, activation counts per phase, LoRA rank, LR, GRPO group size, KL beta, max explanation tokens, eval sizes, model IDs, seeds. Every script reads it. Seed everything.

---

## Phase 1 — Open payments-fraud data from Kaggle

Use the Kaggle API (expects `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`; ask me to set it up if missing). `get_data.py` should take a `--dataset` flag supporting the following, in order of preference:

1. **`ieee-fraud-detection` (default; real payments fraud)** — the IEEE-CIS Fraud Detection competition data from Vesta (~590k real e-commerce card transactions, `isFraud` labels). Closest open analog to a Stripe workload. NOTE: it's a *competition* dataset, so `kaggle competitions download -c ieee-fraud-detection` requires me to have accepted the rules once in the browser — detect the 403 and tell me if so. **Critical design constraint**: many columns are anonymized (V1–V339, D-, C- series). Restrict the MLP's input features to the *verbalizable* subset — `TransactionAmt`, `TransactionDT`-derived time features, `ProductCD`, `card4` (network), `card6` (debit/credit), `P_emaildomain`/`R_emaildomain`, `addr1/addr2`, `dist1`, `DeviceType`, `DeviceInfo`, plus engineered per-card velocity and amount z-score features built from `card1` as the card key. The warm-start summaries can only describe features that have names; a model trained on V-columns would have internals the verbalizer can't ground. Document the chosen feature list in the report. Expect a lower AUC than full-feature Kaggle solutions; that's fine.
2. **`kartik2112/fraud-detection` (easy mode; fully named)** — Sparkov synthetic card transactions, ~1.3M rows, every feature human-readable (merchant, category, amt, geo, job, dob, `is_fraud`). Plain datasets API, no rules acceptance. Use this to debug the pipeline end-to-end even if IEEE-CIS is the headline run.
3. **`ealtman2019/credit-card-transactions` (scale option)** — IBM/TabFormer synthetic card payments, ~24M rows, fully named (amount, merchant, MCC, chip/online, error codes, `is_fraud`). Good if we later want a bigger activation corpus.

Regardless of source, `get_data.py` normalizes to a common internal schema, engineers the shared behavioral features (amount z-score vs card history, 1h/24h velocity, hour-of-day, geo distance where available, account/card tenure), does a time-based train/val/test split (no leakage across the split from per-card aggregates — compute them causally), and writes `data/transactions.parquet`. Keep a small synthetic-generator fallback (faker + numpy, Sparkov-style schema with injected card-testing bursts and geo-anomaly fraud) behind `--dataset synthetic` for offline runs.

Deliverable: `data/transactions.parquet` + printed class balance, feature list, and per-feature verbalizability table (name → how it will be rendered in English).

## Phase 2 — Target model + activation corpus

`target_model.py`: a plain MLP with embeddings for categoricals:

```
embeddings(cats) ++ numerics -> Linear(d_in,256) GELU Dropout
                             -> Linear(256,128)  GELU Dropout   # layer "l2"
                             -> Linear(128,128)  GELU           # layer "l3"
                             -> Linear(128,1)                   # fraud logit
```

Train with BCE + class weighting; report AUC-ROC and AUC-PR on the time-based held-out split (rough bar: >0.85 AUC-ROC on IEEE-CIS with the restricted feature set, >0.9 on Sparkov; if far below, revisit feature engineering before proceeding). Save checkpoint + preprocessing artifacts.

`collect_activations.py`: forward-hook **both** `l2` and `l3`; for a 1M-transaction sample, write parquet files matching the NLA repo's expected format (an `activation_vector` column of float lists), one file per layer, plus columns carrying the row's input features, model score, and label (needed for summaries and evals). L2-normalize a copy for training; keep raw too.

**Mandatory sanity check before proceeding**: PCA each layer's activations; print/plot the explained-variance spectrum and the number of components for 90/95/99% variance. Also fit a quick logistic probe from activations to `is_fraud`. Record all of this in `reports/activation_report.json`. If ~5 components explain 99%, note it — it bounds how rich explanations can be, and it argues for preferring `l2` over `l3` (near-logit layers collapse toward the score).

Default the experiment to **`l2`**, keep `l3` as a config switch.

## Phase 3 — Warm-start text data (this replaces "text context" for a model that has none)

The NLA warm-start pairs activations with LLM-written summaries of the *input context*. Our context is the transaction's feature vector, which is fully named — so we serialize it to English and summarize that.

`serialize.py`: deterministic feature-vector → text rendering driven by the per-dataset verbalizability table from Phase 1, e.g. for IEEE-CIS: "Card-not-present purchase, product code W, $1.02 at 03:14, Visa debit, purchaser email domain protonmail.com, mobile device, card's 47th transaction in the past 24h (velocity z=+8.2), amount z-score −1.9 vs card history, model fraud score 0.93." Only render features the MLP actually consumes.

`gen_summaries.py`: call the Anthropic API (`ANTHROPIC_API_KEY` from env; use `claude-sonnet-4-6`; batch with concurrency and disk-cache responses so reruns are free) to produce a 3–5 bullet summary per row. Adapt the warm-start prompt from the NLA repo's datagen pipeline (`nla/datagen/`) — keep its structure but reframe: "A fraud-detection neural network has processed the following transaction. Identify the 3–5 most important risk or legitimacy signals the network is most likely representing internally at this point. Order by importance." Include the model's fraud score and, ideally, per-feature gradient×input attributions in the prompt so summaries reflect what the model weighted, not just what the input contained. Cap summaries at ~120 tokens — a 128-dim vector cannot support 500-token explanations, and longer targets teach the AV to pad with confabulation.

Generate **~100k** (activation, summary) pairs for SFT (start with 10k to validate the pipeline end-to-end, then scale). Store as parquet: `activation_vector`, `summary`, `features_json`, `fraud_score`, `label`.

## Phase 4 — SFT (adapter + LoRA on released checkpoints)

Load `kitft/nla-qwen2.5-7b-L20-av` and `-ar` with their `nla_meta.yaml` sidecars (prompt template, injection token IDs, scale factors — use them exactly).

`sft_av.py`: add `adapter = nn.Linear(128, 3584)` (init: small-variance random, output L2-normalized then multiplied by the sidecar's injection scale so it lands in the distribution the AV was trained on). Build the prompt embeds trainer-side exactly as the repo does, splicing `adapter(h)` at the injection slot. Train next-token loss on the summaries. Trainable params: the adapter (always) + LoRA on the AV (r=16, attn+MLP). Consider a short adapter-only warm-up epoch before enabling LoRA.

`sft_ar.py`: the AR is the truncated LM + head. Add/replace the output map with `Linear(3584, 128)`; train MSE against L2-normalized target activations, LoRA on the trunk + full-precision head. 

Exit criteria for Phase 4: run AV→AR round-trip on 1k held-out activations; report FVE (using the repo's normalized-MSE convention, `mse = 2(1−cos)`). Expect roughly 0.3–0.5 FVE post-SFT. Print 20 random sample explanations to `reports/` and eyeball-check they're fluent and transaction-relevant before RL.

## Phase 5 — RL (GRPO on AV, simultaneous supervised AR)

`rl_grpo.py`, mirroring the paper/repo recipe on a single GPU:

- Each step: sample a batch of activations (batch 32–64), generate a **group of G=8** explanations per activation from the AV at temperature 1.0, max ~120 new tokens.
- AR step: one supervised gradient step, MSE(AR(z), h) on all sampled (z, h) pairs.
- AV step: GRPO with reward `r = -mse_nrm` (optionally `-log mse`, per the paper appendix), group-normalized advantages, plus KL penalty `β` toward the frozen SFT AV (start β≈0.05, tune if explanations degrade or reward stalls).
- Log per step: FVE (train + a fixed held-out set every N steps), mean reward, KL, mean explanation length, and 3 sample explanations. Checkpoint every 500 steps.
- Run until held-out FVE plateaus; on a 128-dim target expect this to move fast — a few thousand steps is a reasonable first budget. Save curves to `reports/`.

Generation during RL: plain HF `model.generate` with `inputs_embeds` is acceptable at this scale (no SGLang server needed); make it batched. If throughput is painful, generate with the LoRA-merged model in bf16.

## Phase 6 — Evaluations (adapted from the paper)

`evals.py`, all against the **final RL checkpoint and the SFT checkpoint** (so we can show RL improved informativeness, the paper's central claim):

1. **FVE** on held-out activations (headline reconstruction metric).
2. **Automatic claim verification** — our structural advantage over the LLM setting: every input feature is known. Use the Anthropic API to extract atomic claims from each explanation, then verify each claim programmatically against the row's true features (amount bucket, velocity, card network/type, email domain, device, time-of-day, tenure). Report precision of verifiable claims (confabulation rate) at SFT vs post-RL.
3. **Grader evals** (paper §"Evaluating NLAs during training"): given ONLY the explanation, a Claude grader predicts (a) the MLP's fraud score bucket (5 bins), (b) the true label, (c) 3 masked feature values. Report accuracy vs a majority-class baseline, SFT vs RL.
4. **Layer comparison**: repeat 1–3 for `l3` with a cheaper run; confirm/deny the "near-logit layers give accurate-but-boring explanations" hypothesis.

`steering.py` — causal validation, the fraud analog of the paper's rabbit→mouse edit:
- Pick ~200 held-out transactions with fraud score in [0.6, 0.95] whose explanations mention velocity, amount, or device/geo anomalies.
- Programmatically edit the explanation to its benign counterpart ("47 transactions in the past 24h" → "2 transactions in the past 24h"; "amount 6x the card's typical spend" → "amount typical for this card").
- Compute `Δ = AR(z_edit) − AR(z_orig)`, patch `h ← h + α·‖h‖·Δ/‖Δ‖` at the hooked layer (sweep α ∈ {0.25, 0.5, 1.0, 2.0}), rerun the MLP head, and measure the fraud-score shift vs (a) no edit and (b) a random-direction Δ of equal norm.
- Success criterion: semantically-edited directions move the score toward benign significantly more than random directions. Plot score-shift distributions per α.

`inspect_nla.py`: a small CLI/demo — feed one transaction (JSON or a row index), print the serialized input, the MLP score, the NLA explanation, reconstruction cosine, and the verified/unverified breakdown of its claims. This is the artifact for showing people.

## Phase 7 — Baselines (all run through the SAME grader-eval harness as Phase 6)

Add `src/baselines.py`. Every baseline produces "explanations" (or predictions) evaluated with the identical graders, splits, and metrics as the NLA, so the comparison table is apples-to-apples.

**Controls (validity):**
1. **Activation-swap control (floor).** Run the AV on activations from a random *other* row while grading against the original row. Grader-eval accuracy must collapse toward the majority-class baseline; if it doesn't, explanations are coming from the prompt prior, not the vector. Run this at SFT and post-RL checkpoints. This is a hard gate — flag loudly if it fails.
2. **Majority-class / mean-prediction baseline** for every grader task, reported in the same table.

**Information baselines:**
3. **Input-serialization baseline.** Claude (same model as the graders) writes an explanation directly from the serialized transaction text + no activation. Evaluate on all grader tasks. Expectation: it wins or ties on predicting the *true label*, but the NLA should win on predicting the **model's score bucket** and especially on **model-error cases** (score/label disagreements) — carve those out as a separate eval slice, since "the activation knows what the model thinks" is the entire value proposition.
4. **Linear probes (ceiling).** Logistic/linear probes from h to: fraud score (regression), label, and each verbalizable feature. Report probe accuracy alongside grader-from-explanation accuracy; the ratio is the "fraction of linearly-decodable information surviving the text bottleneck."

**Practitioner baseline:**
5. **SHAP/attribution-templated text.** Compute per-feature attributions (SHAP or gradient×input) for each eval row, render the top-5 as templated English, and run the full grader-eval suite plus include 10 side-by-side examples (SHAP-text vs NLA explanation) in the report for qualitative comparison.

**Method-ablation baselines:**
6. **PastLens-style SFT (no RL).** Train an AV variant only to reconstruct the serialized input text from the activation (next-token on serializations rather than summaries), same budget as Phase 4 SFT. The paper found this simpler recipe competitive as an initialization — if it matches the GRPO run on grader evals here, report that prominently as a real finding, not a failure.
7. **SFT-only vs post-RL** (already required in Phase 6) — the replication of the paper's central claim that informativeness improves with reconstruction RL.
8. **Steering comparator.** In `steering.py`, add a probe-derived direction (e.g., the velocity probe's weight vector, norm-matched) alongside the existing random-direction control, so the NLA edit-direction is judged against both a floor and a strong ceiling.

Deliverable: one consolidated table in `reports/EXPERIMENT.md` — rows = {NLA-RL, NLA-SFT, PastLens-SFT, input-serialization, SHAP-text, linear probe, activation-swap, majority-class}, columns = {FVE (where applicable), score-bucket acc, label acc, masked-feature acc, model-error-slice acc, claim precision}.

## Final report

Write `reports/EXPERIMENT.md` summarizing: setup, FVE curves, confabulation rates, the consolidated Phase 7 baseline table, the l2-vs-l3 layer comparison, steering results with plots (NLA edit vs probe direction vs random), 10 curated example explanations (mix of fraud/legit, with the SHAP-text counterpart shown for each), and a short verdict section: on which metrics did the NLA beat the input-serialization and SHAP baselines, and specifically how it performed on the model-error slice.

## Working style

- Work phase by phase; each phase has a runnable script and a checked-in sanity artifact in `reports/` before you move on. Stop and show me results at the end of Phases 2, 4, and 5 before continuing.
- Before writing any NLA-touching code, clone `kitft/natural_language_autoencoders`, and read `README.md`, `docs/inference.md`, `docs/design.md`, `nla_inference.py`, the datagen configs, and one `nla_meta.yaml` sidecar from the HF checkpoint. Lift their conventions (prompt template, injection mechanics, normalization, reward) rather than inventing parallel ones, and cite the file you lifted each convention from in code comments.
- Anything API-costly (summary generation, graders) must be disk-cached, resumable, and gated behind a `--limit` flag with a small default.
- Prefer boring, debuggable code over cleverness. No premature abstraction across phases.