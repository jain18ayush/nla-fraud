# What Needs Improving, and Why

Findings and open problems as of the Phase-4 checkpoint, ordered by expected
value. Every claim here cites a file in `reports/` so it can be re-checked
rather than taken on trust.

Operational instructions live in [RUNBOOK.md](RUNBOOK.md).

---

## Where things actually stand

Two validity gates have been run. Both matter for reading everything below.

**Activation-swap control — PASS** (`reports/phase7_swap_control.json`, n=500)

| condition | FVE | cos (mean) |
|---|---|---|
| real round-trip | **+0.431** | 0.784 |
| swap (AV sees `h_j`, scored vs `h_i`) | **−0.867** | 0.292 |
| mean-activation floor | −0.340 | 0.492 |
| text Jaccard, `expl(h_i)` vs `expl(h_j)` | **0.184** | — |

The AV genuinely reads the activation. A 1.30 FVE gap and a token-overlap of
0.18 (a prior-driven AV would score 0.7–0.9) both rule out the "fluent template
independent of the vector" failure mode.

This also retires an earlier worry: generated explanations state feature values
that contradict the gold summary ($75.00 vs $100, tenure 17 vs 11). That is the
128-d bottleneck being unable to carry exact magnitudes — **not** confabulation
from ignoring the input.

> The swap script originally reported `AMBIGUOUS` here. Its rule required swap
> to land *within* 0.05 of the floor, via `abs(gap_swap_floor)`. But swap
> landing **below** the floor is a pass, not an anomaly: the floor is the
> centroid — blandly wrong about everything, never far from anything — while
> `AR(AV(h_j))` is *confidently* wrong, and a vector aimed at a random other
> point in activation space is further from `h_i` than the centroid is. The
> criterion is fixed in `roundtrip_eval.py`; re-running now yields `PASS`.

**Injection-adapter geometry — partial concern**
(`reports/phase7_adapter_diagnostic.json`, n=500)

| | input (raw 128-d) | after adapter (3584-d) |
|---|---|---|
| mean pairwise cosine | 0.238 | 0.602 |
| effective rank (entropy) | **8.13** | **3.00** |
| components for 90% var | 10 | 3 |

Geometry is preserved — Spearman 0.943 between input and output pairwise
cosines, so nothing is being scrambled. But it is compressed hard.

---

## 1. The AR is the bottleneck — and it is starved of data

**Highest-confidence, highest-value fix.**

| measurement | FVE | source |
|---|---|---|
| AR on **gold** summaries (teacher-forced) | 0.404 | `phase4_ar_sft.json` |
| full round-trip AV→AR, `max_len` 120 | 0.406 | `phase4_roundtrip_sft.json` |
| full round-trip AV→AR, `max_len` 180 | 0.429 | `phase4_roundtrip_sft_len180.json` |

Round-trip **meets or beats** teacher-forced. The AV's generated explanations
are at least as reconstructable as the ground-truth summaries. So the text
bottleneck is not where information is lost — **the AR cannot map summary-text
to 128-d better than ~0.40, and no AV improvement can exceed that ceiling.**

The AR is memorizing 9 000 pairs. Per-epoch train `mse_nrm` vs held-out:

| epoch | train mse | holdout mse |
|---|---|---|
| 0 | 0.740 | — |
| 1 | 0.398 | — |
| 2 | 0.233 | **0.447** (flat) |

By epoch 2 train is nearly 2× better than holdout. Textbook data starvation.

**Action:** generate the full 100 k summary corpus and retrain the AR on it.
The AR is the cheap model (1 067 s vs the AV's 2 208 s), the A100 is idle, and
the expensive part is API generation, not GPU. Consider training the AR on more
pairs than the AV.

**Do not** reach for LoRA rank or learning-rate changes first — the failure is
generalization, not capacity.

---

## 2. The summary corpus is low-entropy

AV final training loss 0.535, **perplexity 1.71** (`phase4_av_sft.json`). That
is not good news: it means the AV has essentially memorized the output template.
Most of its bits go to format, not content.

Every gold summary is the same shape — five bullets of
`` `feature: value` → direction (attribution) ``. Meanwhile the activation needs
~8 effective dimensions to describe (§4). A stereotyped 5-bullet card is a lossy
code for that, and the AR can only recover what the text encodes. **This sets
the ceiling that §1 runs into.**

**Actions:**
- Widen to top-8 features rather than top-5.
- Keep the numeric attributions (gold already carries e.g. `-1.369` — good).
- Deliberately vary phrasing to raise cross-row entropy.
- Consider conditioning generation on the top PC coordinates of the activation,
  not only on input-feature attributions. Today the summaries describe *the
  input*; the NLA's job is to describe *the activation*.

---

## 3. The adapter discards over half the signal before the AV sees it

Effective rank **8.13 → 3.00**. The injected token varies meaningfully across
inputs (so this is not the catastrophic failure mode), but roughly five of the
eight available dimensions do not survive the projection.

Also worth knowing: the injected vector has L2 = 150 while a real Qwen token
embedding averages 0.79 — a **~190× norm ratio**. That is a deliberate design
choice from the sidecar (`injection_scale` is mandatory; the AV was trained in
that norm band), so it is not a bug. But combined with rank-3 output it raises
a real question: a huge, direction-underdiverse token may be acting more like a
near-fixed *mode selector* than a rich content carrier.

The adapter is a single `Linear(128 → 3584)` trained with **one** warm-up epoch
on 9 k samples. That is very little to learn a good embedding-space landing zone.

**Actions (in order):**
1. Settle §4 first — if the activation is intrinsically rank ~3, there is
   nothing to recover and this item closes.
2. Otherwise: more adapter warm-up epochs, and more data (same fix as §1).
3. Only then consider a wider adapter (MLP instead of Linear). Capacity is the
   least likely culprit.

---

## 4. Open question: is the low rank intrinsic or induced?

The **input** activations have effective rank ~8.13 (10 of 128 components for
90% variance; `activation_report.json` reports 14 for the full corpus). This
bounds explanation richness before the NLA is involved at all, and the original
plan anticipated it: *"If ~5 components explain 99%, note it — it bounds how
rich explanations can be."*

It also reframes the headline number: **FVE 0.43 against a rank-8 target is
weak, not respectable.** A signal that low-rank should reconstruct far better.

Nobody has yet determined whether the 8 → 3 compression in §3 is adapter waste
or a faithful reflection of the fraud MLP genuinely needing only ~3 directions.

**Action:** compare the activation's principal directions against the
label/fraud-score structure — a linear probe from each PC to `is_fraud` and to
`fraud_score`. If ~3 PCs carry all the label information, the adapter is doing
the right thing and §3 closes. If PCs 4–8 carry real signal that the adapter
drops, §3 is a genuine loss. This is cheap and unblocks a real decision.

---

## 5. Eval slices are too small to mean anything

`fve_fraud_slice` is reported as 0.158 (len120) and 0.178 (len180) versus ~0.41
overall — alarming, except **n_fraud = 31** in a 1 000-row holdout. That
estimate is noise.

Worse, the slice Phase 7 actually cares about — model errors, where score and
label disagree — is far rarer. In the training split the false-negative cell
(fraud label, score < 0.2) has **663 rows total**. A 2 000-row holdout at the
natural rate would contain roughly three.

"The activation knows what the model thinks" is the entire value proposition,
and it is currently measured on a handful of rows.

**Action:** carve eval holdouts with a stratified floor per (label × score-bin)
cell instead of sampling at the natural rate, and report per-cell n alongside
every sliced metric. `gen_summaries.stratified_sample` already does exactly this
for the generation corpus and can be reused.

---

## 6. Phases 6 and 7 are largely unwritten

Missing: `evals.py`, `steering.py`, `baselines.py`, `inspect_nla.py`.

Of the Phase 7 baselines the plan specifies, only the activation-swap control
exists (as `roundtrip_eval.py --swap`). Still needed: majority-class,
input-serialization, linear-probe ceiling, SHAP-templated text, and the
PastLens-style SFT ablation.

The **input-serialization baseline** deserves priority: it is the honest
comparator. Expect it to win on predicting the true *label*; the NLA must win on
predicting the **model's score bucket** and on the **model-error slice** (see §5)
or the method has not demonstrated its claim.

---

## 7. Does RL rescue any of this?

**Not §1, and it may actively mislead.**

GRPO's reward is `-mse_nrm` *as judged by the AR*. If the AR is the weak link,
RL pushes the AV toward whatever the current AR happens to decode well — reward
hacking against AR quirks rather than genuinely more informative text. The
simultaneous supervised AR step helps, but it starts from an AR that overfits
9 k pairs.

The swap control passing (above) means the reward signal is *real* — RL is
justified rather than speculative. But **fix the corpus first**, or the Phase 6
claim "RL improved informativeness" rests on a weak baseline and the comparison
proves little.

---

## 8. Data logistics

`data/activations_l{2,3}.parquet` (~900 MB each) exist **only on the laptop**;
the box has just `summaries.parquet`. Regenerating summaries on the box requires
getting an activation parquet there first, against 29 GB free. Plan the transfer
rather than discovering it mid-run.

---

## Suggested order

1. **§4** — the PC-vs-label probe. Cheap, and it decides whether §3 is real.
2. **§1 + §2** — generate 100 k summaries with the stratified sampler and
   denser content, retrain the AR. This is the main event.
3. **§5** — stratified eval holdouts, so the retrain can actually be measured.
4. **§6** — input-serialization baseline, for an honest comparator.
5. **§7** — RL, once the above give it a baseline worth beating.
