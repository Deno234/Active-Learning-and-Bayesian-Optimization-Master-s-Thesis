# Thesis Handoff For The Next Model

Last updated: 25 June 2026
Git commit: `7cbfa95d8d56ce4b558238341ed5e19354ffcec2`
Repository branch: `codex/active-learning-thesis`
Repository root: `<local_workspace>`
Author: Denis Ibisi (Git identity: `<github_user>`)
Thesis topic: Machine-learning-guided discovery and active learning of self-assembling peptides

> This document is the current authoritative technical handoff for thesis
> writing. When older documentation conflicts with it, this document and the
> referenced generated implementation audits take precedence.

The commit above identifies the inspected Git baseline. The working tree also
contains many uncommitted generated artefacts and documentation changes, so a
future assistant must inspect `git status` before assuming that the commit
alone reconstructs the current scientific evidence.

## 1. Canonical Project Constants

| Item | Canonical value |
|---|---:|
| Total peptides | 368 |
| Positive experimental labels | 249 |
| Negative experimental labels | 119 |
| Model-fitting / replay-training rows | 235 |
| Validation / calibration rows | 59 |
| Frozen holdout rows | 74 |
| Canonical alphabet | `ACDEFGHIKLMNPQRSTVWY` |
| Valid sequence length range | 3-24 |

The canonical dataset is
`SA_ML_predictive/data/data_SA.csv`. Validation and holdout rows are excluded
from candidate acquisition. Where a protocol explicitly documents it, frozen
holdout sequence identities may be used only for exact duplicate exclusion.
Holdout labels, predictions, metrics, and performance information must not
influence fitting, calibration, acquisition, ranking, or proposal selection.

## 2. Predictive Model And Calibration Contract

### 2.1 AP_SP input representation

The adaptive surrogate is the hybrid `AP_SP` model. Its ordered inputs are:

1. amino-acid aggregation-propensity branch: `(25, 1)`;
2. dipeptide aggregation-propensity branch: `(25, 1)`;
3. tripeptide aggregation-propensity branch: `(25, 1)`;
4. physicochemical sequence-property branch: `(25, 94)`.

All shapes exclude the batch dimension. Runtime padding is 25 positions, and
the masking value is `2`. The tensors are passed as `(length, channels)`.
Detailed runtime evidence is in `THESIS_FEATURE_RUNTIME_SHAPES.md`.

### 2.2 Frozen architecture

Each AP branch uses:

`Masking -> Bidirectional LSTM(5 per direction, return_sequences=True) ->
LSTM(5) -> Dense(96, SELU) -> Dropout(0.5)`.

The SP branch uses:

`Masking -> Conv1D(5 filters, kernel 8, same, linear) -> Conv1D(5 filters,
kernel 8, same, linear) -> Bidirectional LSTM(48 per direction) ->
Dropout(0.5)`.

The three 96-dimensional AP outputs and the 96-dimensional SP output are
concatenated into the mandatory 384-dimensional penultimate representation.
A final one-unit sigmoid layer produces the self-assembly probability.
Architecture evidence is in `THESIS_PREDICTIVE_MODEL_ARCHITECTURE.md`.

### 2.3 Training

- optimizer: Adam;
- initial learning rate: `0.01`;
- batch size: `600`;
- maximum epochs: `70`;
- objective: class-weighted binary cross-entropy;
- checkpoint: in-memory weights at minimum validation loss;
- learning-rate schedule: initial rate for epochs 1-10, followed by
  multiplication by `exp(-0.1)` at each subsequent epoch;
- no patience-based early stopping.

### 2.4 Calibration

Ensemble workflows calibrate each member separately; Phase 5's primary design
calibrates its single member. The shared calibration contract is:

- validate dimensions, lengths, and finiteness; malformed input fails;
- clip raw probabilities to `[1e-6, 1-1e-6]`;
- transform clipped probabilities to logits;
- standardise with the member-specific validation-logit mean and population
  standard deviation;
- fit Platt coefficient and intercept with learning rate `0.05`, at most `500`
  iterations, and `L2=1e-3` towards `a=1`, `b=0`;
- use identity fallback only for valid degeneracy or fitting failure;
- identity fallback returns the clipped raw probability unchanged.

The validation F1-maximising classification threshold uses higher-threshold
tie-breaking. It is reporting-only and is never an acquisition incumbent or
reference value.

## 3. Phase Status

| Phase | Scientific objective | Implementation status | Execution status | Primary output directory | Main source of truth | Outstanding work |
|---|---|---|---|---|---|---|
| 1 | Reproduce five published classifiers | complete | `executed_and_aggregated` | `thesis_results/01_reproduction/` | `tables/reproduced_predictive_performance.csv` | Thesis presentation only |
| 2 | Retrospective acquisition-strategy replay | complete | `executed_and_aggregated` | `thesis_results/02_replay/evidence/` | `benchmark_strategy_summary.csv` | Thesis interpretation only |
| 3 | Eight-round branch-isolated CG-MD active learning | complete | `executed_and_aggregated` | `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/` | `comparison/all_rounds_branch_comparison.md` | Write final result narrative |
| 4 | One-round fixed-surrogate BO-guided proposals | complete | `executed_and_aggregated`; 29 valid proposals simulated and reconciled | `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/` | `implementation_audit/bo_implementation_audit.md`; `thesis_reporting/phase4_assessment_inventory.csv` | Report 29/30 valid-assessed coverage and disclose length-25 MES defect |
| 4-D | Fresh diversity-aware fixed-surrogate replicate | complete | `executed_and_aggregated` | Primary Phase 4 path above plus `phase4d/run_001/` | `phase4d/run_001/phase4d_report.md` | Manual simulation-slate decision |
| 5 | SPAL-inspired self-paced retrospective replay | complete | `executed_and_aggregated` | `thesis_results/05_self_paced_active_learning/` | `PHASE5_RESULTS_SUMMARY.md` | Thesis interpretation only |

The nested Phase 3 and Phase 4 paths above are the completed scientific
archives. The shorter top-level working directories contain stale, partial, or
preview material and must not silently replace the completed sources.

## 4. Phase 1: Predictive Reproduction

The five reproduced model families are `AP`, `SP`, `AP_SP`, `TSNE_SP`, and
`TSNE_AP_SP`. A nested five-outer-fold by five-inner-fold protocol selected
architecture hyperparameters only from inner-fold validation evidence.
Untouched outer folds supplied generalisation estimates.

The AP_SP reproduced PR-threshold aggregate was:

- accuracy: `0.81803`;
- F1: `0.86857`;
- PR-AUC: `0.93557`.

The all-368-row top-level AP_SP model is a deployment-style fit, not a holdout
baseline. Thesis comparisons must use fold-matched outer-fold predictions,
whose models were fitted on the corresponding 294-row development partitions
and evaluated on their 74-row outer test folds.

Thesis evidence:

- `thesis_results/01_reproduction/tables/reproduced_predictive_performance.csv`
- `thesis_results/01_reproduction/tables/nested_cv_outer_predictions.csv`
- `thesis_results/01_reproduction/tables/nested_cv_outer_predictions_AP_SP.csv`
- `thesis_results/01_reproduction/tables/nested_cv_inner_results.csv`
- `thesis_results/01_reproduction/tables/hyperparameter_summary.csv`
- `thesis_results/01_reproduction/tables/threshold_summary.csv`
- `thesis_results/01_reproduction/tables/frozen_model_config_AP_SP.json`
- `thesis_results/01_reproduction/tables/preprocessing_shapes.csv`
- `thesis_results/01_reproduction/folds/`

## 5. Phase 2: Retrospective Replay

The exact ten main replay strategies are:

1. random;
2. ensemble mean;
3. similarity-penalised mean;
4. predictive entropy;
5. ensemble mutual information;
6. UCB;
7. family QBC;
8. cluster-diverse representative selection;
9. OED logdet;
10. hybrid MI-diverse.

PI, EI, and MES were not main Phase 2 replay policies. Their availability in
generic discovery support must not be presented as Phase 2 benchmark evidence.

Five outer folds served as five fold-level repetitions with different
stratified test partitions and deterministic initial seeds. Their training
partitions overlapped, so the repetitions are not statistically independent.
The `n0=10` and `n0=40` conditions are reported separately. Batch size is 5,
the maximum is 20 acquisition rounds, and labels-to-target are reported for F1
0.80, 0.84, and 0.86. Targets are detected only at observed replay points,
without interpolation, and summaries include reach counts or fractions.

For labelled counts `L_0,...,L_T`, the implemented normalised discrete
trapezoidal AULC is:

```text
AULC_F1 =
  [sum from t=0 to T-1 of
   ((F1_t + F1_(t+1))/2) * (L_(t+1) - L_t)]
  / (L_T - L_0)
```

The two initial-label conditions cover different intervals and therefore have
separate AULCs.

Completed evidence:

- `thesis_results/02_replay/evidence/learning_curves.csv`
- `thesis_results/02_replay/evidence/labels_to_target_summary.csv`
- `thesis_results/02_replay/evidence/paired_vs_random.csv`
- `thesis_results/02_replay/evidence/benchmark_strategy_summary.csv`
- `thesis_results/02_replay/evidence/strategy_compatibility_matrix.csv`
- `thesis_results/02_replay/evidence/ablation_summary.csv`
- `thesis_results/02_replay/evidence/ablation_calibration_summary.csv`
- `thesis_results/02_replay/evidence/figures/`
- `thesis_results/02_replay/evidence/figures/thesis_main/`

## 6. Phase 3: Branch-Isolated Prospective Active Learning

The prospective branches were predictive entropy, family QBC, and
cluster-diverse representative selection; ensemble MI was retained as a backup.
All branches began from the same 235 experimental training labels but used
independent ledgers. Each branch proposed five peptides per round for eight
rounds. Acquired CG-MD labels were ingested only into their originating branch;
there was no automatic cross-branch sharing.

The completed archive contains 120 reviewed and ingested CG-MD rows:
`3 branches x 8 rounds x 5 peptides`. Terminal models therefore used 275
labels per branch and were evaluated on the fixed validation and frozen
holdout sets.

| Branch | Frozen-holdout F1 | Frozen-holdout PR-AUC |
|---|---:|---:|
| cluster-diverse representative | 0.875000 | 0.924270 |
| family QBC | 0.871287 | 0.925479 |
| predictive entropy | 0.876190 | 0.911278 |

These results permit descriptive branch comparison. They do not establish
universal superiority or independent external validation.

The operational label is exactly:

```text
label = 1 only when
AP_sasa(200 ns) >= 1.75
AND paper_path_APcontact_last10ns >= 0.5
```

The exact full-profile BURA simulation parameters are frozen in
`THESIS_CGMD_PARAMETER_CONTRACT.md`. This includes the three minimisation
thresholds, the 9 ps and 12.5 ps restrained equilibration stages, the 200 ns
production settings, thermostat/barostat constants, nonbonded cutoffs,
trajectory and energy intervals, active topology constraints, charged termini,
and the all-extended (`E`) secondary-structure assumption. Use that file rather
than reconstructing parameters from narrative text.

The 0.6 nm contacted-molecule fraction `AP_contact` and other contact variants
are diagnostic only. All 120 reviewed rows contain
`paper_path_APcontact_last10ns`, all evidence summaries name it, and all
retained labels match the rule.

Implementation entry points:

- `active_learning_thesis.phase3_real_al.propose_phase3_real_al`
- `active_learning_thesis.phase3_real_al.compare_phase3_real_al`
- `active_learning_thesis.phase3_real_al.make_phase3_ingest_csv`
- `active_learning_thesis.phase3_real_al.ingest_phase3_labels`
- `active_learning_thesis.phase3_real_al.finalize_phase3_real_al`
- `active_learning_thesis.phase3_real_al._collect_phase3_review_rows`
- `active_learning_thesis.phase3_real_al._validate_phase3_ingest_rows`
- `active_learning_thesis.md_workflow._paper_path_ap_contact_for_gro`
- `active_learning_thesis.md_workflow._write_paper_path_last10_ap_contact_file`
- underlying proposal path: `active_learning_thesis.workflow.propose_round`

Completed evidence:

- `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/comparison/all_rounds_branch_summary.csv`
- `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/comparison/all_rounds_branch_metrics.csv`
- `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/comparison/final_branch_holdout_metrics.csv`
- `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/branches/<branch>/current_labeled_ledger.csv`
- `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/branches/<branch>/rounds/round_001...round_008/`

## 7. Primary Phase 4: Fixed-Surrogate BO Proposals

Primary Phase 4 is branch-neutral and single-round. It used the fixed
235/59/74 split, five independently seeded AP_SP members (`20260317` through
`20260321`), member-wise calibration before aggregation, and population
standard deviation (`ddof=0`). All guided policies shared one immutable set of
checkpoints, calibrators, calibrated training predictions, and incumbent.

For candidate `x`, with calibrated member probabilities `p_m(x)`:

```text
mu(x)    = mean_m p_m(x)
sigma(x) = population_std_m p_m(x)
f*       = max over the 235 training rows of mu(x)
```

The reporting threshold does not enter any utility. The six policies are
random, calibrated greedy, calibrated UCB, probability-space approximate PI,
probability-space approximate EI, and calibrated ensemble-based approximate
MES.

Constants: `kappa=1.0`, `xi=0.0`, utility epsilon `1e-8`, MES CDF clipping
`[1e-12, 1]`.

```text
U_greedy(x) = mu(x)
U_UCB(x)    = mu(x) + kappa * sigma(x)

I(x) = mu(x) - f* - xi
z(x) = I(x) / max(sigma(x), 1e-8)
```

When `sigma(x) > 1e-8`:

```text
U_PI(x) = Phi(z)
U_EI(x) = I * Phi(z) + sigma * phi(z)
```

When `sigma(x) <= 1e-8`:

```text
U_PI(x) = 1 if I > 0 else 0
U_EI(x) = max(I, 0)
```

For a scoring pool `C`, MES uses five coherent member-function maxima:

```text
y*_m(C)       = max over x in C of p_m(x)
gamma_m(x; C) = (y*_m(C) - mu(x)) / max(sigma(x), 1e-8)
C_m(x; C)     = clip(Phi(gamma_m), 1e-12, 1)
U_MES(x; C)   = mean_m [
                    gamma_m * phi(gamma_m) / (2 * C_m)
                    - ln(C_m)
                ]
```

MES is zero when `sigma(x) <= 1e-8`. Final ranking uses one common retained-pool
set of member maxima. This is a finite-pool neural-ensemble approximation, not
exact Gaussian-process MES.

The inherited Phase 3 population-composition and preferred-length penalties
were active during GA evolution. Final guided selection ranked the retained
pool by unpenalised acquisition utility. Random used the inherited seeded
shuffle.

All six jobs completed. Each policy retained 50 candidates and selected five.
The comparison contains 30 selected slots, 30 unique exact sequences, and no
cross-policy exact duplicates.

Primary Phase 4 CG-MD simulation evidence has now been reconciled from the
local BURA/SUPEK campaign folders. Twenty-nine of the 30 selected proposals
have complete operational metrics; the excluded record is the known length-25
MES proposal `VLNINNMGAKWRRTCNQRLTPTALP`. The operational label is the same
threshold conjunction used for Phase 3 ingestion:

```text
AP_sasa(200 ns) >= 1.75
AND paper_path_APcontact_last10ns >= 0.5
```

The reconciled primary Phase 4 assessment contains 15 operational positives
among the 29 valid assessed proposals. This is computational CG-MD evidence
from one retained trajectory per sequence, not experimental validation.

Canonical paths:

- `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/implementation_audit/`
- `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/comparison/round_001/`
- `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/branches/<policy>/rounds/round_001/`

Known historical artefact defect: primary MES selected
`VLNINNMGAKWRRTCNQRLTPTALP`, whose length is 25 rather than 3-24. Do not
rewrite the completed historical output. Disclose and handle it explicitly in
any simulation slate.

## 8. Phase 4-D: Diversity-Aware Generative Replicate

Phase 4-D is a completed secondary exploratory experiment, not part of the
original primary Phase 4 comparison. It reused the frozen primary checkpoints,
calibrators, calibrated training predictions, incumbent, utility constants,
preprocessing, and exclusions. It reran the unchanged shared Phase 3/Phase 4
GA with fresh base seeds:

| Policy | Base seed |
|---|---:|
| random | 20270417 |
| greedy | 20270517 |
| UCB | 20270617 |
| PI | 20270717 |
| EI | 20270817 |
| MES | 20270917 |

All retained pools contained 50 candidates. Random produced one fresh
seeded-shuffle batch. Each guided policy produced utility-only and
similarity-aware batches from the identical new pool and frozen final
utilities.

For amino-acid count vector `c(x)`:

```text
s(x,y) = 0.1 * [1 - ||c(x)-c(y)||_1 / (|x|+|y|)]
```

Let `S_(k-1)={x_1,...,x_(k-1)}`. The first peptide uses raw utility:

```text
Q_1(x) = U(x)
```

For later selections:

```text
S_final(x | S_(k-1)) =
    mean over y in S_(k-1) of s(x,y)

Q_k(x) = U(x) - S_final(x | S_(k-1))
```

Selected candidates are removed from later consideration. Ties use larger
selection score, larger original utility, smaller penalty, then retained-pool
input order.

Exact overlap with the corresponding primary Phase 4 selected set was zero for
every Phase 4-D batch. This is descriptive because the pools used different
generation seeds.

| Guided policy | Change in mean normalised Levenshtein diversity | Change in mean utility |
|---|---:|---:|
| greedy | +0.20000 | -0.001652 |
| UCB | +0.10000 | -0.001660 |
| PI | +0.30045 | -0.006288 |
| EI | +0.35714 | -0.001945 |
| MES | +0.03268 | -0.001156 |

Canonical outputs are below the completed Phase 4 root at:

- `phase4d/run_001/phase4d_report.md`
- `phase4d/run_001/all_policy_tradeoffs.csv`
- `phase4d/run_001/manual_review_recommendations.csv`
- `phase4d/run_001/policy_status.csv`
- `phase4d/run_001/policies/<policy>/`

The simulation-slate decision remains manual and unresolved.

## 9. Phase 5: SPAL-Inspired Self-Paced Replay

Phase 5 is `executed_and_aggregated`: all 12 replay jobs completed, all
trajectories reached 235 labels, and aggregation reports zero missing jobs with
results claims enabled. It is a calibrated neural-model adaptation of SPAL's
easy-to-hard acquisition principle, not an exact reproduction of the
kernel/MMD/ADMM SPAL optimizer.

Canonical design:

| Setting | Value |
|---|---|
| outer folds | 1, 2, 3 |
| inner fold | 1 |
| initial labelled count | 10 |
| replay-training pool | 235 |
| validation/calibration | 59 |
| holdout | 74 |
| batch size | 5 |
| acquisition steps | 45 |
| replay points | 46 |
| model count per replay point | 1 AP_SP |
| strategies | 4 |
| jobs | 12 replay jobs + 1 aggregation job |

The strategies are `random`, `predictive_entropy`, `static_easy_entropy`, and
`self_paced_entropy`. Self-paced MI is not a primary Phase 5 strategy.

For the strict 384-dimensional penultimate embedding `h_t(x)`:

```text
h_tilde_t(x) = h_t(x) / max(||h_t(x)||_2, 1e-12)
d_t(x)       = min over l in L_t of
               ||h_tilde_t(x) - h_tilde_t(l)||_2
```

`L_t` contains only currently revealed rows from the 235-row replay-training
pool. `d_t(x)` is only an operational model-familiarity or
labelled-manifold-distance proxy.

Stable-sort the hidden pool `U_t` by increasing distance:

```text
r_t(x) = position_t(x) / (|U_t|-1), if |U_t| > 1
r_t(x) = 0, otherwise
```

The pace is:

```text
lambda_t = clip(0.30 + 0.70*t/44, 0.30, 1), t=0,...,44
```

Random uses inherited RNG selection before diagnostics are attached.
Predictive entropy considers all candidates. Static easy entropy uses
`lambda=0.30`. Self-paced entropy uses the schedule. Entropy ties preserve the
inherited stable input order.

One AP_SP model is trained per replay point. Model seeds are independent of
strategy and keyed by fold and replay point. Revealed rows are ordered by
immutable original row identifier before fitting. Validation alone controls
checkpointing and calibration. Entropy uses calibrated probability.
`pred_std` and `ensemble_mi` are not applicable.

Hidden labels are joined only after selection. With
`p_clip=clip(p_t(x),1e-6,1-1e-6)`:

```text
post_hoc_pre_query_log_loss =
    -y*ln(p_clip) - (1-y)*ln(1-p_clip)

post_hoc_pre_query_error_fixed_0_5 =
    1[1[p_t(x)>=0.5] != y]

absolute_probability_error = |y-p_t(x)|
```

Spearman proxy-validity correlations are calculated separately by fold,
strategy, and acquisition step from raw distance with average ranks for ties.

The principal normalised AULC covers 10-235 labels. Preregistered partial AULCs
cover 10-60, 10-110, and 10-160. Also report labels-to-target at 0.80, 0.84,
and 0.86 with reach counts; terminal F1, PR-AUC, Brier, and ECE-10; proxy
validity; compute time; and terminal convergence. Yield, diversity, and overlap
are budget-aware and reported at 60, 110, and 160 labels. Terminal cumulative
values are consistency checks because all policies then reveal the same rows.

Fold-matched Phase 1 AP_SP outer-fold results provide context only. Those
models use 294 development rows; Phase 5 fits at most 235 and reserves 59 for
validation/calibration. Phase 1 is not a fifth strategy or a controlled
sample-efficiency baseline.

Current artefacts:

- `thesis_results/05_self_paced_active_learning/config/phase5_config.json`
- `thesis_results/05_self_paced_active_learning/manifests/phase5_manifest.json`
- `thesis_results/05_self_paced_active_learning/manifests/aggregation_status.json`
- `thesis_results/05_self_paced_active_learning/replay/`
- `thesis_results/05_self_paced_active_learning/tables/`
- `thesis_results/05_self_paced_active_learning/figures/`
- `active_learning_thesis/phase5_self_paced.py`
- `tests/test_phase5_self_paced.py`

### 9.5 Completed results

Mean normalised AULC-F1 across the three overlapping folds was:

| Strategy | 10-60 | 10-110 | 10-160 | 10-235 |
|---|---:|---:|---:|---:|
| predictive entropy | 0.7773 | 0.8049 | 0.8183 | **0.8223** |
| random | **0.7778** | 0.7982 | 0.8101 | 0.8171 |
| self-paced entropy | 0.7687 | 0.7944 | 0.8077 | 0.8147 |
| static easy entropy | 0.7723 | 0.7846 | 0.7915 | 0.8016 |

Self-paced entropy improved over static easy entropy on the full interval by
`+0.01309` AULC-F1, but trailed predictive entropy by `-0.00768` and random by
`-0.00242`. It reached F1 0.86 in all three folds at a conditional mean of
71.7 labels; predictive entropy also reached it in all folds at 105 labels,
whereas random and static easy entropy reached it in two of three folds.

At 235 labels, every strategy within each fold converged to identical model
parameters, predictions, F1, and PR-AUC. Mean terminal holdout F1 was `0.7951`
and PR-AUC was `0.8995`. The contextual Phase 1 fold-matched means were
`0.8624` and `0.9276`, respectively, under a different 294-row training
protocol.

The operational familiarity proxy had mixed, trajectory-dependent validity.
Mean fold-step Spearman correlation with pre-query log loss was 0.3865 for
static easy entropy, 0.2399 for random, 0.2095 for self-paced entropy, and
0.0099 for predictive entropy, with wide per-step ranges.

The thesis-safe conclusion is that progressive pacing was better than a fixed
easy-only restriction, but did not improve overall AULC over ordinary
predictive entropy or random selection in this reduced retrospective study.
See `PHASE5_RESULTS_SUMMARY.md`.

## 10. Supported And Prohibited Thesis Claims

### Supported claims

- The five Phase 1 model families were reproduced under nested cross-validation.
- AP_SP recovered strong fold-matched predictive performance.
- Phase 2 compares retrospective label efficiency and strategy behaviour under
  matched fold conditions.
- Phase 3 completed eight branch-isolated CG-MD feedback rounds and terminal
  frozen-holdout evaluation.
- The retained Phase 3 CG-MD labels consistently use the documented SASA and
  final-10-ns path-contact rule.
- Primary Phase 4 completed a fixed-surrogate comparison of six proposal
  policies, generated 30 exact-unique selected sequences, and reconciled CG-MD
  evidence for 29 valid proposals.
- Phase 4-D demonstrates an operational utility-diversity trade-off on fresh
  generated retained pools.
- Phase 5 completed as a reduced retrospective replay; its mixed result is
  documented in `PHASE5_RESULTS_SUMMARY.md`.

### Prohibited or premature claims

- Do not call overlapping outer folds statistically independent repetitions.
- Do not claim prospective validation from Phase 2 or Phase 5.
- Do not describe Phase 5 as an exact SPAL reproduction.
- Do not describe Phase 4 MES as exact Gaussian-process MES.
- Do not imply that holdout predictions or metrics drove acquisition.
- Do not merge Phase 4-D into the original primary Phase 4 experiment.
- Do not present the Phase 5 result as prospective validation, exact SPAL, or
  evidence that self-pacing universally improves active learning.
- Do not call CG-MD operational labels universal biological ground truth.
- Do not claim one Phase 3 or Phase 4 policy is universally superior.
- Do not claim all primary Phase 4 selections satisfy the 3-24 length range
  without disclosing the retained length-25 MES artefact.

## 11. Artefact Map

| Evidence | Canonical path | State |
|---|---|---|
| Experimental dataset | `SA_ML_predictive/data/data_SA.csv` | present |
| Phase 1 split manifests | `thesis_results/01_reproduction/folds/` | present |
| Phase 1 predictions/tables | `thesis_results/01_reproduction/tables/` | present |
| Phase 2 evidence | `thesis_results/02_replay/evidence/` | present |
| Phase 2 thesis figures | `thesis_results/02_replay/evidence/figures/thesis_main/` | present |
| Curated thesis-results handoff | `thesis_results/THESIS_RESULTS_HANDOFF_20260623/` | present; includes presentation SVGs, print-friendly SVG variants, tables, and overview docs |
| Phase 3 complete archive | `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/` | present |
| Phase 3 ledgers/evidence | complete archive plus `branches/<branch>/` | present |
| Phase 4 complete archive | `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/` | present |
| Phase 4 audit | complete archive plus `implementation_audit/` | present |
| Phase 4 comparison | complete archive plus `comparison/round_001/` | present |
| Phase 4-D results | complete archive plus `phase4d/run_001/` | present |
| Phase 5 configuration | `thesis_results/05_self_paced_active_learning/config/phase5_config.json` | present |
| Phase 5 manifest/PBS | `thesis_results/05_self_paced_active_learning/manifests/` and `pbs/` | present |
| Phase 5 replay results | `thesis_results/05_self_paced_active_learning/replay/` | present: 12/12 complete |
| Phase 5 aggregate results | `thesis_results/05_self_paced_active_learning/tables/` and `figures/` | present |
| Thesis LaTeX source | expected thesis `.tex` source | not found |
| Bibliography | expected `.bib` file | not found |

## 12. Tests And Known Issues

Scientific workflow command executed locally on 22 June 2026:

```text
python -m unittest \
  tests.test_phase1_reproduction \
  tests.test_phase2_replay \
  tests.test_phase3_strategy_selection \
  tests.test_phase3_real_al \
  tests.test_phase4_bo \
  tests.test_phase4_diversity \
  tests.test_phase5_self_paced
```

Result after the Phase 5 aggregate corrections: `111 tests`, `OK`,
approximately `36.2 s`. Expected argparse error text
was emitted by tests deliberately rejecting superseded options
`--lambda-similarity` and `--phase4d-round`.

A separate dashboard-focused suite ran `367 tests` and ended with `20
failures` and `11 errors`.

Errors:

- `test_md_submit_blocks_readiness_duplicates_and_conflicts`
- `test_action_contract_harness_validates_bura_upload_draft_button`
- `test_action_contract_harness_validates_md_slate_launch_button`
- `test_action_contract_harness_validates_md_slate_rehearsal_button`
- `test_blocked_bura_upload_button_is_disabled_by_readiness`
- `test_guided_remote_bura_full_runner_uploads_unsynced_full_campaign`
- `test_draft_md_slate_run_action_accepts_dashboard_generated_md_batch`
- `test_draft_md_slate_run_action_creates_single_approval_and_snapshots_peptides`
- `test_pause_md_slate_marks_running_supervisor_action_paused`
- `test_resume_md_slate_requeues_paused_supervisor_action`
- `test_load_and_validate_profiles`

Failures:

- `test_guided_remote_bura_full_runner_advances_staged_campaign_by_prerequisite`
- `test_guided_remote_bura_full_runner_blocks_smoke_campaign`
- `test_guided_remote_bura_full_runner_finalizes_after_successful_copy_back_action`
- `test_guided_remote_bura_full_runner_uses_absolute_action_history_for_relative_campaign`
- `test_manual_md_sandbox_upload_is_allowed_as_first_bura_step`
- `test_operations_remote_reconciliation_fetch_logs_queues_bura_action`
- `test_render_dashboard_ladder_surfaces_bura_health_warning`
- `test_render_dashboard_ladder_view_shows_remote_commands`
- `test_render_dashboard_md_validation_runner_records_progress_memory`
- `test_render_dashboard_md_validation_shows_guided_macro_for_uploaded_campaign_step`
- `test_render_dashboard_peptides_candidate_selection_launches_saved_slate_planner`
- `test_render_dashboard_peptides_candidate_selection_runs_md_slate_rehearsal`
- `test_render_dashboard_peptides_candidate_selection_shows_md_slate_launch_panel`
- `test_render_dashboard_remote_bura_console_parses_pending_dependency_reason`
- `test_render_dashboard_remote_bura_shows_parse_after_copy_back`
- `test_render_dashboard_remote_bura_shows_readiness_check_and_console`
- `test_golden_candidate_selection_to_saved_slate_planner_to_launch_visibility`
- `test_build_md_slate_launch_readiness_reports_cap_queued_partial_plan`
- `test_build_md_slate_launch_readiness_reports_ready_plan_and_child_actions`
- `test_tick_md_slate_respects_profile_cap_before_submit`

Each listed issue concerns dashboard UI/action readiness, BURA remote
orchestration, profile loading, or MD-slate controls. None exercises the Phase
1-5 scientific methodology, predictive calibration, Phase 4 utilities,
CG-MD metric calculation, CG-MD label assignment, or Phase 3 ingestion.

Other known issues:

- The Git working tree is very dirty and includes an unresolved `.gitignore`
  conflict. Do not revert or resolve unrelated work implicitly.
- The original Phase 5 AULC aggregate duplicated fold comparisons four times;
  raw runs were unaffected and corrected tables were regenerated with
  `fold_count=3`.
- Primary Phase 4 valid proposals have been simulated and reconciled; Phase
  4-D remains a separate exploratory replicate whose simulation-slate decision
  is manual and unresolved.
- Primary Phase 4 contains the length-25 MES selection documented above.
- Exact thesis `.tex` and bibliography files are absent from this repository.

## 13. Next Thesis-Writing Tasks

1. Update methodology chapters using the formulas and source paths in this
   handoff.
2. Insert completed Phase 3 terminal results, primary Phase 4 CG-MD outcomes,
   Phase 4-D exploratory trade-offs, and completed Phase 5 results.
3. Decide whether Phase 4-D should be simulated. Primary Phase 4 has already
   been reconciled except for the disclosed length-25 MES defect.
4. Integrate the completed Phase 5 mixed/negative result without overstating
   the three overlapping folds.
5. Complete limitations, reproducibility, and computational-resource sections.
6. Add or locate the thesis LaTeX source and bibliography, then perform a final
   citation, figure, table, and cross-reference audit.
