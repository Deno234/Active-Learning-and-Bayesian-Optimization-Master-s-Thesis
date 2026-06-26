# Phase 5 Results: SPAL-Inspired Self-Paced Replay

## Status

Phase 5 is `executed_and_aggregated`.

- completed replay jobs: 12/12;
- outer folds: 1, 2, and 3;
- strategies: random, predictive entropy, static easy entropy, and self-paced
  entropy;
- replay points per trajectory: 46;
- acquisition steps per trajectory: 45;
- labelled-count interval: 10-235;
- aggregation status: complete;
- results claims allowed: true.

The three outer folds are overlapping fold-level repetitions, not statistically
independent replicates.

## Main Result

Ordinary predictive entropy achieved the highest mean full-interval
normalised AULC-F1. Random selection was second, self-paced entropy was third,
and the fixed easy-only entropy restriction was weakest.

| Strategy | AULC-F1 10-60 | AULC-F1 10-110 | AULC-F1 10-160 | Full AULC-F1 10-235 |
|---|---:|---:|---:|---:|
| predictive entropy | 0.7773 | 0.8049 | 0.8183 | **0.8223** |
| random | **0.7778** | 0.7982 | 0.8101 | 0.8171 |
| self-paced entropy | 0.7687 | 0.7944 | 0.8077 | 0.8147 |
| static easy entropy | 0.7723 | 0.7846 | 0.7915 | 0.8016 |

For the full 10-235 interval, mean paired AULC-F1 differences across the three
fold conditions were:

- self-paced minus predictive entropy: `-0.00768`;
- self-paced minus random: `-0.00242`;
- self-paced minus static easy entropy: `+0.01309`;
- static easy entropy minus predictive entropy: `-0.02077`.

Self-pacing therefore recovered performance relative to a permanently
restricted easy-candidate policy, particularly at intermediate and full
budgets, but it did not improve the overall trajectory over ordinary
predictive entropy or random selection in this reduced experiment.

## Labels To Target

| Strategy | F1 target | Folds reaching target | Conditional mean labels |
|---|---:|---:|---:|
| predictive entropy | 0.80 | 3/3 | 33.3 |
| predictive entropy | 0.84 | 3/3 | **56.7** |
| predictive entropy | 0.86 | 3/3 | 105.0 |
| random | 0.80 | 3/3 | **23.3** |
| random | 0.84 | 3/3 | 71.7 |
| random | 0.86 | 2/3 | 112.5 |
| self-paced entropy | 0.80 | 3/3 | 38.3 |
| self-paced entropy | 0.84 | 3/3 | 71.7 |
| self-paced entropy | 0.86 | 3/3 | **71.7** |
| static easy entropy | 0.80 | 3/3 | 53.3 |
| static easy entropy | 0.84 | 3/3 | 75.0 |
| static easy entropy | 0.86 | 2/3 | 102.5 |

The self-paced strategy reached F1 0.86 in all three folds and did so earlier
on average than predictive entropy. This is a useful secondary observation,
but it does not overturn the AULC result: target crossing can be affected by
non-monotonic trajectory fluctuations and is based on only three overlapping
fold conditions.

## Terminal Convergence And Phase 1 Context

At 235 revealed training rows, all four strategies within each fold had:

- the same 235 sequences in the same canonical order;
- the same model seed;
- identical parameters;
- identical holdout predictions;
- identical F1 and PR-AUC.

All maximum differences recorded in the terminal convergence audit were zero,
within the predefined `1e-6` tolerance.

Across the three folds, the common terminal Phase 5 model had:

- mean holdout F1: `0.7951`;
- mean holdout PR-AUC: `0.8995`.

The contextual fold-matched Phase 1 models, trained on 294 development rows,
had:

- mean F1: `0.8624`;
- mean PR-AUC: `0.9276`.

The descriptive Phase 5-minus-Phase 1 gaps were `-0.0672` F1 and `-0.0282`
PR-AUC. This is not a controlled sample-efficiency comparison because the
training and validation protocols differ.

## Familiarity-Proxy Diagnostics

Mean per-step Spearman correlations between raw familiarity distance and
post-hoc pre-query log loss were:

| Strategy trajectory | Mean Spearman | Positive correlations |
|---|---:|---:|
| static easy entropy | 0.3865 | 94.8% |
| random | 0.2399 | 91.1% |
| self-paced entropy | 0.2095 | 80.0% |
| predictive entropy | 0.0099 | 50.4% |

Each row summarises 135 fold-step correlations. Individual values varied
widely and included negative correlations. Repeated steps within a trajectory
are not independent observations. The proxy therefore has descriptive,
trajectory-dependent validity, not a universal interpretation as intrinsic
peptide difficulty.

## Yield And Runtime

Random selection had the highest cumulative positive-label yield at the fixed
budgets:

| Labels | Random | Predictive entropy | Self-paced entropy | Static easy entropy |
|---:|---:|---:|---:|---:|
| 60 | 0.667 | 0.573 | 0.447 | 0.513 |
| 110 | 0.680 | 0.560 | 0.490 | 0.527 |
| 160 | 0.682 | 0.573 | 0.558 | 0.580 |

This does not directly measure classifier improvement: uncertainty-based
strategies deliberately query difficult examples and may select more negative
or ambiguous peptides.

The 12 jobs consumed approximately `5.73` summed GPU-hours. Mean wall time was
`28.64` minutes per job, with a range of `27.71-30.98` minutes.

## Thesis-Safe Conclusion

The reduced Phase 5 experiment did not show that self-paced entropy improves
overall label efficiency over ordinary predictive entropy or random selection.
It did show that gradually relaxing the familiarity restriction is preferable
to retaining a fixed easy-only restriction over the full trajectory. The
familiarity-distance proxy was positively associated with pre-query error in
many trajectory-step conditions, but the association was heterogeneous and
should be presented as operational rather than intrinsic.

These findings are retrospective and based on three overlapping folds. They do
not constitute prospective peptide validation or an exact reproduction of the
original SPAL optimiser.

## Canonical Evidence

- `thesis_results/05_self_paced_active_learning/manifests/aggregation_status.json`
- `thesis_results/05_self_paced_active_learning/tables/learning_curves.csv`
- `thesis_results/05_self_paced_active_learning/tables/paired_aulc_differences.csv`
- `thesis_results/05_self_paced_active_learning/tables/paired_aulc_summary.csv`
- `thesis_results/05_self_paced_active_learning/tables/labels_to_target_summary.csv`
- `thesis_results/05_self_paced_active_learning/tables/terminal_convergence_audit.csv`
- `thesis_results/05_self_paced_active_learning/tables/phase1_contextual_baseline.csv`
- `thesis_results/05_self_paced_active_learning/tables/proxy_validity_summary.csv`
- `thesis_results/05_self_paced_active_learning/tables/selected_positive_yield.csv`
- `thesis_results/05_self_paced_active_learning/tables/sequence_diversity.csv`
- `thesis_results/05_self_paced_active_learning/tables/selection_overlap.csv`
- `thesis_results/05_self_paced_active_learning/tables/compute_time.csv`
- `thesis_results/05_self_paced_active_learning/figures/`

## Aggregation Correction

The original aggregate duplicated each fold-level AULC comparison once per
strategy key, producing `fold_count=12` instead of 3. The raw replay outputs
were unaffected. The aggregator was corrected to emit one comparison per
unique `(outer_fold, initial_label_count)` condition, covered by a regression
test, and the aggregate tables and figures were regenerated locally from the
fetched completed replay outputs.
