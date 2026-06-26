# Thesis Canonical Results Index

Status vocabulary: **canonical**, **corrected canonical**, **derived reproducibly**, **superseded**, and **unresolved**.

| Phase | Result family | Metric/outcome | Canonical file | Row/key/filter | Aggregation | Denominator | Fold/replicate count | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Predictive reproduction | Outer-fold model-family metrics | thesis_reporting/phase1_model_family_summary.csv | one row/model | mean, sample SD, range | 5 folds/model | 5 overlapping folds | derived reproducibly | PR-threshold outer-test evidence |
| 1 | Architecture selection | Selected cells/kernel | thesis_results/01_reproduction/tables/hyperparameter_summary.csv | one row/model | lowest mean inner-validation loss | inner folds | nested CV | canonical | Deployment fits must remain separate |
| 2 | Strategy replay n0=10 | AULC, terminal metrics, yield, diversity | thesis_reporting/phase2_strategy_summary_n0_10.csv | one row/strategy | fold summaries plus selected-record yield | 500 selected records/strategy | 5 overlapping folds | derived reproducibly | n0 intervals kept separate |
| 2 | Strategy replay n0=40 | AULC, terminal metrics, yield, diversity | thesis_reporting/phase2_strategy_summary_n0_40.csv | one row/strategy | fold summaries plus selected-record yield | 500 selected records/strategy | 5 overlapping folds | derived reproducibly | n0 intervals kept separate |
| 2 | Paired comparison | Differences versus random | thesis_reporting/phase2_paired_vs_random.csv | canonical rows | matched fold differences | 5 fold pairs | 5 overlapping folds | corrected canonical | No independent-fold significance claim |
| 2 | Calibration/ensemble ablation | F1, AULC, Brier, ECE | thesis_results/02_replay/evidence/ablation_summary.csv | setup x n0 x dataset | canonical aggregate | 5 folds | 5 overlapping folds | canonical | Separate from policy benchmark |
| 3 selection | Role-constrained recommendation | Composite score and decision | thesis_reporting/phase3_strategy_selection_evidence.csv | one row/strategy | executable min-max composite plus role assembly | 10 strategies | combined n0 evidence | canonical | Post hoc; not preregistered |
| 3 | CG-MD campaign | Round outcomes and acquisition scores | thesis_reporting/phase3_round_outcomes.csv | branch/round/sequence | exact selected-review join | 120 simulations | one trajectory/sequence | derived reproducibly | Human-reviewed and ingested |
| 3 | Terminal comparison | Holdout F1, PR-AUC, calibration | thesis_reporting/phase3_terminal_holdout.csv | one row/branch | direct JSON transcription | 74 holdout peptides | same holdout per branch | canonical | Descriptive, not wholly independent external validation |
| 3+4 | Shared CG-MD | SASA, path contact, criterion outcome | thesis_reporting/shared_cgmd_inventory.csv | one row/unique simulated sequence | threshold conjunction | 149 trajectories | one trajectory/sequence | derived reproducibly | Phase 4 formal review fields blank |
| 4 | Primary assessment | Policy yield and coverage | thesis_reporting/phase4_policy_summary.csv | one row/policy | positives/assessed; assessed/5 | 5 archived records/policy | 29 assessed total | derived reproducibly | Unassessed MES record is not negative |
| 4 | Primary record inventory | Sequence-level utilities and outcomes | thesis_reporting/phase4_assessment_inventory.csv | 30 archived records | exact-sequence reconciliation | 30 selected, 29 assessed | single fixed surrogate | derived reproducibly | No cross-policy exact duplicates |
| 4-D | Diversity-aware replicate | Utility/diversity trade-off | thesis_reporting/phase4d_tradeoffs.csv | one row/guided policy | same-pool paired selectors | 5+5 sequences/policy | fresh pool | canonical | No CG-MD yield |
| 5 | Self-paced replay | Full/partial AULC, terminal metrics | thesis_reporting/phase5_strategy_summary.csv | one row/strategy | corrected fold_count=3 summary | 3 folds/strategy | 3 overlapping folds | corrected canonical | Not exact SPAL |
| 5 | Proxy validity | Distance-log-loss Spearman | thesis_reporting/phase5_proxy_summary.csv | one row/strategy | descriptive fold-step aggregation | 135 correlations/strategy | repeated steps non-independent | derived reproducibly | Familiarity is model-dependent |

## Superseded and unresolved items

See `thesis_reporting/superseded_and_unresolved.csv`. Raw or canonical files were not overwritten.
