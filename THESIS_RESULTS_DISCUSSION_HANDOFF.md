# Thesis Results And Discussion Handoff

## 1. Purpose and authority

This is the authoritative handoff for writing the **Results, Discussion,
Conclusions, and Abstract**. It does not replace canonical raw artifacts.

- Thesis source reviewed: `<local_thesis_dir>\main.tex`
- Thesis source SHA-256: `5db6eb2e9988db7cf04c360c96b4ba312de5247e92112c4b813c2a120213d70b`
- Compiled PDF reviewed: `<local_thesis_dir>\Denis_Ibiši_Master_s_thesis.pdf` (27 pages)
- PDF SHA-256: `9b6825abe591d482d4a2a1304812ce77acd46b302f43512c772c0d9ab69e3402`
- Bibliography reviewed: `<local_thesis_dir>\references_rsc.bib`
- Repository branch: `codex/active-learning-thesis`
- Commit SHA: `7cbfa95d8d56ce4b558238341ed5e19354ffcec2`
- Worktree: dirty, with unresolved conflicts; file hashes and canonical paths are therefore part of the evidence contract.
- Documentation update: `2026-06-25T00:00:00+02:00`

Source hierarchy: canonical raw/reviewed outputs, executable code/configuration,
manifests, corrected aggregates, technical documentation, then thesis prose.

## 2. Research questions

| Research question | Phases | Primary metrics | Supporting metrics | Limitations |
|---|---|---|---|---|
| RQ1: Which acquisition objectives improve retrospective label efficiency relative to random selection? | Phases 1-2 and Phase 5 | Normalised AULC-F1; paired delta vs random | Terminal F1/PR-AUC, labels-to-target, calibration, diversity | Overlapping folds; retrospective replay |
| RQ2: How do complementary acquisition strategies behave under isolated CG-MD feedback? | Phase 3 | Branch-local positive operational outcomes; terminal holdout metrics | Acquisition scores, class balance, sequence diversity and overlap | One trajectory/peptide; no cross-branch label sharing; reused holdout |
| RQ3: How do fixed-surrogate optimisation and familiarity restrictions affect proposals, diversity, and learning? | Phases 4, 4-D, 5 | Phase 4 yield+coverage; Phase 4-D utility/diversity delta; Phase 5 AULC | Lengths, overlap, familiarity correlations, eligibility, runtime | Small fixed batches; Phase 4 not closed loop; Phase 5 not exact SPAL |

## 3. Global interpretation boundaries

- Experimental labels are distinct from reviewed or threshold-derived CG-MD operational outcomes.
- Phases 2 and 5 are retrospective replay; Phase 3 is prospective computational acquisition with branch-local feedback.
- Phase 4 is fixed-surrogate proposal ranking, not adaptive improvement or closed-loop Bayesian optimisation.
- Phase 3 branches have separate ledgers even though simulations share one physical protocol.
- Phase 4-D is unsimulated and separate from primary Phase 4. Primary Phase 4
  itself has reconciled CG-MD evidence for 29 valid proposals; the known
  length-25 MES proposal is excluded from assessment denominators.
- Phase 5 is SPAL-inspired, not exact SPAL; familiarity is a model-dependent manifold-distance proxy.
- Outer folds overlap. Fold aggregates are descriptive and are not independent replications.
- One 200 ns CG-MD trajectory does not establish an experimental hit or aggregation probability.

## 4. Phase 1 canonical results

| Model | Mean F1 | SD | Range | Mean PR-AUC | Mean ROC-AUC | Mean Brier | Mean ECE-10 |
|---|---|---|---|---|---|---|---|
| AP | 0.8285 | 0.0246 | 0.8000-0.8596 | 0.9059 | 0.8234 | 0.1779 | 0.1395 |
| SP | 0.8706 | 0.0290 | 0.8333-0.9000 | 0.9396 | 0.8837 | 0.1404 | 0.1381 |
| AP_SP | 0.8686 | 0.0253 | 0.8431-0.9072 | 0.9356 | 0.8673 | 0.1481 | 0.1232 |
| TSNE_SP | 0.7924 | 0.0382 | 0.7451-0.8333 | 0.8904 | 0.7620 | 0.1980 | 0.1535 |
| TSNE_AP_SP | 0.8425 | 0.0281 | 0.8148-0.8800 | 0.9217 | 0.8567 | 0.1590 | 0.1439 |

Each row summarises five PR-threshold outer-test folds. Threshold-dependent
accuracy/F1 and threshold-independent ROC-AUC/PR-AUC are kept separate in
`thesis_reporting/phase1_outer_fold_metrics.csv`. AP_SP was retained as the
later surrogate because it reproduced the hybrid architecture central to the
project, achieved mean F1 0.8686,
and exposes the joint 384-dimensional penultimate representation used by later
workflows. SP has slightly higher mean F1 and PR-AUC in this reproduction; AP_SP
must not be called uniquely best on Phase 1 metrics.

Deployment-style all-data fits are not fold-matched evidence and must be
reported separately if used.

## 5. Phase 2 canonical results

### n0=10

| Strategy | AULC-F1 | Terminal F1 | PR-AUC | F1=.86 reach | Conditional labels | Positive selections |
|---|---|---|---|---|---|---|
| predictive_entropy | 0.8284 | 0.8463 | 0.9217 | 4/5 | 58.7500 | 285/500 |
| ensemble_mi | 0.8145 | 0.8362 | 0.9048 | 3/5 | 88.3333 | 319/500 |
| hybrid_mi_diverse | 0.8126 | 0.8372 | 0.9129 | 3/5 | 93.3333 | 331/500 |
| family_qbc | 0.8105 | 0.8506 | 0.9057 | 3/5 | 60.0000 | 237/500 |
| random | 0.8064 | 0.8222 | 0.9011 | 2/5 | 87.5000 | 336/500 |
| cluster_diverse_representative | 0.8058 | 0.8543 | 0.9086 | 3/5 | 80.0000 | 351/500 |
| oed_logdet | 0.8050 | 0.8369 | 0.9018 | 2/5 | 95.0000 | 341/500 |
| ucb | 0.8001 | 0.7976 | 0.8756 | 1/5 | 65.0000 | 446/500 |
| similarity_penalized_mean | 0.8000 | 0.8303 | 0.8953 | 2/5 | 62.5000 | 436/500 |
| ensemble_mean | 0.7949 | 0.8249 | 0.8926 | 2/5 | 72.5000 | 432/500 |

### n0=40

| Strategy | AULC-F1 | Terminal F1 | PR-AUC | F1=.86 reach | Conditional labels | Positive selections |
|---|---|---|---|---|---|---|
| family_qbc | 0.8290 | 0.8402 | 0.9209 | 4/5 | 96.2500 | 268/500 |
| predictive_entropy | 0.8279 | 0.8239 | 0.9189 | 3/5 | 106.6667 | 276/500 |
| ensemble_mi | 0.8261 | 0.8242 | 0.9218 | 2/5 | 77.5000 | 310/500 |
| cluster_diverse_representative | 0.8239 | 0.8324 | 0.9161 | 4/5 | 81.2500 | 356/500 |
| oed_logdet | 0.8195 | 0.8259 | 0.9183 | 3/5 | 81.6667 | 332/500 |
| hybrid_mi_diverse | 0.8165 | 0.8384 | 0.9170 | 2/5 | 100.0000 | 333/500 |
| ucb | 0.8155 | 0.8326 | 0.9098 | 2/5 | 87.5000 | 438/500 |
| similarity_penalized_mean | 0.8138 | 0.8070 | 0.9070 | 2/5 | 95.0000 | 439/500 |
| random | 0.8108 | 0.8313 | 0.8999 | 2/5 | 75.0000 | 342/500 |
| ensemble_mean | 0.8064 | 0.8246 | 0.9058 | 1/5 | 80.0000 | 439/500 |

The two conditions cover different labelled-count intervals and are never
pooled as a single AULC experiment. Fold-wise values, ranges, worst folds,
labels-to-target reach counts, positive yield, diversity, overlap, and paired
differences from random are in `thesis_reporting/`.

The calibration/ensemble ablation is a separate result family. PI, EI and MES
were not principal Phase 2 benchmark strategies.

## 6. Phase 3 strategy-selection evidence

The executable composite was:

`C = 0.35*N_up(AULC_F1) + 0.25*N_down(L_0.86) + 0.15*N_down(R_fold) + 0.15*N_up(D_Lev) + 0.10*N_down(J_mean) + B_role`.

`N` is min-max normalisation over non-excluded strategies; a constant component
receives 1.0. The role bonus was 0.04 for committee-uncertainty or
diversity/novelty roles. Labels-to-target used the implementation's incomplete
target penalty before scoring. Strategies below random combined AULC by more
than the `1e-6` tolerance were exploratory/control. Recommendation assembly was
sequential: best eligible AULC, then an eligible committee-uncertainty role,
then an eligible diversity role satisfying the Jaccard gate where possible.

| Strategy | Decision | Role | Combined AULC | Composite | Diversity | Max overlap |
|---|---|---|---|---|---|---|
| predictive_entropy | recommended | best replay/sample efficiency | 0.8281 | 0.8416 | 0.8614 | 0.5390 |
| family_qbc | recommended | model/committee uncertainty | 0.8197 | 0.7950 | 0.8702 | 0.5390 |
| cluster_diverse_representative | recommended | diversity/novelty for CG-MD slate | 0.8149 | 0.7240 | 0.8979 | 0.3853 |
| ensemble_mi | backup | model/uncertainty | 0.8203 | 0.5964 | 0.8840 | 0.4922 |

Predictive entropy was the best eligible combined-AULC strategy. Family QBC was
chosen as the committee representative, not simply the second-highest global
score. Cluster-diverse was the highest-composite eligible diversity-oriented
choice satisfying the assembly logic, but OED logdet had the largest measured
Levenshtein diversity. Ensemble MI was the highest-ranked unselected backup.

The code generated evidence and a recommendation, followed by an explicit human
adoption in `real_al_strategy_recommendation.md`. There is no evidence that this
rule was preregistered before inspecting Phase 2 results.

## 7. Phase 3 campaign results

| Branch | Positives | Yield | Mean diversity |
|---|---|---|---|
| predictive_entropy | 19 | 0.4750 | 0.9023 |
| family_qbc | 21 | 0.5250 | 0.9100 |
| cluster_diverse_representative | 19 | 0.4750 | 0.9129 |

| Branch | Terminal F1 | PR-AUC | Brier | ECE-10 | Validation threshold |
|---|---|---|---|---|---|
| predictive_entropy | 0.8762 | 0.9113 | 0.1497 | 0.1131 | 0.4583 |
| family_qbc | 0.8713 | 0.9255 | 0.1411 | 0.1238 | 0.5450 |
| cluster_diverse_representative | 0.8750 | 0.9243 | 0.1465 | 0.1465 | 0.3909 |

Each branch started with 235 labels, acquired five labels in each of eight
rounds, and ended with 275. All 120 sequences were unique across branches.
Round-level sequences, acquisition scores, predictions, CG-MD metrics, labels,
and cumulative positives are in `phase3_round_outcomes.csv`.

The same 74-peptide holdout was used once after terminal retraining. Its outer
fold condition had also contributed to Phase 2 strategy-family selection, so
the terminal branch comparison is descriptive rather than a wholly independent
external validation. Canonical per-round validation trajectories were not
located in the completed archive.

## 8. Shared CG-MD evidence

The operational criterion was:

`AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`.

The inventory contains 120 Phase 3 reviewed/ingested simulations and 29 primary
Phase 4 complete simulations. Every canonical sequence has one retained
trajectory; replica uncertainty is absent. The source metrics, expected
criterion outcome, formal review state, and provenance are recorded in
`shared_cgmd_inventory.csv`. These are computational outcomes, not experimental
validation.

## 9. Primary Phase 4 canonical results

| Policy | Positive/assessed | CG-MD yield | Coverage | Selected diversity |
|---|---|---|---|---|
| random | 3/5 | 0.6000 | 1.0000 | 0.8600 |
| greedy | 2/5 | 0.4000 | 1.0000 | 0.4800 |
| ucb | 1/5 | 0.2000 | 1.0000 | 0.6100 |
| pi | 5/5 | 1.0000 | 1.0000 | 0.0810 |
| ei | 3/5 | 0.6000 | 1.0000 | 0.7690 |
| mes | 1/4 | 0.2500 | 0.8000 | 0.0829 |

The 30 archived records are exact-unique across policies. One 25-residue MES
record (`VLNINNMGAKWRRTCNQRLTPTALP`) lies outside the intended 3-24 range and
was archived, excluded from simulation, unassessed, not replaced, and is not a
negative. Therefore 29 eligible records correspond to 29 unique simulations.

The campaign files contain complete metrics but blank formal label/reviewer
fields. Phase 4 outcomes are consequently described as **threshold-derived
operational outcomes**, not human-reviewed labels. The evidence is sufficient
to write Phase 4 Results if this distinction and the raw counts are preserved.
Utility magnitudes must not be compared across policies with different scales.

## 10. Phase 4-D canonical results

`phase4d_tradeoffs.csv` preserves the controlled within-pool comparison between
utility-only and similarity-aware final selectors. Utility and diversity
changes are attributable to final selection because each pair shares a retained
pool and frozen utilities. Comparisons with primary Phase 4 are descriptive
only because Phase 4-D used fresh GA seeds and fresh pools. No Phase 4-D
candidate has CG-MD evidence, so no yield is reported.

## 11. Phase 5 canonical results

| Strategy | AULC 10-60 | AULC 10-110 | AULC 10-160 | AULC 10-235 | Terminal F1 | PR-AUC |
|---|---|---|---|---|---|---|
| random | 0.7778 | 0.7982 | 0.8101 | 0.8171 | 0.7951 | 0.8995 |
| predictive_entropy | 0.7773 | 0.8049 | 0.8183 | 0.8223 | 0.7951 | 0.8995 |
| static_easy_entropy | 0.7723 | 0.7846 | 0.7915 | 0.8016 | 0.7951 | 0.8995 |
| self_paced_entropy | 0.7687 | 0.7944 | 0.8077 | 0.8147 | 0.7951 | 0.8995 |

Only the corrected aggregation with `fold_count=3` is authoritative. The prior
paired-comparison aggregation duplicated fold comparisons; raw jobs were
unaffected. Predictive entropy led full AULC. Self-paced entropy exceeded
static easy entropy over the full interval, but did not exceed predictive
entropy or random. Proxy-validity, eligibility, runtime, target reach, and
terminal convergence records are indexed in `thesis_reporting/`.

## 12. Cross-phase synthesis

| Question | Observation | Interpretation | Limitation | Strength |
|---|---|---|---|---|
| Which strategies improved retrospective label efficiency? | Phase 2 predictive entropy had the strongest combined holdout AULC among eligible strategies; Phase 5 predictive entropy had the highest full AULC, while self-paced entropy exceeded static easy entropy but not predictive entropy or random. | Uncertainty sampling was useful, but additional pacing restrictions did not consistently improve the whole learning trajectory. | Retrospective replay with overlapping folds; no independent replication. | supported descriptively |
| Did retrospective performance translate into computational discovery? | Phase 3 branches produced similar overall operational positive yields (19/40, 21/40, 19/40). Primary Phase 4 threshold-derived yields varied from 1/5 to 5/5, with very small policy batches. | Replay ranking did not map monotonically onto branch-local CG-MD yield; fixed-surrogate policies explored very different sequence regimes. | One trajectory per peptide, no wet-lab validation, and only five Phase 4 records per policy. | suggestive |
| How did uncertainty, disagreement, representativeness, diversity, and exploitation differ? | The Phase 3 selection deliberately retained predictive entropy, family QBC, and cluster-diverse roles. Phase 4 generated policy-specific sequence families and Phase 4-D increased diversity at small utility cost within fresh pools. | Acquisition objectives shape candidate composition and redundancy as well as predictive score. | Phase 3 role selection was post hoc and Phase 4-D differs from primary Phase 4 in generation seed and retained pool. | supported descriptively |
| What did fixed-surrogate ranking show? | Primary Phase 4 generated 30 unique archived records, assessed 29, and yielded 15 threshold-positive outcomes; PI selected five closely related long sequences and all five met the operational criterion. | A fixed surrogate can concentrate proposals in high-scoring motifs, but apparent yield may coincide with low within-policy diversity. | Single round, policy-specific utility scales, small batches, and no retraining. | suggestive |
| What did Phase 5 show about familiarity restrictions? | Self-paced entropy improved over permanently easy-only entropy in full AULC but remained below predictive entropy and random; familiarity distance often correlated positively with pre-query loss. | The proxy contains useful difficulty information, but restricting selection by it can sacrifice informative hard examples. | Model-dependent proxy, single AP_SP member per replay point, and three overlapping folds. | supported descriptively |

## 13. Discussion framework

| RQ | Supported claim | Evidence | Interpretation | Alternative | Limitation | Literature | Strength |
|---|---|---|---|---|---|---|---|
| RQ1 | Predictive entropy was consistently competitive and led Phase 5 full AULC; Phase 2 effects depended on n0 and strategy. | Phase 2 summaries and Phase 5 corrected AULCs. | Uncertainty can focus labels near the decision boundary. | Non-monotonic trajectories and initialization can favour random. | Overlapping folds and small dataset. | Settles; Barrett & White; Evans et al. | supported descriptively |
| RQ2 | All three Phase 3 branches found computational positives, with similar overall yield and modest terminal metric differences. | 19/40, 21/40, and 19/40 positives plus terminal holdout metrics. | Complementary acquisition roles explore different regions while maintaining model performance. | Threshold and sequence-length effects may dominate branch strategy. | One trajectory, fixed CG model, no wet lab. | Shmilovich et al.; Talluri et al.; Thapa et al. | supported descriptively |
| RQ3 | Primary Phase 4 produced 15 threshold positives among 29 assessed records; Phase 4-D exposed utility-diversity trade-offs; self-pacing did not beat predictive entropy overall. | Phase 4 inventory, Phase 4-D trade-offs, Phase 5 corrected summaries. | Goal-directed utilities strongly shape motif concentration and diversity. | Surrogate bias, fresh-pool confounding, and familiarity-proxy artifacts. | Single fixed-surrogate round and three overlapping Phase 5 folds. | Njirjak et al.; Di Fiore et al.; Tang & Huang. | suggestive |

Required limitations: small and imbalanced dataset; overlapping folds; no
independent external experimental test set; calibration dependence; no
cross-member latent-axis averaging; post hoc role-constrained Phase 3 strategy
selection; branch-local feedback; one trajectory per peptide; sequence-length
effects; fixed force field and operational thresholds; 200 ns duration; no
wet-lab validation; single-round fixed-surrogate Phase 4; 25-residue boundary
artifact; Phase 4-D fresh-pool confounding; model-dependent familiarity proxy;
and corrected Phase 5 aggregation.

## 14. Literature-comparison map

| Thesis finding | Related citation | Relation | Permitted claim | Gap |
|---|---|---|---|---|
| Hybrid AP_SP reproduction supports surrogate use | Njirjak et al. (2024) | Reproduction/extension | The thesis reproduces a competitive AP_SP predictor and extends it into calibrated active-learning and proposal workflows. | Verify exact paper metric wording before final prose. |
| Retrospective uncertainty sampling can improve label efficiency but is condition-dependent | Settles (2009); Barrett & White (2021); Evans et al. (2014) | Agreement with conditional AL benefit | Results support strategy- and initialization-dependent gains rather than universal AL superiority. | Targeted quotation check recommended. |
| CG-MD can support peptide discovery prioritization | Shmilovich et al. (2020); Thapa et al. (2024); Talluri et al. (2025) | Methodological extension | The thesis uses CG-MD as computational evidence in iterative and fixed-surrogate selection workflows. | Literature details and experimental validation rates require targeted search. |
| BO-style utilities emphasize goal-directed candidate ranking | Jones et al. (1998); Srinivas et al. (2010); Wang & Jegelka (2017); Di Fiore et al. (2024) | Adaptation | PI, EI, UCB and MES were adapted to calibrated neural-ensemble probability space in a single fixed-surrogate round. | Do not imply exact GP BO or closed-loop BO. |
| Self-paced restriction embodies easy-to-hard acquisition | Tang & Huang (2019) | SPAL-inspired adaptation, not reproduction | The familiarity-percentile schedule tests an easy-to-hard principle with neural embeddings. | Exact SPAL objective and solver differ; discuss explicitly. |

Bibliography entries were inspected, but paper-specific numerical claims were
not re-audited here. Rows marked with a gap require targeted literature
verification before detailed comparison prose.

## 15. Conclusions contract

### Permitted

- The five predictive families were reproduced, and AP_SP provided a documented joint surrogate representation.
- Some acquisition strategies improved retrospective label efficiency under specific replay conditions.
- Phase 3 completed three isolated eight-round computational feedback loops and found operational positives in every branch.
- Primary Phase 4 demonstrates policy-dependent fixed-surrogate proposal concentration and threshold-derived CG-MD outcomes.
- Phase 4-D demonstrates a controlled within-pool utility-diversity trade-off.
- Phase 5 shows that progressive familiarity pacing is better than permanently easy-only selection over the full trajectory, but not better than predictive entropy or random in this reduced replay.

### Prohibited

- One strategy is universally best.
- CG-MD proves experimental self-assembly.
- Phase 4 is closed-loop Bayesian optimisation.
- Phase 4-D candidates were simulated or have yield.
- Overlapping folds are independent replications.
- Phase 5 measures intrinsic chemical difficulty or exactly reproduces SPAL.
- The 25-residue MES record is an in-range proposal or a negative outcome.
- One trajectory estimates aggregation probability.

## 16. Abstract evidence sheet

- **Background:** Data-efficient discovery of self-assembling peptides requires predictive models, acquisition strategies, and computational validation under limited labelled data.
- **Objective:** Evaluate retrospective label efficiency, branch-local CG-MD feedback, fixed-surrogate proposal policies, diversity-aware selection, and self-paced familiarity restrictions.
- **Methods:** Nested predictive reproduction, calibrated ensemble replay, three isolated eight-round CG-MD loops, one-round fixed-surrogate proposal comparison, a diversity-aware replicate, and reduced SPAL-inspired replay.
- **Quantitative results:**
  - Phase 1 AP_SP achieved mean outer-fold F1 0.8686 and PR-AUC 0.9356.
  - Phase 2 predictive entropy had combined strategy-selection AULC-F1 0.8281, compared with random 0.8086.
  - Phase 3 yielded 19/40, 21/40, and 19/40 operational positives for predictive entropy, family QBC, and cluster-diverse branches.
  - Primary Phase 4 assessed 29/30 archived records and produced 15 threshold-positive outcomes; policy yields ranged from 1/5 to 5/5, with MES coverage 4/5.
  - Phase 5 full AULC-F1 was 0.8223 for predictive entropy and 0.8147 for self-paced entropy.
- **Interpretation:** Acquisition objectives changed both learning efficiency and the composition of proposed peptide sets, but benefits were strategy-, budget-, and protocol-dependent.
- **Principal limitation:** Evidence is computational and retrospective, with overlapping folds, one trajectory per simulated peptide, and no wet-laboratory validation.
- **Conclusion:** The project supports a calibrated, uncertainty-aware and diversity-conscious computational workflow while rejecting claims of universal strategy superiority.

The final abstract must be written only after Results, Discussion, and
Conclusions are approved.

## Writing-model safeguards

Do not invent missing values, estimate from figures when tables exist, imply
causation from descriptive comparisons, call CG-MD experimental ground truth,
call Phase 4 closed-loop BO, call Phase 5 exact SPAL, call folds independent,
compare cross-policy utility magnitudes, treat unassessed records as negative,
report Phase 4-D yield, claim cluster-diverse was most sequence-diverse, call
the Phase 3 rule preregistered, or hide null/contradictory results.
