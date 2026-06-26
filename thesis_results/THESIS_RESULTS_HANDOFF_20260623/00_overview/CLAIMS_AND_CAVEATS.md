# Results Claims And Caveats

## Supported results

- Phase 1 reproduced the five predictive model families under nested cross-validation.
- Phase 2 completed the ten-strategy retrospective replay for `n0=10` and `n0=40`.
- Phase 3 completed 120 branch-local CG-MD acquisitions across three strategies and eight rounds.
- Primary Phase 4 produced 30 exact-unique proposals. Twenty-nine valid proposals were simulated and have complete operational CG-MD evidence; the known 25-residue MES proposal was not simulated.
- Phase 4-D completed as a separate exploratory diversity-aware replicate.
- Phase 5 completed 12 replay jobs and aggregation. Predictive entropy had the highest mean full-interval AULC-F1; self-paced entropy exceeded static easy entropy but not predictive entropy or random.

## Mandatory caveats

- The outer folds overlap and are fold-level repetitions, not statistically independent replicates.
- Phase 2 and Phase 5 are retrospective replay experiments, not prospective peptide validation.
- CG-MD operational labels are modelled simulation outcomes, not universal biological ground truth.
- Phase 3 branches are isolated trajectories; labels were not shared across branches.
- Primary Phase 4 policy yields are based on only five proposals per policy, and MES has four simulated proposals because the invalid length-25 item was excluded.
- Phase 4-D is not part of the primary Phase 4 policy comparison.
- Phase 5 is SPAL-inspired and uses a neural familiarity proxy; it is not an exact reproduction of SPAL.
- Cross-phase predictive metrics are not directly interchangeable because training-set sizes and validation/calibration protocols differ.
