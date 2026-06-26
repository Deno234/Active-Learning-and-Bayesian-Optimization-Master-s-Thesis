# ML-guided self-assembling peptide discovery

This repository contains the cleaned code and result artefacts for my Master's
thesis project on machine-learning-guided discovery of self-assembling peptides.

It is not a complete copy of my working directory. I kept the files that are
useful for checking the implementation and thesis results, and removed raw
trajectory files, temporary archives, scheduler logs, local dashboard state, and
large replay checkpoint caches.

## What Is In This Repository

The project combines:

- reproduced neural-network predictors for peptide self-assembly;
- retrospective active-learning replay on the experimental peptide dataset;
- CG-MD feedback campaigns for selected peptides;
- a one-round fixed-surrogate proposal comparison;
- a secondary diversity-aware proposal replicate;
- a SPAL-inspired self-paced replay experiment.

The main predictive model used in later phases is the hybrid `AP_SP` model,
which combines aggregation-propensity and sequential physicochemical features.

## Directory Layout

```text
active_learning_thesis/       Main thesis implementation and CLI.
tests/                        Unit and regression tests retained for the release.
SA_ML_predictive/             Reproduced predictive-model support code/data.
SA_ML_generative/             Genetic-algorithm support code.
MD/                           CG-MD preparation templates and helper scripts.
models/                       Compact trained Phase 4 AP_SP ensemble bundle.
thesis_results/               Curated final figures, tables, and evidence files.
thesis_reporting/             Cross-phase summary tables.
docs/                         Extra notes and the original copied README.
scripts/                      Release validation helper.
```

The main thesis evidence package is:

```text
thesis_results/THESIS_RESULTS_HANDOFF_20260623/
```

## Setup

```bash
conda env create -f environment.yml
conda activate ml_peptide_self_assembly
```

The trained Phase 4 model checkpoints are tracked with Git LFS. After cloning
from GitHub, run:

```bash
git lfs pull
```

## Basic Checks

```bash
python scripts/validate_release.py
python -m unittest tests.test_phase3_strategy_selection tests.test_phase4_bo tests.test_phase4_diversity tests.test_phase5_self_paced
```

These checks do not rerun all expensive workflows. They verify the release
structure and several important code paths used by the thesis.

## Command Line Entry Point

Most workflows are exposed through:

```bash
python -m active_learning_thesis --help
```

The most relevant implementation files are:

- `active_learning_thesis/phase1_reproduction.py`
- `active_learning_thesis/phase2_replay.py`
- `active_learning_thesis/phase3_strategy_selection.py`
- `active_learning_thesis/phase3_real_al.py`
- `active_learning_thesis/phase4_bo.py`
- `active_learning_thesis/phase4_diversity.py`
- `active_learning_thesis/phase5_self_paced.py`

## Where To Start Reading

- `THESIS_HANDOFF_FOR_NEXT_MODEL.md`
- `THESIS_METHODOLOGY_TECHNICAL_SUMMARY.md`
- `THESIS_RESULTS_DISCUSSION_HANDOFF.md`
- `THESIS_FEATURE_RUNTIME_SHAPES.md`
- `THESIS_PREDICTIVE_MODEL_ARCHITECTURE.md`
- `REPOSITORY_CONSISTENCY_AUDIT.md`
- `PHASE5_RESULTS_SUMMARY.md`
- `MODELS.md`
- `thesis_results/THESIS_RESULTS_HANDOFF_20260623/README.md`

## Models

This repository includes two compact trained-model bundles:

```text
models/phase4_ap_sp_fixed_split_ensemble/
models/phase5_initial_replay_point_000/
models/phase3_round001_pre_proposal/
```

The first bundle is the five-member `AP_SP` ensemble used for the primary Phase
4 fixed-surrogate proposal experiment. The second contains the initial
`replay_point_000` Phase 5 models for the three fold conditions and four
strategies. The third contains the Phase 3 Round 1 pre-proposal branch models,
including the family-model set used by family QBC. Phase 2 checkpoint binaries
are not included because the available Phase 2 cache is too large for this
release. The remaining Phase 5 replay checkpoints are also much larger and are
not included here. See `MODELS.md` for details.

## Important Limitations

- The Phase 2 replay folds are overlapping fold-level conditions, not
  statistically independent replicates.
- CG-MD labels are operational computational labels from retained trajectories,
  not experimental validation.
- Phase 4 is a one-round fixed-surrogate proposal comparison, not a closed-loop
  Bayesian-optimisation campaign.
- Phase 4-D is a secondary diversity-aware replicate and was not simulated.
- Phase 5 is SPAL-inspired; it is not an exact reproduction of the original SPAL
  method.

## Citation

If using this repository, cite the thesis and the upstream scientific work
listed in `CITATION.cff` and the thesis bibliography.
