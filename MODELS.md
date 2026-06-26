# Model Artefacts

This file explains which trained model files are included in the release and
which are not.

## Included Model Bundles

### Phase 4 Fixed-surrogate Ensemble

```text
models/phase4_ap_sp_fixed_split_ensemble/
```

Included files:

```text
member_calibrations.json
model_manifest.json
training_incumbent.json
training_predictions.csv
validation_calibration_report.json
validation_predictions.csv
ensemble/ap_sp_member_00.h5
ensemble/ap_sp_member_01.h5
ensemble/ap_sp_member_02.h5
ensemble/ap_sp_member_03.h5
ensemble/ap_sp_member_04.h5
ensemble/*.h5.meta.json
```

The `.h5` files are tracked with Git LFS. After cloning the repository, use:

```bash
git lfs pull
```

If Git LFS is not installed, the `.h5` files may appear as small pointer files
instead of actual Keras/HDF5 model files.

### Phase 5 Initial Replay Models

The release also includes the initial `replay_point_000` Phase 5 models:

```text
models/phase5_initial_replay_point_000/
```

These are the trained initial AP_SP models for:

```text
outer folds: 1, 2, 3
initial labelled count: 10
strategies: random, predictive_entropy, static_easy_entropy, self_paced_entropy
```

Each folder contains:

```text
embedding_manifest.json
ensemble/ap_sp_member_00.h5
ensemble/ap_sp_member_00.h5.meta.json
```

Only the initial replay point is included. The later Phase 5 replay models are
intermediate checkpoints generated after additional acquisitions.

### Phase 3 Round 1 Pre-proposal Models

The release includes the compact Phase 3 Round 1 pre-proposal model bundle:

```text
models/phase3_round001_pre_proposal/
```

This bundle contains:

```text
predictive_entropy/pre_proposal/ensemble/*.h5
family_qbc/pre_proposal/ensemble/*.h5
family_qbc/pre_proposal/family/*.h5
cluster_diverse_representative/pre_proposal/ensemble/*.h5
```

These are the branch models before any Phase 3 CG-MD labels were acquired.
They are useful because they document the common starting point of the
prospective Phase 3 branches. The full round-by-round Phase 3 model history is
not included.

## Why Only This Model Bundle Is Included

The Phase 4 bundle is small and directly tied to the final proposal experiment.
It is useful for inspecting the fixed surrogate without retraining. The Phase 3
Round 1 pre-proposal models and Phase 5 initial models are also compact and are
useful for checking the starting point of their corresponding workflows.

Phase 2 checkpoint binaries are not included because the available replay cache
contains 22,300 files and occupies about 11.58 GiB. That cache is not suitable
for a normal thesis GitHub repository. Phase 2 configurations, metrics,
predictions, and result summaries are retained.

Phase 5 replay checkpoints exist in the full working tree, but they are not
all included in this GitHub release. The full set contains 1,104 checkpoint
files and occupies about 588 MiB. Most of those files are replay intermediates
rather than final models. If the full checkpoint archive needs to be published
later, a better home would be a separate GitHub Release, Zenodo record, OSF
project, or institutional data repository.

## Recreating Excluded Models

Excluded replay models can be regenerated with the relevant Phase 2, Phase 3, or
Phase 5 workflow commands, provided the same software environment and random
seeds are used. Exact floating-point identity may still depend on TensorFlow,
GPU drivers, and hardware.

## Optional Recovery From SUPEK

If additional historical model binaries are later recovered from SUPEK, place
them under `models/` rather than inside `thesis_results/`. Suggested locations:

```text
models/phase1_reproduction/
models/phase2_initial_or_reference/
models/phase3_initial_or_final/
```

Recommended SUPEK search commands from the project root are:

```bash
find thesis_results SA_ML_predictive active_learning_runs \
  -type f \( -name '*.h5' -o -name '*.keras' -o -name '*.pkl' -o -name '*.joblib' \) \
  | sort
```

For Phase 1, look first for reproduced model-family checkpoints or saved
training outputs under:

```text
SA_ML_predictive/
thesis_results/01_phase1*
thesis_results/phase1*
```

For Phase 2, useful compact choices would be initial replay models rather than
all replay checkpoints, for example files from the first replay point of each
configuration if they exist:

```text
thesis_results/02_replay/**/replay_point_000/**
thesis_results/02_replay/**/initial*/**/models/**
```

For Phase 3, useful compact choices would be either the common pre-proposal
Round 1 ensembles or the final post-Round-8 branch ensembles, if the binaries
exist:

```text
thesis_results/03_real_al/branches/*/models/real_al/round_001/pre_proposal/
thesis_results/03_real_al/branches/*/models/real_al/round_008/
thesis_results/03_real_al/branches/*/models/final*
```

After copying any recovered model binaries into this repository, update
`.gitignore`, `.gitattributes`, `scripts/validate_release.py`, and this file so
the added artefacts are explicitly allowed and tracked through Git LFS. Avoid
committing large checkpoint caches unless they are scientifically necessary.

## Checksums

The release-level file manifest:

```text
RELEASE_FILE_MANIFEST.csv
```

contains SHA-256 checksums for the files included in this repository.
