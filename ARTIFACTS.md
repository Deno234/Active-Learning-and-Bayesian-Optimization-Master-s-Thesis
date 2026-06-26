# Artefact Inventory

## Curated Thesis Artefacts

The primary thesis-facing artefacts are in:

```text
thesis_results/THESIS_RESULTS_HANDOFF_20260623/
```

Important subfolders:

```text
00_overview/
01_phase1_reproduction/
02_phase2_replay/
03_phase3_real_active_learning/
04_phase4_bayesian_optimization/
05_phase5_self_paced/
cgmd_peptide_evidence/
```

Each phase folder contains the final figures and tables used for thesis writing
or appendix support. The `00_overview` folder contains figure/table indexes,
claim caveats, and provenance notes.

## Cross-phase Reporting Tables

`thesis_reporting/` contains CSV summaries assembled for thesis writing and
consistency checks. These files are derived evidence, not raw simulation output.

## Models And Checkpoints

Two compact trained-model bundles are included:

```text
models/phase4_ap_sp_fixed_split_ensemble/
models/phase5_initial_replay_point_000/
models/phase3_round001_pre_proposal/
```

The Phase 4 bundle is the five-member `AP_SP` ensemble used for the primary
fixed-surrogate proposal experiment. It includes the `.h5` checkpoints,
member-wise calibration file, training/validation prediction exports,
surrogate-space incumbent, and model manifest.

The Phase 5 bundle contains only the `replay_point_000` initial models for the
three reduced replay folds and four strategies. These are included because they
are compact and useful for checking the initial self-paced replay setup.

The Phase 3 bundle contains the Round 1 pre-proposal branch models for
predictive entropy, family QBC, and cluster-diverse representative selection.
For family QBC it also includes the five family-model checkpoints used by that
strategy.

The checkpoint files are tracked with Git LFS.

The remaining replay checkpoints from Phase 5 are not included. They are large
intermediate files and are not needed to inspect the thesis figures and tables.
Phase 2 checkpoint binaries are not included because the available replay cache
contains tens of thousands of files and is too large for a thesis GitHub
repository.

## Release Manifest

`RELEASE_FILE_MANIFEST.csv` lists every file in this release and its SHA-256
hash. Regenerate it with:

```bash
python scripts/validate_release.py --write-manifest
```
