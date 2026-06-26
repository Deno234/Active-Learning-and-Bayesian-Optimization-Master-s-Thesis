# Data Notes

## Included Data

The release includes the small peptide dataset and feature lookup arrays needed
by the reproduced predictive pipeline:

```text
SA_ML_predictive/data/data_SA.csv
SA_ML_predictive/data/amino_acids_AP.npy
SA_ML_predictive/data/dipeptides_AP.npy
SA_ML_predictive/data/tripeptides_AP.npy
SA_ML_predictive/data/TSNE_SP_1_component.npy
SA_ML_predictive/data/TSNE_SP_2_components.npy
SA_ML_predictive/data/TSNE_SP_3_components.npy
```

The thesis uses 368 unique peptide sequences with binary experimental labels.
Phase-specific train/validation/holdout splits and replay ledgers are documented
in the result artefacts and methodology files.

## CG-MD Operational Labels

The operational CG-MD criterion used in Phases 3 and primary 4 was:

```text
AP_SASA_legacy_200ns >= 1.75
AND
AP_contact_path_190_200ns_mean >= 0.5
```

The auditable peptide-level exports are:

```text
thesis_results/THESIS_RESULTS_HANDOFF_20260623/cgmd_peptide_evidence/
```

Phase 4-D candidates were not simulated and are therefore not included as CG-MD
result rows.

## Excluded Data

The release intentionally excludes:

- raw GROMACS trajectories (`.xtc`, `.trr`);
- binary run inputs/checkpoints (`.tpr`, `.cpt`, `.edr`);
- local cluster downloads and scheduler logs;
- temporary dashboard state;
- duplicated historical archives.

These exclusions keep the GitHub repository reviewable and avoid publishing
machine-specific state.

## Model Artefacts

The included Phase 4 model bundle is documented in `MODELS.md`. It is a trained
artefact, not a raw dataset. It is retained so that the final fixed-surrogate
proposal experiment can be inspected without retraining the ensemble.
