# Reproducibility Notes

I have separated reproducibility into three practical levels. The first two can
be checked from this repository on a normal workstation. The third level needs
substantial compute time and, for the original runs, access to SUPEK/BURA-like
HPC systems.

## Level 1: Inspect The Final Evidence

No HPC access is required.

Use:

```text
thesis_results/THESIS_RESULTS_HANDOFF_20260623/
thesis_reporting/
THESIS_METHODOLOGY_TECHNICAL_SUMMARY.md
THESIS_RESULTS_DISCUSSION_HANDOFF.md
```

This level supports inspection of the final thesis figures, tables, CG-MD
evidence exports, and cross-phase claims without rerunning expensive workflows.

## Level 2: Re-run Lightweight Analysis And Tests

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ml_peptide_self_assembly
```

Run:

```bash
python scripts/validate_release.py
python -m unittest tests.test_phase1_reproduction tests.test_phase2_replay tests.test_phase3_strategy_selection tests.test_phase4_bo tests.test_phase4_diversity tests.test_phase5_self_paced
```

These checks verify the main code paths and curated artefact structure. They do
not rerun the full replay campaigns or CG-MD simulations.

## Level 3: Re-run Expensive Workflows

The following require substantial runtime and, for the original project,
SUPEK/BURA-like HPC resources:

- Phase 2 full retrospective replay and ablations;
- Phase 3 branch-isolated active-learning campaigns;
- Phase 4 fixed-surrogate training and proposal generation;
- Phase 5 self-paced replay;
- CG-MD campaign construction, execution, and analysis.

The historical runbooks are retained:

- `SUPEK_RUNBOOK.md`
- `BURA_MD_RUNBOOK.md`

The runbooks contain cluster-specific examples. Treat user names, paths, queues,
and modules as site-specific placeholders when adapting them elsewhere.

## Included Trained Models

Two compact trained-model bundles are included:

```text
models/phase4_ap_sp_fixed_split_ensemble/
models/phase5_initial_replay_point_000/
models/phase3_round001_pre_proposal/
```

The `.h5` checkpoint files are tracked with Git LFS. If they appear as small
text pointer files after cloning, run:

```bash
git lfs pull
```

Phase 2 checkpoint binaries are not included because the available replay cache
is very large. The remaining Phase 5 replay checkpoints are deliberately
excluded because they are large replay intermediates; see `MODELS.md`.

## Known External Or Hard-to-repeat Elements

- Raw production trajectories are not included because they are large.
- Scheduler stdout/stderr logs and local dashboard state are not included.
- Some exact wall-clock timings depend on the HPC system.
- TensorFlow/GPU determinism can vary across hardware and library versions.

## Regenerating Thesis Evidence Packages

The scripts:

```text
build_thesis_reporting_evidence.py
build_thesis_results_handoff.py
```

were used to assemble thesis-facing evidence packages from canonical outputs.
They are retained so the packaging logic is auditable.
