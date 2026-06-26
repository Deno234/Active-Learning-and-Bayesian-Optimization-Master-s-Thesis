# Data Provenance

## Canonical scientific archives

- Phase 1: `thesis_results/01_reproduction/`
- Phase 2: `thesis_results/02_replay/evidence/` and the associated benchmark overlap tables
- Phase 3: `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/`
- Phase 4 and Phase 4-D: `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/`
- Phase 5: `thesis_results/05_self_paced_active_learning/`

## Primary Phase 4 simulation evidence

The primary proposal list comes from the completed Phase 4 archive. Simulation
evidence is resolved by exact peptide sequence from these three local campaign
roots:

- `active_learning_runs/thesis_main_20260502/md_campaigns/`
- `active_learning_runs/thesis_main_supek_20260502/md_campaigns/`
- `active_learning_runs/thesis_main_supek_clean_20260502_original/md_campaigns/`

For each selected sequence, a campaign is accepted only when `md_review.csv`
reports `job_root_status=analysis_complete` and contains both `ap_200ns` and
`paper_path_ap_contact_last10ns_mean`. When duplicate review rows exist, a
complete row is preferred over an incomplete prepared package. No conflicting
complete metric pairs were found.

## Derived outputs

Every derived table is explicitly stored in the corresponding phase `tables/`
directory. Figures are generated from those copied or derived tables. Original
checkpoints, trajectories, scheduler output, and large candidate-level archives
are intentionally excluded from this writing packet.

## Figure variants

The base SVG figures listed in `00_overview/FIGURE_INDEX.csv` are the canonical
data visualisations. Later `_thesis_print.svg`, `_thesis_paper_compact.svg`,
`_title_removed.svg`, and `_thesis_clean.svg` files are print-readability
variants prepared for thesis layout. They may remove in-figure titles or
subtitles, enlarge labels/ticks/legends, adjust whitespace, or use clearer
colours, but they should not be treated as new scientific analyses.
