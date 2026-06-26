# Master's Thesis Results Handoff (20260623; documentation refreshed 20260625)

This packet is a curated, lightweight result bundle for writing the thesis Results section. It contains canonical aggregate tables, presentation-ready SVG figures, the newly reconciled primary Phase 4 CG-MD evidence, and the project documentation required to interpret the results safely.

The canonical scientific tables and base figures were assembled on 23 June
2026. Several SVGs were later polished for print/Overleaf readability without
changing the underlying data. Prefer files ending in `_thesis_print.svg`,
`_thesis_paper_compact.svg`, or `_thesis_clean.svg` when a corresponding base
figure is visually too small in the thesis PDF; cite the same data source and
caption as the canonical base figure.

## Start here

1. `00_overview/CLAIMS_AND_CAVEATS.md`
2. `00_overview/phase_status.csv`
3. `00_overview/FIGURE_GALLERY.html`
4. `00_overview/FIGURE_REVIEW.md`
5. `00_overview/DATA_PROVENANCE.md`
6. `00_overview/FIGURE_INDEX.csv`
7. `00_overview/TABLE_INDEX.csv`
8. `00_overview/project_documentation/THESIS_HANDOFF_FOR_NEXT_MODEL.md`

For a new ChatGPT model, send the files in `00_overview/project_documentation/`
plus `00_overview/CLAIMS_AND_CAVEATS.md`, `00_overview/DATA_PROVENANCE.md`,
`00_overview/FIGURE_INDEX.csv`, `00_overview/TABLE_INDEX.csv`, and the specific
phase result summaries needed for the chapter being written.

## Phase 4 CG-MD reconciliation

- Selected proposals: 30
- Complete simulations with both operational metrics: 29
- Operational positives: 15
- Missing simulation: `VLNINNMGAKWRRTCNQRLTPTALP` (MES, 25 residues, known invalid-length proposal)
- Operational label: `AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`
- The last-10-ns AP-SASA value is exported as a diagnostic, but it is not substituted for the implemented 200 ns AP-SASA criterion.

## Scope

The ZIP intentionally excludes neural-network checkpoints, raw trajectories, temporary scheduler output, and duplicated exploratory archives. Canonical source paths are retained in the tables and manifest.
