# Figure Review Record

Review date: 2026-06-23; print-readability update: 2026-06-25

- The original curated handoff contained 34 canonical SVG figures. They were
  parsed successfully as XML and rendered in the local browser gallery with
  non-zero dimensions.
- The handoff now also contains 19 print/Overleaf readability variants with
  suffixes such as `_thesis_print.svg`, `_thesis_paper_compact.svg`, and
  `_thesis_clean.svg`. These variants alter titles, whitespace, font sizes,
  tick visibility, legends, or label placement only; they do not change the
  plotted data.
- No included figure contains the former `Matplotlib unavailable` placeholder.
- Phase 4 threshold lines, axis labels, policy legend, and operational-positive
  marker outlines were visually checked.
- Phase 4 policy yield values were cross-checked against the 29 sequence-level
  completed simulation rows.
- Phase 3 terminal F1 and PR-AUC bars were cross-checked against each branch's
  `final_holdout.json`.
- Phase 5 learning curves, paired AULC differences, labels-to-target bars,
  familiarity diagnostics, overlap heatmap, diversity bars, and compute-time
  chart were visually inspected for clipping, missing series, and overlapping
  labels.
- The selected Phase 2 and Phase 1 figures are canonical repository figures,
  not recomputed approximations.

Print variants currently available include Phase 2 learning curves, AULC bars,
labels-to-target, Jaccard heatmaps, Phase 3 cumulative positives, the primary
Phase 4 CG-MD threshold scatter, and Phase 5 holdout/eligibility figures. Use
`00_overview/FIGURE_INDEX.csv` to distinguish canonical base figures from
print variants.

Remaining interpretive caution: small fold or policy sample sizes are stated in
the captions and `CLAIMS_AND_CAVEATS.md`; visual polish does not increase their
statistical evidential strength.
