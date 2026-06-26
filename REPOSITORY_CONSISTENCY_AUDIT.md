# Repository Consistency Audit

Current consolidation date: 22 June 2026.

The authoritative thesis handoff is now
`THESIS_HANDOFF_FOR_NEXT_MODEL.md`. This audit remains the focused scientific
consistency record for Phase 2, Phase 3 CG-MD labels, and Phase 4. When older
status wording conflicts with the handoff, the handoff takes precedence.

Original Phase 4 audit date: 21 June 2026.

## Phase 4 source of truth

The generated audit under
`thesis_results/04_bayesian_optimization/implementation_audit/` is the
authoritative description of the thesis Phase 4 experiment.

Verified implementation:

- branch-neutral, single-round fixed-surrogate comparison;
- fixed 235-row model-fitting and 59-row validation/calibration partitions;
- 74-row frozen holdout excluded from fitting, checkpoint selection,
  calibration, threshold selection, incumbent calculation, prediction,
  acquisition, ranking, and selection;
- holdout identities used only for exact duplicate exclusion;
- five independently seeded `AP_SP` members;
- member-wise Phase 3 calibration before ensemble aggregation;
- population ensemble SD (`ddof=0`);
- surrogate-space incumbent equal to the maximum calibrated ensemble mean over
  the 235 training peptides;
- one immutable checkpoint/calibrator/incumbent set shared by all guided
  policies;
- validation \(F_1\)-maximising threshold used only for reporting;
- policies limited to random, calibrated greedy, calibrated UCB,
  probability-space approximate PI, probability-space approximate EI, and
  calibrated ensemble-based approximate MES;
- final MES ranking based on one common retained-pool set of five member-wise
  maxima.

Entropy, expected entropy, mutual information, and disagreement remain
diagnostics and do not define extra Phase 4 policies.

Within the shared generative algorithm and retained-pool selection path, Phase
4 changes only the calculation of \(U_{\mathrm{policy}}(x)\); all Phase 3
penalty functions, genetic-algorithm mechanics, pool construction, and selector
behaviour remain unchanged.

## Generic discovery boundary

`run-discovery` is legacy/generic support. It may load a latest post-ingest
ensemble, fall back to a baseline ensemble, and use
`discovery_mes_samples=128`. None of those behaviours defines the thesis Phase
4 experiment, and generic discovery outputs are not Phase 4 thesis evidence.

## Operational CG-MD contact metric

The full-profile simulation parameter contract has been checked separately
against the canonical `.mdp` templates, generated scripts and topologies, and
completed GROMACS production logs. The authoritative transcription is
`THESIS_CGMD_PARAMETER_CONTRACT.md`. In particular, the production
`constraints=none` setting does not remove explicit Martini peptide or
polarizable-water topology constraints; completed production used LINCS for
those constraints.

The retained Phase 3 operational label uses:

```text
AP_sasa(200 ns) >= 1.75
AND paper_path_APcontact_last10ns >= 0.5
```

`paper_path_APcontact_last10ns` is implemented in
`active_learning_thesis.md_workflow._paper_path_ap_contact_for_gro` and
`_write_paper_path_last10_ap_contact_file`. Minimum inter-bead distances are
converted using the piecewise paper weight. The maximum mean
consecutive-edge Hamiltonian-path score is calculated by exact dynamic
programming for at most 16 peptide copies and deterministic beam search for
larger systems. Eleven frames from 190 through 200 ns are averaged.

The repository also calculates:

- `AP_contact`: fraction of molecules with at least one inter-peptide bead
  contact within 0.6 nm;
- `paper_APcontact`: smooth strongest-contact diagnostic;
- `AP_contact_same_paper_formula`: strongest-contact aggregation with the
  piecewise paper distance weight;
- `paper_path_APcontact`: path score at individual target frames.

The 0.6 nm contacted-molecule fraction and other contact variants are
diagnostics. They do not define the retained Phase 3 label.

The retained Phase 3 backup contains 120 reviewed rows across rounds 1-8.
All 120 contain `paper_path_APcontact_last10ns`, all 120 evidence summaries
name that field, and all 120 labels match the rule above. Historical campaign
directories may contain earlier diagnostic files or pre-reparse review files;
they were not rewritten by this audit.

## Official Phase 2 protocol

The official replay benchmark uses:

1. random;
2. ensemble mean;
3. similarity-penalised mean;
4. predictive entropy;
5. ensemble mutual information;
6. UCB;
7. family QBC;
8. cluster-diverse representative selection;
9. OED logdet;
10. hybrid MI-diverse.

The \(n_0=10\) and \(n_0=40\) regimes are reported separately. Batch size is
five and the maximum is 20 acquisition rounds. Labels-to-target thresholds are
0.80, 0.84, and 0.86, with 0.84 as the principal practical target. No
interpolation is performed. Unreached targets remain not reached, and reach
counts or fractions accompany successful-fold labels-to-target values.

PI, EI, and MES are not main Phase 2 replay strategies.

## Artifact policy

This audit updates current code, tests, runbooks, and thesis-facing
documentation. It does not rewrite historical Phase 1-3 result files,
previously ingested labels, MD outputs, or archived Phase 4 previews.
No local or cluster jobs are submitted automatically.

## Verified execution status

- Phase 1: `executed_and_aggregated`.
- Phase 2: `executed_and_aggregated`.
- Phase 3: `executed_and_aggregated`; the canonical completed source is
  `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/`.
- Primary Phase 4: `executed_and_aggregated`; the canonical completed source is
  `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/`.
- Phase 4-D: `executed_and_aggregated` under the completed Phase 4 source at
  `phase4d/run_001/`.
- Phase 5: `executed_and_aggregated`; all 12 replay jobs completed and the
  corrected aggregate reports zero missing jobs and permits results claims.

The original Phase 5 AULC aggregate repeated each fold-level comparison once
per strategy key, producing `fold_count=12`. Raw replay outputs were unaffected.
The aggregation loop now emits one comparison per unique fold condition, a
regression test protects the behaviour, and tables/figures were regenerated
from the fetched completed outputs with `fold_count=3`.

Primary Phase 4 selected 30 exact-unique peptides across six policies. One
historical MES selection has length 25 despite the declared 3-24 range. The
completed artefact is retained unchanged and must be disclosed in simulation
or thesis use.
