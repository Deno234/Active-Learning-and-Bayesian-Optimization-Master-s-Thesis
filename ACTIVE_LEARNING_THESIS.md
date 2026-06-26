# Active-Learning Thesis Workflow

> **Current thesis source of truth (22 June 2026):**
> [`THESIS_HANDOFF_FOR_NEXT_MODEL.md`](THESIS_HANDOFF_FOR_NEXT_MODEL.md).
> It records the verified execution status and canonical completed artefact
> paths. When this operational guide conflicts with the handoff or a generated
> implementation audit, the handoff and audit take precedence.

This repository now includes a first implementation of the thesis workflow in
the `active_learning_thesis` package. The new workflow keeps the existing
predictive models and the genetic algorithm, but adds:

- a reproducible experimental split
- a central peptide ledger
- a replay benchmark for acquisition-function comparison
- a round-based real active-learning loop with CSV CG-MD export/import
- a separate discovery mode with BO-style GA fitness functions

Practical cluster runbooks:

- `SUPEK_RUNBOOK.md` for the ML side on Supek
- `BURA_MD_RUNBOOK.md` for the MD side on BURA

## Commands

Initialize a run:

```bash
python -m active_learning_thesis init-run --run-name thesis_run
```

If you want TensorFlow to use an NVIDIA GPU on native Windows, recreate or
update the environment from `ml_peptide_self_assembly.yml`. The environment file
pins the last officially supported native-Windows TensorFlow GPU stack:

```bash
conda env update -f ml_peptide_self_assembly.yml --prune
conda activate ml_peptide_self_assembly
python -c "import tensorflow as tf; print(tf.__version__); print(tf.test.is_built_with_cuda()); print(tf.config.list_physical_devices('GPU'))"
```

Run the offline replay benchmark:

```bash
python -m active_learning_thesis run-replay --run-dir active_learning_runs/thesis_run
```

Run a resumable multi-seed replay study for thesis-grade evidence:

```bash
python -m active_learning_thesis run-study --study-name thesis_full_replay --seeds 5 --epochs 70 --max-rounds 10 --target 0.85
```

This creates per-seed run directories, reuses completed runs when resumed, writes
`active_learning_runs/_studies/<study-name>/study_manifest.json`, and then
summarizes only the completed study runs into the study `evidence/` folder.
Use `--dry-run` first if you want to inspect the planned run names and seeds
before launching expensive training.

New runs use validation-calibrated probabilities for acquisition by default.
Use `--raw-acquisition` to create a matched raw-probability study for thesis
ablation:

```bash
python -m active_learning_thesis run-study --study-name thesis_raw_replay --seeds 5 --epochs 70 --max-rounds 10 --target 0.85 --raw-acquisition
```

Compare matched studies, for example raw acquisition versus calibrated
acquisition:

```bash
python -m active_learning_thesis compare-studies --baseline-study thesis_raw_replay --candidate-study thesis_full_replay --metric f1 --target 0.85
```

This writes paired seed/strategy deltas, strategy-level AULC advantages,
label-efficiency estimates, and a short thesis narrative under
`active_learning_runs/_studies/_comparisons/`.

Aggregate replay runs into thesis-ready evidence tables:

```bash
python -m active_learning_thesis summarize-study --run-root active_learning_runs --metric f1 --target 0.85
```

This writes run/strategy summaries, AULC rankings, label-efficiency estimates,
and paired comparisons against `random` under
`active_learning_runs/_study_evidence/`. Use it after replay runs when you want
evidence for the thesis discussion rather than one-off smoke metrics.

Propose the next CG-MD batch:

```bash
python -m active_learning_thesis propose-round --run-dir active_learning_runs/thesis_run
```

Ingest returned CG-MD labels:

```bash
python -m active_learning_thesis ingest-round --run-dir active_learning_runs/thesis_run --import-csv path/to/round_001_labels.csv
```

Run BO-style discovery on the latest trained ensemble:

```bash
python -m active_learning_thesis run-discovery --run-dir active_learning_runs/thesis_run
```

Evaluate the final frozen holdout once, after development is finished:

```bash
python -m active_learning_thesis evaluate-final --run-dir active_learning_runs/thesis_run
```

Freeze the official thesis result after `evaluate-final` has produced
`metrics/final_holdout.json`:

```bash
python -m active_learning_thesis freeze-final --run-dir active_learning_runs/thesis_run
```

This writes `final_freeze/final_freeze.json`,
`final_freeze/freeze_checks.csv`, `final_freeze/model_artifacts.csv`, and
`final_freeze/model_card.md`. The freeze checks that the final metric is really
marked as holdout, the ledger has no unresolved proposed batch, terminal full-MD
review rows are evidence-backed and ingested or converted to `cgmd_ingest.csv`,
and the model/config/ledger/import artifacts are fingerprinted. Use
`--run-evaluation` if you want the command to run `evaluate-final` first, and
`--allow-unresolved` only when intentionally preserving an unresolved state for
audit rather than declaring it clean.

Prepare a BURA-safe MD campaign from an exported batch:

```bash
python -m active_learning_thesis prepare-md-campaign --run-dir active_learning_runs/thesis_run --batch-csv active_learning_runs/thesis_run/batches/round_001_batch.csv --campaign round_001_bura --md-profile line_smoke
```

Build PDBs locally with PyMOL, or validate manually provided PDBs:

```bash
python -m active_learning_thesis build-pdbs --campaign-dir active_learning_runs/thesis_run/md_campaigns/round_001_bura
python -m active_learning_thesis build-pdbs --campaign-dir active_learning_runs/thesis_run/md_campaigns/round_001_bura --validate-only
```

Parse finished MD results and convert a reviewed file into the standard ingest schema:

```bash
python -m active_learning_thesis parse-md-results --campaign-dir active_learning_runs/thesis_run/md_campaigns/round_001_bura
python -m active_learning_thesis make-md-ingest-csv --campaign-dir active_learning_runs/thesis_run/md_campaigns/round_001_bura --review-csv active_learning_runs/thesis_run/md_campaigns/round_001_bura/md_review.csv
```

Launch the local monitoring dashboard over your synced run tree:

```bash
python -m active_learning_thesis dashboard --run-root active_learning_runs
```

The dashboard is now a local admin cockpit. It can still run the safe local actions (`prepare-md-stage`, `finalize-md-stage`, `make-md-ingest-csv`, `propose-round`, `run-discovery`, and `evaluate-final`), and it can now also draft, approve, and track allowlisted Supek/BURA actions over system `ssh`/`scp`. Mutating remote actions remain human-approved, while queue polling stays read-only. Action logs live under `active_learning_runs/_dashboard_actions/`, and remote sync state / cluster snapshots live under `active_learning_runs/_dashboard_remote_state/`.
Use Today -> Guided thesis checklist, or Operations -> Thesis checklist, when
you want the dashboard to show the current thesis phase and the next safe click.
The checklist is read-only and follows Setup -> Run -> Study -> MD -> Ingest ->
Freeze -> Export using existing run, study, MD, freeze, packet, and figure
artifacts.
Use Operations -> Action debugger when a GUI action fails or appears stale. It
reads action metadata plus stdout/stderr tails, classifies likely causes such as
missing paths, SSH auth, scheduler issues, config mismatch, MD artifact gaps, or
environment/import problems, and shows one safe next move before retrying. The
panel also emits a copy-friendly debug packet for lab notes or support.

Run the seeded thesis canary when you want a deterministic end-to-end rehearsal
of the local thesis loop without TensorFlow training or remote cluster access:

```bash
python -m active_learning_thesis thesis-canary --run-root active_learning_runs --seed 20260425 --peptides 2 --force
```

The canary writes an isolated run under `active_learning_runs/_thesis_canaries/`.
It creates a proposed batch, prepares a full MD campaign, injects synthetic
AP/SASA/trajectory outputs, parses `md_review.csv`, saves evidence-backed
labels, creates `cgmd_ingest.csv`, validates/imports the labels, writes a
synthetic post-ingest metric, creates a next-round batch stub, and saves
`canary_report.json` plus `canary_report.md`. Its metrics and labels are
synthetic; use them to verify workflow contracts, not as scientific evidence.

Export the thesis evidence packet when you want one folder of copy-ready tables,
figure data, canary status, MD review evidence, and reproducibility metadata:

```bash
python -m active_learning_thesis export-thesis-packet --run-root active_learning_runs --metric f1
```

By default this writes under `active_learning_runs/_thesis_packets/`. The packet
includes `README.md`, `packet_manifest.json`, `tables/runs.csv`,
`tables/md_review_evidence.csv`, `tables/canary_reports.csv`,
`tables/final_freezes.csv`, `tables/study_artifacts.csv`, `tables/metrics.csv`,
`figure_data/learning_curves.csv`, dashboard lifecycle/readiness tables when the
dashboard state can be collected, and `metadata/reproducibility.json` with git
commit/status information. Use `--skip-dashboard` for a disk-only export.

Build an output-only thesis figure bundle from an exported packet:

```bash
python -m active_learning_thesis build-thesis-figures --packet-dir active_learning_runs/_thesis_packets/<packet-folder>
```

This writes dependency-free SVG figures, clean source tables, and caption
Markdown under `<packet-folder>/thesis_figures/`. The generated bundle includes
final scorecard, replay learning-curve, MD review evidence, strategy AULC, and
discovery-utility figures when the packet contains enough data. It never changes
the active-learning run, dashboard state, final freeze, or original thesis
packet; treat it as a writing/export layer.

Optional sidecar: if you decide to report external-evaluator disagreement or
curriculum-style peptide complexity as a "by the way" thesis analysis, keep it
separate from the main AL loop:

```bash
python -m active_learning_thesis optional-evaluator-study --run-dir active_learning_runs/thesis_run --external-scores path/to/external_scores.csv
```

The external score CSV should contain `sequence` plus one score column named
`score`, `external_score`, `probability`, `prob_self_assembly`, or `pred_mean`.
An optional `evaluator` column lets you compare multiple external sources. By
default the command writes under
`active_learning_runs/thesis_run/_optional_evaluator_study/` and does not modify
the ledger, dashboard state, thesis packet, or final freeze. Outputs include
`tables/evaluator_disagreement.csv`, `tables/complexity_bins.csv`,
`tables/complexity_summary.csv`, and a standalone manifest/README. You can also
run it without `--external-scores` to generate only the transparent peptide
complexity bins.

Operations -> Remote jobs also includes a read-only remote heartbeat autopilot.
It reconciles tracked SUPEK/BURA sync records, queue snapshots, cluster health,
MD slate state, artifact checks, and recovery rows into one verdict per remote
item: `watch`, `needs_check`, `needs_recovery`, `ready`, `staged`, or
`complete`. Use its suggested read-only follow-ups first, such as polling the
queue, fetching logs, or refreshing health/readiness, before approving any
mutating recovery action.

For dashboard remote actions on Windows, use native OpenSSH rather than Pageant:

```powershell
Get-Service ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
ssh-add -l
ssh supek "hostname"
ssh bura "hostname"
```

If BURA requires FortiClient, connect VPN before the `ssh bura` check. The older Pageant helper workflow is deprecated and unsupported by the dashboard.

## MD Validation

The MD integration is intentionally **semi-automated** and **BURA-first**:

- PDB generation stays local.
- Heavy MD work must run only through Slurm jobs on BURA.
- Login/access nodes are for staging, the exact `find . -type f -name "*.sh" -exec dos2unix {} \+` command, module inspection, preflight checks, and `sbatch` only.
- The repo never auto-assigns `cgmd_label`. Remote cluster actions are now available in the dashboard, but every mutating Supek/BURA action still requires explicit admin approval.
- `md_review.csv` is the human-review step before producing the exact `ingest-round` CSV. New review files also carry `label_rubric`, `label_confidence`, `label_evidence_tags`, `label_evidence_summary`, `reviewer`, and `reviewed_at` so every model-feedback label has thesis-readable evidence behind it.
- MD Validation -> Review & ingest includes an evidence packet for AP/SASA outputs, a structured label rubric (`self_assembling`, `not_self_assembling`, or `uncertain_rerun`), confidence, tags, and a short evidence summary. Newly structured review rows must be evidence-backed before the dashboard offers `cgmd_ingest.csv` creation; older notes-only rows remain usable as legacy review evidence.
- `prepare-md-campaign` now requires `--md-profile {line_smoke,production_smoke,full}`.
- Guided MD automation is available through `prepare-md-stage`, `finalize-md-stage`, and `md-ladder-status` for the single-peptide `line_smoke -> production_smoke -> full` ladder.

Recommended BURA execution order:

1. `line_smoke`
2. `production_smoke`
3. `full`

Profile behavior:

- `line_smoke`: all minimization, equilibration, and production `.mdp` files are rewritten to `nsteps = 200`, and the generated chain stops after `10_Dynamics_b.sh`
- `production_smoke`: minimization and equilibration keep template values; only `martini_22P_md.mdp` is rewritten to `nsteps = 200`, and the generated chain stops after `10_Dynamics_b.sh`
- `full`: all `.mdp` files keep the bundled template values, including the 200 ns production run needed for AP targets through 200 ns, and the generated chain continues through `11_SASA_and_FrameDump.sh` and `12_AP_calc.sh`

Generated MD chain semantics:

- smoke profiles validate package generation, submission, and dynamics execution only
- `full` is the only profile that includes SASA/AP post-analysis
- local finalization also computes an explicit diagnostic `AP_contact` companion metric from returned `.gro` frame dumps when they are available: the score is the fraction of peptide molecules with at least one inter-peptide bead contact within 0.6 nm. It is supporting evidence only; the retained Phase 3 label uses `paper_path_APcontact_last10ns`, not this contact fraction
- local finalization also writes an aggregate-structure summary with AP_contact cutoff sensitivity, largest-cluster fraction, cluster count, singleton fraction, and mean contacts per peptide. This prevents saturated AP_contact values from being over-interpreted as a single coherent aggregate.

Excluded from automation:

- `11_Analysis_SASAnFrameDump.sh` because it references a missing downstream script
- anything after `12_AP_calc.sh` because the original `13_Array_APcontact.sh` is missing; the dashboard therefore computes a documented local `AP_contact` replacement during `finalize-md-stage`

## Run Directory Layout

Each run is stored under `active_learning_runs/<run_name>/` and contains:

Round-by-round metrics written during `init-run`, `run-replay`, `propose-round`, and `ingest-round` are validation metrics. The frozen holdout is reserved for the dedicated `evaluate-final` command.

For thesis workflows, threshold-dependent metrics use a validation
\(F_1\)-maximising classification threshold. The threshold is selected only
from the designated validation predictions; ties retain the higher threshold.
It is then frozen and applied unchanged to the corresponding test or holdout
predictions. It is never selected on the frozen holdout and is used only for
threshold-dependent reporting. Fixed `0.5` cutoff metrics remain available with
`_fixed_0_5` suffixes. Generic helper commands may still expose
`pr_best_f1` evaluation-set behaviour for non-thesis diagnostics, but that is
not the thesis holdout protocol.

- `config.json`: run configuration
- `split_manifest.json`: frozen experimental split and replay seed definition
- `ledger.csv`: master peptide ledger
- `snapshots/`: ledger snapshots after important state transitions
- `models/`: ensemble, family, replay, and real-loop model artifacts
- `metrics/`: headline baseline metrics
- `replay/`: learning curves and replay summaries
- `candidates/`: scored candidate pools for each real AL round
- `batches/`: exported CG-MD batches
- `imports/`: imported CG-MD labels
- `discovery/`: per-strategy discovery summaries, candidate rankings, and top batches

## Acquisition Utilities

Milestone 1 includes:

- `random`
- `ensemble_mi`
- `similarity_penalized_mean`
- `family_qbc`
- `cluster_diverse_representative`
- `oed_logdet`
- `hybrid_mi_diverse`

The default cluster-diversity multiplier for new runs is `3`, so cluster-based
diversity methods request at least `batch_size` clusters and typically
`3 * batch_size` clusters, capped by candidate count. Existing stored configs
keep their saved value.

The primary uncertainty signal is a five-member `AP_SP` deep ensemble with
mutual information. The predictive-family committee is kept as a separate
baseline for disagreement sampling.

## Legacy / Generic Discovery Strategies

The `run-discovery` command is legacy/generic export-only discovery support. It
compares these BO-style utilities using an existing trained run and the generic
discovery configuration:

- `ensemble_mean`
- `ucb`
- `ei`
- `pi`
- `mes`

Generic discovery loads the latest available real-AL `post_ingest` ensemble and
falls back to the baseline ensemble if no post-ingest round exists. Its current
defaults include `discovery_ucb_beta = 1.0`,
`discovery_improvement_xi = 0.0`, and `discovery_mes_samples = 128`. These
settings and outputs belong only to generic discovery.

> **Warning:** The generic `run-discovery` command is separate from the thesis
> Phase 4 fixed-surrogate comparison and must not be used to generate Phase 4
> thesis evidence.

## Active-Learning Strategy Audit

- Replay benchmark remains the clean acquisition-function comparison: every strategy selects from the same hidden experimental pool, with no generative candidate-pool effects. Real AL / Propose Next Batch also uses the chosen acquisition strategy, but the available candidate pool can be shaped by generator objective mode and similarity/length penalties, so real AL evidence should be described as acquisition plus generator effects.
- `random` is the negative-control baseline for replay and study comparisons.
- `ensemble_mean` is greedy exploitation, ranking candidates directly by calibrated ensemble mean self-assembly probability.
- `similarity_penalized_mean` is a paper-style practical-discovery baseline, not a clean uncertainty-based AL acquisition function. In Replay it greedily ranks candidates by `pred_mean - similarity_penalty`, where the amino-acid-composition similarity penalty is calculated against the current labeled/training sequences plus candidates already selected earlier in the same batch. Replay does not apply a length penalty. In Real AL, similarity penalty shapes the GA-generated pool and the final batch selection uses unpenalized `pred_mean`.
- `predictive_entropy` ranks candidates by binary predictive entropy, targeting uncertain decision-boundary candidates without requiring family models or embeddings.
- `ensemble_mi` ranks BALD-style ensemble mutual information, so it is a correct epistemic-uncertainty strategy but intentionally not diversity-aware.
- `ucb` ranks candidates by `pred_mean + discovery_ucb_beta * pred_std`, a pointwise upper-confidence-bound acquisition rule.
- `family_qbc` ranks heterogeneous family-committee vote entropy, with probability-standard-deviation tie-breaking, so it is a correct query-by-committee comparator.
- `cluster_diverse_representative` uses KMeans representatives in learned embedding space, but selects clusters by centroid novelty relative to labeled/training embeddings. Cluster novelty selects clusters; distance to centroid selects the representative inside each selected cluster.
- `cluster_representative` and `embedding_farthest_first` remain runnable for legacy configs and ablations, but are not part of the default or thesis-full preset.
- `oed_logdet` uses greedy D-optimal log-det information gain over embedding features, aligned with design-of-experiments sampling.
- `hybrid_mi_diverse` is an explicit split-batch hybrid: the first half of the batch is selected by mutual information, and the second half is selected by farthest-first embedding diversity relative to the labeled set plus the uncertainty-selected peptides. Odd batch sizes give the extra slot to mutual information.
- Generic discovery utilities are BO-style helpers and are not Phase 4 thesis
  policies or evidence. Their legacy MES implementation must not be used to
  describe the Phase 4 MES procedure.

Selected-batch CSVs include reporting-only interpretability columns such as `pointwise_score`, `selection_score`, `similarity_penalty`, `cluster_id`, `distance_to_centroid`, `distance_to_labeled`, `oed_gain`, and `diversity_rank`. These fields do not affect acquisition behavior and do not replace `acquisition_score`. For `similarity_penalized_mean`, `selection_score` is workflow-specific: Replay writes `selection_score = pred_mean - similarity_penalty`, while Real AL final selection writes `selection_score = pred_mean` because similarity penalty already shaped GA generation. For `cluster_diverse_representative`, `acquisition_score` and `selection_score` are the assigned cluster novelty score for every candidate, while `distance_to_centroid` explains the representative choice inside each cluster. Its outputs also report `requested_batch_size`, `candidate_count`, `requested_cluster_count`, `non_empty_cluster_count`, `selected_cluster_count`, and `fallback_fill_count`; `fallback_fill_count = 0` means the strategy fully operated as novel-cluster representative selection, while larger values should be reported as deterministic fill behavior. For `hybrid_mi_diverse`, uncertainty-selected rows use mutual information as `selection_score`, while diversity-selected rows use the dynamic farthest-first distance and receive `diversity_rank`. For `embedding_farthest_first`, `selection_score` is the dynamic greedy farthest-first distance at the moment the candidate was selected, while `distance_to_labeled` remains the distance to the original labeled set only.

## Phase 4 Fixed-Surrogate BO Comparison

Phase 4 is a branch-neutral, single-round proposal comparison. It recreates the
immutable Phase 1 split as 235 model-fitting rows, 59 fixed
validation/calibration rows, and 74 frozen-holdout rows. Holdout labels,
predictions, metrics, and performance information are unavailable to Phase 4;
holdout sequence identities are used only for exact duplicate exclusion.

Five independently seeded `AP_SP` members are fitted on the 235-row training
partition. Each member is independently calibrated on its own predictions for
the fixed 59-row validation/calibration partition using the shared Phase 3
standardised-logit Platt implementation. Calibration clips probabilities to
`[1e-6, 1-1e-6]`, uses learning rate `0.05`, at most `500` iterations, and
`L2=1e-3` towards coefficient `1` and intercept `0`. Valid degenerate cases use
identity fallback, which returns clipped raw probabilities unchanged.

All guided policies share the same immutable checkpoints, calibrators,
calibrated training predictions, and surrogate-space incumbent. The incumbent
is the maximum calibrated ensemble-mean probability among the 235 training
peptides. Ensemble uncertainty is the population standard deviation across the
five calibrated member probabilities (`ddof=0`).

The six proposal policies are:

- deterministic seeded random selection through the inherited Phase 3 random path
- calibrated greedy exploitation
- calibrated UCB with `kappa=1.0`
- probability-space approximate PI with `xi=0.0` and epsilon `1e-8`
- probability-space approximate EI with `xi=0.0` and epsilon `1e-8`
- calibrated ensemble-based approximate MES using five coherent member-function maxima

Phase 4 MES is a finite-pool calibrated neural-ensemble approximation. Each
calibrated ensemble-member prediction vector is treated as one coherent
approximate function sample. Temporary member-wise maxima may be calculated
over the current GA scoring population during evolution, while final ranking
uses member-wise maxima recomputed over the complete retained pool. It is not
exact Gaussian-process MES.

Within candidate generation, Phase 4 changes only the policy utility callback.
It directly reuses the Phase 3 genetic algorithm, population-composition
penalty, piecewise preferred-length penalty, retry and seed behaviour,
deduplication, pool construction, stable guided selector, and seeded random
selector. GA evolution uses penalised fitness, while guided final selection
uses the unpenalised utility recomputed over the complete retained pool. MES
final ranking uses one common retained-pool set of five member maxima.

Initialization writes manifests, schemas, implementation audits, PBS previews,
and the dependency-aware submission script under
`thesis_results/04_bayesian_optimization/`. It does not train locally or submit
jobs. In the generated SUPEK DAG, random is independent, the five guided jobs
depend `afterok` on the shared training job, and comparison/status depends
`afterany` on all six proposal jobs.

### Phase 4-D secondary diversity-aware generative replicate

Phase 4-D is an executed and aggregated exploratory operational analysis and
does not replace the primary Phase 4 acquisition-policy comparison. It reuses the completed frozen
Phase 4 checkpoints, member-wise calibrators, calibrated training predictions,
surrogate-space incumbent, acquisition constants, preprocessing, and exact
proposal exclusions. It does not retrain or recalibrate, use the
validation-selected classification threshold, score the frozen holdout, ingest
labels, or launch simulations.

Candidate generation is repeated through the unchanged shared Phase 3/Phase 4
genetic-algorithm path using new non-overlapping policy seed blocks. For each
guided policy, final unpenalised utilities are frozen over the complete newly
generated retained pool. A utility-only control selects the stable top five,
while the diversity-aware selector subtracts the inherited amino-acid-
composition similarity averaged over peptides already selected into the
current batch. Both batches therefore use the same retained pool and model
utilities; their difference isolates final diversity-aware selection. Random
uses only the inherited seeded shuffle.

Phase 4-D intentionally trades acquisition utility against the inherited
composition-similarity penalty and is not a pure acquisition-utility ranking.
Comparisons with primary Phase 4 are descriptive because the generation seeds
and retained pools differ. All simulation-slate decisions remain manual.

## Official Phase 2 Replay Configuration

The main thesis replay benchmark contains exactly ten strategies:

1. random
2. ensemble mean
3. similarity-penalised mean
4. predictive entropy
5. ensemble mutual information
6. UCB
7. family QBC
8. cluster-diverse representative selection
9. OED logdet
10. hybrid MI-diverse

PI, EI, and MES are generic discovery utilities and are not main Phase 2 replay
strategies. The official replay evaluates the \(n_0=10\) and \(n_0=40\)
starting-label regimes separately, uses batch size \(B=5\), and runs for at
most 20 acquisition rounds. Labels-to-target are reported for
\(F_1=0.80,0.84,0.86\), with \(0.84\) as the principal practical target.
Targets are detected only at observed replay points; no interpolation is used.
Unreached targets remain `not reached`, and labels-to-target summaries must
include reach counts or reach fractions alongside successful-fold values.

## Operational Phase 3 CG-MD Contact Metrics

The exact full-profile BURA simulation settings used to generate these metrics
are recorded in `THESIS_CGMD_PARAMETER_CONTRACT.md`. That source-checked table
covers the GROMACS/Martini versions, minimisation and equilibration stages,
coupling constants, nonbonded settings, active topology constraints, output
intervals, charged termini, and the all-extended secondary-structure input.

The primary contact metric used by the retained Phase 3 labels is
`paper_path_APcontact_last10ns`. It applies the paper piecewise distance weight
to minimum inter-bead distances, maximises the mean consecutive-edge weight
over Hamiltonian paths, uses exact dynamic programming for at most 16 peptide
copies and deterministic beam search otherwise, and averages the 11 frames
from 190 through 200 ns.

The retained operational rule is:

```text
cgmd_label = 1 only when
AP_sasa(200 ns) >= 1.75
and paper_path_APcontact_last10ns >= 0.5
```

`AP_contact`, the fraction of peptide molecules with at least one inter-peptide
bead contact within 0.6 nm, remains a separate diagnostic together with cluster
summaries. It does not define the primary Phase 3 label.

## Phase 5: SPAL-Inspired Self-Paced Replay

Phase 5 is executed and aggregated. It is an isolated retrospective extension
of the official Phase 2 replay and does not mutate Phase 3 ledgers, consume
Phase 4 proposals, or submit MD simulations.

The four primary strategies are `random`, `predictive_entropy`,
`static_easy_entropy`, and `self_paced_entropy`. The self-paced policies use an
operational calibrated neural-model labelled-manifold familiarity proxy. The
primary experiment trains exactly one calibrated `AP_SP` model at each replay
point. Its 384-dimensional penultimate embedding is normalised and the nearest
currently labelled replay-training embedding distance is calculated.
Validation and holdout rows are excluded from this reference set.

The reduced primary design uses outer folds 1--3, one initial condition
(`n0=10`), batches of five, and 45 acquisitions so every strategy progresses
through the same labelled counts from 10 to all 235 replay-training rows.
Before each fit, revealed training rows are canonically ordered by immutable
dataset row identifier. The terminal 235-row fits therefore share sequence
identities, row order, architecture, validation set, training procedure, and
model seed across strategies.

The principal trajectory summary is normalised AULC-F1 over 10--235 labels.
Preregistered descriptive partial AULCs cover 10--60, 10--110, and 10--160.
Positive yield, selected-sequence diversity, and strategy overlap are reported
per batch, cumulatively at matched labelled counts, and at the fixed budgets
60, 110, and 160. Terminal cumulative values are consistency checks only.

The fold-matched Phase 1 `AP_SP` result is contextual: it uses all 294
non-holdout development rows, whereas a Phase 5 terminal model fits 235 rows
and reserves 59 for validation and calibration. Phase 1 does not enter Phase 5
rankings, AULCs, or hypothesis tests.

Completed mean full AULC-F1 was 0.8223 for predictive entropy, 0.8171 for
random, 0.8147 for self-paced entropy, and 0.8016 for static easy entropy.
Self-pacing therefore improved on the fixed easy-only restriction but did not
outperform ordinary predictive entropy or random selection overall. See
`PHASE5_RESULTS_SUMMARY.md` and
`thesis_results/05_self_paced_active_learning/tables/`.
