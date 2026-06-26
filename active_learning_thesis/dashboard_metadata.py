from __future__ import annotations

VIEW_NAMES = ["Today", "Model Workflow", "Results", "Peptides", "MD Validation", "Operations"]
WORKSPACE_SCOPES = ["Current Thesis Work", "All Runs", "Historical / Test"]
BEGINNER_WORKFLOW_GUIDE = [
    {
        "step": "1. Create or pick the real thesis run",
        "goal": "Start from a clean config and keep smoke tests from distracting you.",
        "where": "Operations -> New run wizard or Operations -> Run curation",
        "main_control": "Create a run from a preset, clone an existing run, or pin the runs you actually want.",
        "done_when": "Today shows the run you mean to work on.",
    },
    {
        "step": "2. Benchmark the model",
        "goal": "Create the clean initial-dataset baseline.",
        "where": "Model Workflow -> Guided workflow runner",
        "main_control": "Advance plan: Replay benchmark.",
        "done_when": "Replay evidence appears in Results.",
    },
    {
        "step": "3. Propose candidates",
        "goal": "Generate peptides for validation.",
        "where": "Model Workflow -> Guided workflow runner",
        "main_control": "Advance plan: Propose next batch.",
        "done_when": "Suggested peptides appear on Today and Peptides.",
    },
    {
        "step": "4. Choose what goes to MD",
        "goal": "Make candidate selection explicit and traceable.",
        "where": "Peptides -> Candidate selection",
        "main_control": "Select, defer, reject, or create dashboard-local MD batch rows.",
        "done_when": "Chosen peptides are launch-ready for MD Validation.",
    },
    {
        "step": "5. Run guided MD",
        "goal": "Move one peptide through the safe ladder.",
        "where": "MD Validation -> Guided ladder",
        "main_control": "Prepare -> upload -> submit -> monitor -> finalize.",
        "done_when": "The full analysis row is ready for review.",
    },
    {
        "step": "6. Review the label",
        "goal": "Turn MD evidence into a thesis-defensible human label.",
        "where": "MD Validation -> Review & model feedback",
        "main_control": "Save the evidence-backed cgmd_label and create cgmd_ingest.csv.",
        "done_when": "The peptide is Ready for ingest.",
    },
    {
        "step": "7. Feed labels back",
        "goal": "Close the active-learning loop.",
        "where": "Model Workflow -> Local model actions",
        "main_control": "Continue AL from reviewed peptides or Ingest returned labels.",
        "done_when": "The model is retrained and ready for another proposal, discovery, or final evaluation.",
    },
    {
        "step": "8. Freeze and report",
        "goal": "Create final, thesis-safe outputs without touching the CLI.",
        "where": "Model Workflow -> Thesis freeze, then Results -> Thesis output builder",
        "main_control": "Freeze final thesis result, export thesis packet, build thesis figures.",
        "done_when": "The final freeze, packet, figures, tables, and captions exist under the run root.",
    },
]
GUI_COVERAGE_GUIDE = [
    {
        "task": "Guided thesis phase checklist",
        "gui_status": "GUI-native",
        "where": "Today and Operations -> Thesis checklist",
        "notes": "Audits Setup -> Run -> Study -> MD -> Ingest -> Freeze -> Export and recommends the next safe click.",
    },
    {
        "task": "Create a new run",
        "gui_status": "GUI-native",
        "where": "Operations -> New run wizard",
        "notes": "Create from beginner presets, clone existing settings, auto-pin, label, and optionally run replay.",
    },
    {
        "task": "Run-level AL workflow",
        "gui_status": "GUI-native",
        "where": "Model Workflow",
        "notes": "Replay, propose, discovery, final evaluation, ingest, and feedback continuation have guided buttons.",
    },
    {
        "task": "Peptide candidate decisions",
        "gui_status": "GUI-native",
        "where": "Peptides",
        "notes": "Candidate selection, provenance, promotion, and bulk handoff are dashboard-first.",
    },
    {
        "task": "MD validation",
        "gui_status": "GUI-native",
        "where": "MD Validation",
        "notes": "The guided ladder covers local preparation, BURA actions, monitoring, finalization, review, and ingest CSV creation.",
    },
    {
        "task": "Remote monitoring",
        "gui_status": "GUI-native",
        "where": "Operations",
        "notes": "Health, approvals, action debugging, queues, watchdog drift, reconciliation, transfers, and logs are available without shell commands.",
    },
    {
        "task": "Final thesis freeze",
        "gui_status": "GUI-native",
        "where": "Model Workflow -> Thesis freeze",
        "notes": "Creates the freeze manifest, checks, and model card from the selected run.",
    },
    {
        "task": "Thesis packet and figures",
        "gui_status": "GUI-native",
        "where": "Results -> Thesis output builder",
        "notes": "Exports the packet, builds figure/table/caption bundles, and can run the seeded thesis canary.",
    },
    {
        "task": "Advanced study sweeps",
        "gui_status": "GUI-native",
        "where": "Operations -> Study designer and Results -> Study comparison hub",
        "notes": "Create dry-run plans, queue multi-seed studies, summarize replay evidence, compare matched studies, and inspect thesis-ready tables.",
    },
]
PAGE_GUIDES = {
    "Today": {
        "purpose": "Use this page as the cockpit home screen for deciding what to do next across runs, peptides, cluster readiness, and notifications.",
        "contains": "Guided thesis checklist, ranked next actions, remembered notifications, workspace summary, and the highest-priority blockers.",
        "best_for": "Choosing the next thesis step without opening every page manually.",
    },
    "Model Workflow": {
        "purpose": "Use this page when you are acting on one thesis run: replay, propose, discovery, ingest, final evaluation, or SUPEK execution.",
        "contains": "Run guidance, checkpoint runner, local model actions, remote SUPEK controls, and run-level decisions.",
        "best_for": "Moving one active-learning run safely from its current checkpoint to the next one.",
    },
    "Results": {
        "purpose": "Use this page when you want reporting evidence rather than operational controls.",
        "contains": "Thesis packet, scorecards, comparisons, milestones, figure-ready summaries, and audit trails.",
        "best_for": "Preparing tables, figures, captions, and thesis narrative support.",
    },
    "Peptides": {
        "purpose": "Use this page when the question is about peptide state rather than run state.",
        "contains": "Lifecycle ledger, candidate selection, bulk review / ingest, and peptide-level backlog buckets.",
        "best_for": "Choosing what goes into MD next and tracking which peptides are ready to feed back into the model.",
    },
    "MD Validation": {
        "purpose": "Use this page for the guided MD ladder on one peptide at a time.",
        "contains": "Ladder checkpoints, review workspace, local MD actions, remote BURA controls, and peptide-level decisions.",
        "best_for": "Advancing or reviewing a specific peptide through the MD validation pipeline.",
    },
    "Operations": {
        "purpose": "Use this page for global operator controls rather than thesis content.",
        "contains": "Cluster health, notifications, thesis checklist, action debugging, approval queue, remote job consoles, transfer manifest, and run curation.",
        "best_for": "Monitoring infrastructure, debugging failed actions, approving drafts, and checking the global state of the cockpit.",
    },
}
HISTORICAL_RUN_MARKERS = (
    "smoke",
    "bugcheck",
    "bug_check",
    "tune",
    "tuning",
    "validation",
    "regression",
    "perf",
    "benchmark",
    "gpu",
    "cpu",
    "e2e",
)
MD_PROFILE_INFO = {
    "line_smoke": {
        "label": "Quick package check (line_smoke)",
        "short_label": "Quick package check",
        "description": "Build and sanity-check one short MD package. This is the cheapest first gate and it is not ingest-ready.",
        "produces": "A staged package plus a short dynamics result proving the setup can run.",
        "terminal_status": "dynamics_complete",
    },
    "production_smoke": {
        "label": "Short dynamics validation (production_smoke)",
        "short_label": "Short dynamics validation",
        "description": "Run a longer smoke validation after the package check succeeds. This still stops before full ingest-ready analysis.",
        "produces": "A short production-style dynamics result that validates the peptide can survive the next rung.",
        "terminal_status": "dynamics_complete",
    },
    "full": {
        "label": "Full analysis run (full)",
        "short_label": "Full analysis run",
        "description": "Run the full guided MD analysis chain. This is the only ladder stage that can become ready for review and ingest.",
        "produces": "Full analysis outputs, reviewable metrics, and eventually an ingest-ready label path.",
        "terminal_status": "analysis_complete",
    },
}
MD_STATUS_INFO = {
    "not_started": "Not started yet",
    "pdb_missing": "PDB still missing",
    "package_prepared": "Prepared locally, not sent yet",
    "dynamics_complete": "Dynamics finished",
    "sasa_complete": "Partial analysis finished, not ingest-ready",
    "analysis_complete": "Analysis finished, ready for review",
}
ML_STATUS_INFO = {
    "config-only": {
        "label": "Config only",
        "summary": "This run only has configuration scaffolding so far.",
    },
    "initialized": {
        "label": "Initialized",
        "summary": "Baseline metrics exist, so you can benchmark the initial-dataset workflow.",
    },
    "replay-complete": {
        "label": "Replay benchmark finished",
        "summary": "Initial acquisition benchmarking exists. The next step is usually proposing a real peptide batch or running discovery.",
    },
    "batch-proposed": {
        "label": "Batch proposed",
        "summary": "The model has already suggested peptides. Those suggestions now need MD validation and feedback before more proposing.",
    },
    "discovery-complete": {
        "label": "Discovery finished",
        "summary": "Discovery summaries exist. Review the candidates and decide whether to validate them or lock in a final evaluation.",
    },
    "final-evaluated": {
        "label": "Final holdout evaluated",
        "summary": "This run already has a frozen holdout evaluation.",
    },
}
RUN_ACTION_INFO = {
    "continue-feedback": {
        "label": "Continue AL from reviewed peptides",
        "what": "Validate the whole pending proposed batch, create or refresh the per-peptide ingest rows, aggregate them into one run-level import CSV, and retrain the model on the newest MD labels.",
        "when": "Use this when every peptide in the currently pending proposed batch has finished full analysis and has a final human `cgmd_label`.",
        "produces": "A run-level import CSV, an `ingest-round` retrain/update, and optionally the next proposed batch if you choose the extended runner.",
        "next": "The refreshed run will be ready to propose another batch, run discovery, or freeze a final evaluation from the updated model state.",
    },
    "run-replay": {
        "label": "Replay benchmark (run-replay)",
        "what": "Benchmark acquisition strategies on the initial labeled dataset only, without proposing new peptides.",
        "when": "Use this after a run is initialized and before the first real peptide batch.",
        "produces": "Replay summaries and strategy comparison curves.",
        "next": "You can propose the first real validation batch or inspect which strategy looks strongest.",
    },
    "propose-round": {
        "label": "Propose next batch (propose-round)",
        "what": "Generate the next peptide batch for wet-lab / MD validation.",
        "when": "Use this when the current run is ready to suggest new peptides.",
        "produces": "A new `round_XXX_batch.csv` plus scored candidates for validation.",
        "next": "The proposed peptides can be sent into the guided MD workflow.",
    },
    "run-discovery": {
        "label": "Run discovery (run-discovery)",
        "what": "Search for novel promising peptides beyond the next standard acquisition batch.",
        "when": "Use this when you want exploratory candidates for thesis discussion or follow-up validation.",
        "produces": "Discovery summaries and ranked candidate sets.",
        "next": "Review the discovery candidates and decide which ones to validate.",
    },
    "evaluate-final": {
        "label": "Final evaluation (evaluate-final)",
        "what": "Evaluate the latest trained ensemble once on the frozen holdout.",
        "when": "Use this only when you are ready to freeze the model state for thesis reporting.",
        "produces": "Final holdout metrics in `metrics/final_holdout.json`.",
        "next": "Use the reported metrics directly in thesis results and comparisons.",
    },
    "freeze-final": {
        "label": "Freeze final thesis result (freeze-final)",
        "what": "Create the final reproducibility freeze for one run: consistency checks, file fingerprints, model manifest, and a thesis model card.",
        "when": "Use this after final holdout metrics exist, or enable the option to run final evaluation first when you are ready to move into reporting mode.",
        "produces": "A `final_freeze/` bundle with `final_freeze.json`, check tables, model manifest, and `model_card.md`.",
        "next": "Export a thesis packet from Results, then build the figure/table/caption bundle from that packet.",
    },
    "ingest-round": {
        "label": "Ingest returned labels (ingest-round)",
        "what": "Feed reviewed MD labels back into the active-learning run and retrain the model.",
        "when": "Use this after a `cgmd_ingest.csv` has been created from reviewed full-analysis outputs.",
        "produces": "Updated model weights, metrics, and the next active-learning-ready run state.",
        "next": "You can propose another batch, run discovery, or do a final evaluation from the updated model.",
    },
}
DECISION_TYPE_INFO = {
    "choose_primary_run": {
        "label": "Choose as primary thesis run",
        "scope": "run",
        "default_title": "Use this run as a primary thesis comparison",
        "default_next_step": "Keep this run pinned and include it in selected-run thesis comparisons.",
    },
    "freeze_final": {
        "label": "Freeze a final evaluation",
        "scope": "run",
        "default_title": "Freeze this run for final thesis reporting",
        "default_next_step": "Run or preserve the frozen final evaluation and use Results for reporting.",
    },
    "validate_discovery": {
        "label": "Validate discovery shortlist",
        "scope": "run",
        "default_title": "Move discovery candidates into MD validation",
        "default_next_step": "Open MD Validation for the chosen peptides and advance the ladder.",
    },
    "defer_discovery": {
        "label": "Defer discovery for now",
        "scope": "run",
        "default_title": "Keep discovery as supporting evidence only",
        "default_next_step": "Use Results and Thesis narrative to report the shortlist without validating it yet.",
    },
    "strategy_note": {
        "label": "Record model / strategy note",
        "scope": "run",
        "default_title": "Record a thesis note for this run",
        "default_next_step": "Use this note in Results, milestones, or the thesis narrative later.",
    },
    "accept_md_label": {
        "label": "Accept the current MD label",
        "scope": "peptide",
        "default_title": "Accept the reviewed MD label for this peptide",
        "default_next_step": "Create `cgmd_ingest.csv`, then ingest the label back into the model.",
    },
    "reject_md_label": {
        "label": "Reject the peptide after review",
        "scope": "peptide",
        "default_title": "Reject this peptide after MD review",
        "default_next_step": "Keep the review record, then decide whether to ingest the negative label or stop here.",
    },
    "override_md_label": {
        "label": "Override the current MD label",
        "scope": "peptide",
        "default_title": "Override the current MD label for this peptide",
        "default_next_step": "Save the revised review, then create the ingest CSV only after the label is final.",
    },
    "send_to_ingest": {
        "label": "Send this peptide to model ingest",
        "scope": "peptide",
        "default_title": "This peptide is ready to feed back into the model",
        "default_next_step": "Create `cgmd_ingest.csv`, then run Ingest returned labels in Model Workflow.",
    },
    "hold_for_review": {
        "label": "Hold for later review",
        "scope": "peptide",
        "default_title": "Hold this peptide before model ingest",
        "default_next_step": "Keep the peptide in the review queue until the evidence or label is final.",
    },
    "select_candidate_for_md": {
        "label": "Select this candidate for MD validation",
        "scope": "candidate",
        "default_title": "Validate this candidate in MD",
        "default_next_step": "Open MD Validation and prepare the first safe ladder step for this peptide.",
    },
    "defer_candidate": {
        "label": "Defer this candidate for now",
        "scope": "candidate",
        "default_title": "Defer this candidate until later",
        "default_next_step": "Leave the peptide out of the next MD batch and revisit it after stronger evidence or more urgent candidates.",
    },
    "reject_candidate": {
        "label": "Do not validate this candidate",
        "scope": "candidate",
        "default_title": "Reject this candidate from MD validation",
        "default_next_step": "Keep the decision in the thesis log so the peptide does not re-enter the shortlist by accident.",
    },
    "candidate_note": {
        "label": "Record candidate-selection note",
        "scope": "candidate",
        "default_title": "Record a thesis note for this candidate",
        "default_next_step": "Use the note later when comparing which candidates were advanced, deferred, or rejected.",
    },
}
RUN_DECISION_TYPES = ["choose_primary_run", "freeze_final", "validate_discovery", "defer_discovery", "strategy_note"]
PEPTIDE_DECISION_TYPES = ["accept_md_label", "reject_md_label", "override_md_label", "send_to_ingest", "hold_for_review"]
CANDIDATE_DECISION_TYPES = ["select_candidate_for_md", "defer_candidate", "reject_candidate", "candidate_note"]
REMOTE_SYNC_INFO = {
    "not_synced": "Not staged remotely",
    "staged_remote": "Staged on the remote cluster",
    "submitted": "Submitted to the remote queue",
    "running": "Running remotely",
    "outputs_staged": "Downloaded into dashboard staging",
    "outputs_returned": "Copied back into the campaign folder",
    "finalized_local": "Parsed locally and ready for the next decision",
    "stale": "Marked stale locally",
}
PEPTIDE_BUCKET_ORDER = [
    "Ready for ingest",
    "Reviewed for reporting",
    "Needs review / label",
    "MD in progress",
    "Sent for MD",
    "Suggested by model",
    "Already ingested",
]
