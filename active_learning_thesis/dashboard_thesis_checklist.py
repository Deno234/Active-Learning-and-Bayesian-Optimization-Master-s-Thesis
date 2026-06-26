from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from active_learning_thesis.dashboard_action_debugger import build_action_debug_rows


THESIS_PHASES = ("Setup", "Run", "Study", "MD", "Ingest", "Freeze", "Export")

STATUS_DONE = "Done"
STATUS_IN_PROGRESS = "In progress"
STATUS_READY = "Ready"
STATUS_BLOCKED = "Blocked"
STATUS_NOT_STARTED = "Not started"

_ACTIVE_ACTION_STATUSES = {"queued", "running", "draft", "awaiting_approval", "paused"}
_ATTENTION_STATUSES = {"Needs fix", "Possibly stuck", "Waiting approval", "Waiting operator"}

_PHASE_ACTION_KINDS: dict[str, set[str]] = {
    "Setup": {"init-run"},
    "Run": {"run-replay", "propose-round", "run-discovery", "evaluate-final", "run-workflow", "supek-submit", "supek-sync"},
    "Study": {"run-study", "summarize-study", "compare-studies"},
    "MD": {
        "prepare-md-stage",
        "upload-md-campaign",
        "submit-md-chain",
        "poll-bura-queue",
        "fetch-md-logs",
        "finalize-md-stage",
        "recover-md-stage",
        "launch-md-slate",
        "advance-md-slate",
        "update-md-review",
        "bulk-update-md-review",
    },
    "Ingest": {"make-md-ingest-csv", "ingest-round", "continue-feedback", "promote-reporting-md-campaign"},
    "Freeze": {"freeze-final", "evaluate-final"},
    "Export": {"export-thesis-packet", "build-thesis-figures", "thesis-canary"},
}

_PHASE_COPY = {
    "Setup": {
        "why_it_matters": "A clean run root keeps smoke tests and thesis evidence separated.",
        "next_click": "Operations -> New run wizard",
        "open_view": "Operations",
    },
    "Run": {
        "why_it_matters": "Replay, proposal, discovery, and final evaluation produce the core active-learning evidence.",
        "next_click": "Model Workflow -> Guided workflow runner",
        "open_view": "Model Workflow",
    },
    "Study": {
        "why_it_matters": "Seeded studies make the thesis claim stronger than a single lucky run.",
        "next_click": "Operations -> Study designer",
        "open_view": "Operations",
    },
    "MD": {
        "why_it_matters": "MD validation turns model suggestions into physically inspected peptide evidence.",
        "next_click": "Peptides -> Candidate selection, then MD Validation",
        "open_view": "Peptides",
    },
    "Ingest": {
        "why_it_matters": "Reviewed `cgmd_label` rows only affect learning after they are ingested back into the run.",
        "next_click": "Model Workflow -> Local model actions",
        "open_view": "Model Workflow",
    },
    "Freeze": {
        "why_it_matters": "The freeze locks a report-safe result with reproducibility checks and a model card.",
        "next_click": "Model Workflow -> Thesis freeze",
        "open_view": "Model Workflow",
    },
    "Export": {
        "why_it_matters": "Packets and figures turn dashboard evidence into thesis-ready tables, captions, and appendix material.",
        "next_click": "Results -> Thesis output builder",
        "open_view": "Results",
    },
}


def _safe_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Path(text)
    except OSError:
        return None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_runs(state: dict[str, object]) -> list[dict[str, object]]:
    runs = state.get("runs", [])
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _all_state_runs(state: dict[str, object]) -> list[dict[str, object]]:
    runs = state.get("all_runs", state.get("runs", []))
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _run_dirs(state: dict[str, object]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for run in _all_state_runs(state):
        path = _safe_path(run.get("run_dir"))
        if path is None:
            continue
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def _glob_count(paths: Iterable[Path], pattern: str) -> int:
    total = 0
    for path in paths:
        try:
            total += sum(1 for item in path.glob(pattern) if item.exists())
        except OSError:
            continue
    return total


def _rglob_count(paths: Iterable[Path], pattern: str) -> int:
    total = 0
    for path in paths:
        try:
            total += sum(1 for item in path.rglob(pattern) if item.exists())
        except OSError:
            continue
    return total


def _csv_has_final_label(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("cgmd_label", "")).strip() in {"0", "1"}:
                    return True
    except (OSError, csv.Error):
        return False
    return False


def _json_status_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and str(payload.get("status", "")).strip() == "completed"


def _count_labeled_reviews(run_dirs: list[Path]) -> int:
    total = 0
    for run_dir in run_dirs:
        try:
            review_paths = list((run_dir / "md_campaigns").glob("*/md_review.csv"))
        except OSError:
            continue
        total += sum(1 for path in review_paths if _csv_has_final_label(path))
    return total


def _peptide_count(state: dict[str, object], bucket: str) -> int:
    inventory = state.get("peptide_inventory", {})
    counts = inventory.get("counts", {}) if isinstance(inventory, dict) and isinstance(inventory.get("counts", {}), dict) else {}
    return _safe_int(counts.get(bucket, 0))


def _active_action_count(actions: list[dict[str, object]], phase: str) -> int:
    kinds = _PHASE_ACTION_KINDS.get(phase, set())
    return sum(
        1
        for action in actions
        if str(action.get("kind", "")).strip() in kinds
        and str(action.get("status", "")).strip() in _ACTIVE_ACTION_STATUSES
    )


def _phase_attention(action_debug_rows: list[dict[str, object]], phase: str) -> dict[str, object] | None:
    kinds = _PHASE_ACTION_KINDS.get(phase, set())
    for row in action_debug_rows:
        if str(row.get("kind", "")).strip() not in kinds:
            continue
        if str(row.get("attention", "")).strip() in _ATTENTION_STATUSES:
            return row
    return None


def _phase_row(
    phase: str,
    *,
    status: str,
    progress: int,
    evidence: str,
    missing: str,
    safe_next_move: str,
    blocker: str = "",
    next_click: str = "",
    open_view: str = "",
) -> dict[str, object]:
    copy = _PHASE_COPY[phase]
    return {
        "phase": phase,
        "status": status,
        "progress": max(0, min(100, int(progress))),
        "evidence": evidence or "-",
        "missing": missing or "-",
        "safe_next_move": safe_next_move or "-",
        "blocker": blocker or "-",
        "next_click": next_click or copy["next_click"],
        "open_view": open_view or copy["open_view"],
        "why_it_matters": copy["why_it_matters"],
    }


def _blocked_row(phase: str, issue: dict[str, object], *, progress: int, evidence: str, missing: str) -> dict[str, object]:
    return _phase_row(
        phase,
        status=STATUS_BLOCKED,
        progress=progress,
        evidence=evidence,
        missing=missing,
        blocker=f"{issue.get('attention', '')}: {issue.get('title', '')}".strip(": "),
        safe_next_move=str(issue.get("safe_next_move", "")),
    )


def _signals(state: dict[str, object]) -> dict[str, object]:
    run_root = _safe_path(state.get("run_root"))
    runs = _state_runs(state)
    all_runs = _all_state_runs(state)
    run_dirs = _run_dirs(state)
    raw_actions = state.get("actions", [])
    actions = [action for action in raw_actions if isinstance(action, dict)] if isinstance(raw_actions, list) else []
    action_debug_rows = build_action_debug_rows(actions)

    replay_runs = sum(1 for run in all_runs if _safe_int(_dict_value(run.get("replay")).get("count", 0)) > 0)
    proposed_runs = sum(1 for run in all_runs if bool(run.get("latest_batch")) or str(run.get("ml_status", "")) in {"batch-proposed", "discovery-complete", "final-evaluated"})
    discovery_runs = sum(1 for run in all_runs if _safe_int(_dict_value(run.get("discovery")).get("count", 0)) > 0 or str(run.get("ml_status", "")) in {"discovery-complete", "final-evaluated"})
    final_metric_runs = sum(1 for run in all_runs if isinstance(run.get("final_metrics"), dict) and bool(run.get("final_metrics")))
    import_runs = sum(1 for run in all_runs if bool(run.get("import_rows")) or bool(run.get("latest_import")))
    baseline_runs = sum(1 for run in all_runs if isinstance(run.get("baseline_metrics"), dict) and bool(run.get("baseline_metrics")))

    md_campaigns = sum(len(list(run.get("md_campaigns", []))) for run in all_runs if isinstance(run.get("md_campaigns", []), list))
    analysis_ready = sum(
        1
        for run in all_runs
        for campaign in list(run.get("md_campaigns", []))
        if isinstance(campaign, dict)
        and (
            str(campaign.get("job_root_status", "")).strip() == "analysis_complete"
            or str(campaign.get("cgmd_label", "")).strip() in {"0", "1"}
        )
    )
    ingest_csvs = sum(len(list(run.get("available_ingest_csvs", []))) for run in all_runs if isinstance(run.get("available_ingest_csvs", []), list))

    if run_dirs:
        replay_runs = max(replay_runs, _glob_count(run_dirs, "replay/*/summary.json"))
        proposed_runs = max(proposed_runs, _glob_count(run_dirs, "batches/*_batch.csv"))
        discovery_runs = max(discovery_runs, _glob_count(run_dirs, "discovery/aggregate_summary.csv"))
        final_metric_runs = max(final_metric_runs, _glob_count(run_dirs, "metrics/final_holdout.json"))
        import_runs = max(
            import_runs,
            _glob_count(run_dirs, "imports/*_labels.csv") + _glob_count(run_dirs, "imports/*_import.csv"),
        )
        baseline_runs = max(baseline_runs, _glob_count(run_dirs, "metrics/baseline_round_000.json"))
        md_campaigns = max(md_campaigns, _rglob_count(run_dirs, "md_campaigns/*/manifest.csv"))
        ingest_csvs = max(ingest_csvs, _rglob_count(run_dirs, "md_campaigns/*/cgmd_ingest.csv"))
        labeled_reviews = _count_labeled_reviews(run_dirs)
    else:
        labeled_reviews = 0

    study_manifests = 0
    completed_studies = 0
    study_summaries = 0
    study_comparisons = 0
    packet_manifests = 0
    figure_manifests = 0
    if run_root is not None:
        try:
            study_manifests = sum(1 for path in (run_root / "_studies").glob("*/study_manifest.json") if path.is_file())
            completed_studies = sum(
                1
                for path in (run_root / "_studies").glob("*/study_manifest.json")
                if _json_status_completed(path)
            )
            study_summaries = sum(1 for path in (run_root / "_study_evidence").glob("*_study_summary.json") if path.is_file())
            study_summaries += sum(1 for path in (run_root / "_studies").glob("*/evidence/*_study_summary.json") if path.is_file())
            study_comparisons = sum(1 for path in (run_root / "_studies").glob("_comparisons/**/*_study_comparison_summary.json") if path.is_file())
            packet_manifests = sum(1 for path in (run_root / "_thesis_packets").glob("*/packet_manifest.json") if path.is_file())
            figure_manifests = sum(1 for path in (run_root / "_thesis_packets").glob("*/thesis_figures/thesis_figures_manifest.json") if path.is_file())
            figure_manifests += sum(1 for path in (run_root / "_thesis_packets").glob("*/thesis_figures_manifest.json") if path.is_file())
        except OSError:
            pass

    freeze_count = _glob_count(run_dirs, "final_freeze/final_freeze.json")
    raw_md_slates = state.get("md_slates", [])
    md_slates = [slate for slate in raw_md_slates if isinstance(slate, dict)] if isinstance(raw_md_slates, list) else []
    active_slates = sum(
        1
        for slate in md_slates
        if str(slate.get("effective_status", "")).strip() not in {"completed", "completed_with_failures", "cancelled"}
    )

    return {
        "run_count": len(runs),
        "all_run_count": len(all_runs),
        "baseline_runs": baseline_runs,
        "replay_runs": replay_runs,
        "proposed_runs": proposed_runs,
        "discovery_runs": discovery_runs,
        "final_metric_runs": final_metric_runs,
        "import_runs": import_runs,
        "study_manifests": study_manifests,
        "completed_studies": completed_studies,
        "study_summaries": study_summaries,
        "study_comparisons": study_comparisons,
        "suggested_peptides": _peptide_count(state, "Suggested by model"),
        "selected_or_sent_peptides": _peptide_count(state, "Sent for MD") + _peptide_count(state, "MD in progress"),
        "ready_for_ingest": _peptide_count(state, "Ready for ingest"),
        "already_ingested": _peptide_count(state, "Already ingested"),
        "md_campaigns": md_campaigns,
        "active_slates": active_slates,
        "analysis_ready": analysis_ready,
        "labeled_reviews": labeled_reviews,
        "ingest_csvs": ingest_csvs,
        "freeze_count": freeze_count,
        "packet_manifests": packet_manifests,
        "figure_manifests": figure_manifests,
        "actions": actions,
        "action_debug_rows": action_debug_rows,
    }


def build_thesis_phase_rows(state: dict[str, object]) -> list[dict[str, object]]:
    """Return live, read-only thesis phase guidance for the dashboard."""

    signals = _signals(state)
    actions = list(signals["actions"])
    action_debug_rows = list(signals["action_debug_rows"])

    rows: list[dict[str, object]] = []

    setup_evidence = f"{signals['run_count']} visible run(s), {signals['all_run_count']} total run(s)"
    setup_issue = _phase_attention(action_debug_rows, "Setup")
    if signals["all_run_count"]:
        setup = _phase_row(
            "Setup",
            status=STATUS_DONE,
            progress=100,
            evidence=setup_evidence,
            missing="Nothing required for setup.",
            safe_next_move="Open Model Workflow for the selected run or pin the real thesis run if the visible set looks noisy.",
        )
    elif setup_issue:
        setup = _blocked_row("Setup", setup_issue, progress=15, evidence=setup_evidence, missing="No usable thesis run is visible yet.")
    elif _active_action_count(actions, "Setup"):
        setup = _phase_row(
            "Setup",
            status=STATUS_IN_PROGRESS,
            progress=35,
            evidence="A run setup action is in the dashboard queue.",
            missing="Wait for the run folder and baseline artifacts to appear.",
            safe_next_move="Watch Operations -> Action debugger or the action history until the setup finishes.",
        )
    else:
        setup = _phase_row(
            "Setup",
            status=STATUS_READY,
            progress=0,
            evidence=setup_evidence,
            missing="Create or pin the real thesis run.",
            safe_next_move="Create a run from a thesis preset, or pin the run that should count as Current Thesis Work.",
        )
    rows.append(setup)

    run_done = bool(signals["replay_runs"] or signals["proposed_runs"] or signals["discovery_runs"] or signals["final_metric_runs"])
    run_evidence_parts = [
        f"{signals['baseline_runs']} baseline",
        f"{signals['replay_runs']} replay",
        f"{signals['proposed_runs']} proposed-batch",
        f"{signals['discovery_runs']} discovery",
        f"{signals['final_metric_runs']} final-eval",
    ]
    run_evidence = ", ".join(run_evidence_parts)
    run_issue = _phase_attention(action_debug_rows, "Run")
    if run_done:
        run_row = _phase_row(
            "Run",
            status=STATUS_DONE,
            progress=100,
            evidence=run_evidence,
            missing="Nothing required for the first run checkpoint.",
            safe_next_move="Use Results for evidence, Study designer for multi-seed support, or Peptides for MD candidate selection.",
        )
    elif run_issue:
        run_row = _blocked_row("Run", run_issue, progress=35 if signals["baseline_runs"] else 15, evidence=run_evidence, missing="Replay, proposal, discovery, or final evaluation evidence is not complete yet.")
    elif _active_action_count(actions, "Run"):
        run_row = _phase_row(
            "Run",
            status=STATUS_IN_PROGRESS,
            progress=55,
            evidence=run_evidence,
            missing="Wait for the model workflow action to finish.",
            safe_next_move="Monitor the active action and logs from Model Workflow or Operations.",
        )
    elif setup["status"] == STATUS_DONE:
        run_row = _phase_row(
            "Run",
            status=STATUS_READY,
            progress=25 if signals["baseline_runs"] else 10,
            evidence=run_evidence,
            missing="Run replay, proposal, discovery, or final evaluation from the selected run.",
            safe_next_move="Start with Replay benchmark unless you intentionally already have a proposed batch or final evaluation target.",
        )
    else:
        run_row = _phase_row(
            "Run",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=run_evidence,
            missing="Finish setup first.",
            safe_next_move="Create or pin the thesis run before running model actions.",
        )
    rows.append(run_row)

    study_done = bool(signals["study_summaries"] or signals["study_comparisons"] or signals["completed_studies"])
    study_evidence = (
        f"{signals['study_manifests']} manifest(s), {signals['completed_studies']} completed, "
        f"{signals['study_summaries']} summary file(s), {signals['study_comparisons']} comparison(s)"
    )
    study_issue = _phase_attention(action_debug_rows, "Study")
    if study_done:
        study = _phase_row(
            "Study",
            status=STATUS_DONE,
            progress=100,
            evidence=study_evidence,
            missing="No missing study evidence detected.",
            safe_next_move="Open Results -> Study comparison hub and decide which table supports the thesis narrative.",
            next_click="Results -> Study comparison hub",
            open_view="Results",
        )
    elif study_issue:
        study = _blocked_row("Study", study_issue, progress=35 if signals["study_manifests"] else 15, evidence=study_evidence, missing="A study action needs attention before the evidence is trustworthy.")
    elif _active_action_count(actions, "Study"):
        study = _phase_row(
            "Study",
            status=STATUS_IN_PROGRESS,
            progress=55,
            evidence=study_evidence,
            missing="Wait for the study action to finish or summarize.",
            safe_next_move="Keep an eye on Operations -> Study designer and Action debugger.",
        )
    elif run_row["status"] == STATUS_DONE:
        study = _phase_row(
            "Study",
            status=STATUS_READY,
            progress=15,
            evidence=study_evidence,
            missing="No seeded study summary or comparison is visible yet.",
            safe_next_move="Create a dry-run plan first, then queue the multi-seed study only when the plan looks right.",
        )
    else:
        study = _phase_row(
            "Study",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=study_evidence,
            missing="Finish the first run checkpoint before spending compute on a study.",
            safe_next_move="Complete Run first, then return to the Study designer.",
        )
    rows.append(study)

    md_done = bool(signals["labeled_reviews"] or signals["ingest_csvs"] or signals["ready_for_ingest"])
    md_ready = bool(signals["suggested_peptides"] or signals["proposed_runs"] or signals["selected_or_sent_peptides"])
    md_in_progress = bool(signals["md_campaigns"] or signals["active_slates"] or signals["analysis_ready"] or signals["selected_or_sent_peptides"])
    md_evidence = (
        f"{signals['suggested_peptides']} suggested, {signals['selected_or_sent_peptides']} selected/in MD, "
        f"{signals['md_campaigns']} campaign(s), {signals['analysis_ready']} analysis-ready, "
        f"{signals['labeled_reviews']} labeled review(s), {signals['ingest_csvs']} ingest CSV(s)"
    )
    md_issue = _phase_attention(action_debug_rows, "MD")
    if md_done:
        md = _phase_row(
            "MD",
            status=STATUS_DONE,
            progress=100,
            evidence=md_evidence,
            missing="No missing MD review evidence detected for the next handoff.",
            safe_next_move="If an ingest CSV exists, move to Ingest; otherwise create `cgmd_ingest.csv` from the reviewed full-analysis row.",
            next_click="MD Validation -> Review & ingest",
            open_view="MD Validation",
        )
    elif md_issue:
        md = _blocked_row("MD", md_issue, progress=55 if md_in_progress else 20, evidence=md_evidence, missing="The active MD/review path has a dashboard action issue.")
    elif _active_action_count(actions, "MD") or md_in_progress:
        md = _phase_row(
            "MD",
            status=STATUS_IN_PROGRESS,
            progress=65 if signals["analysis_ready"] else 45,
            evidence=md_evidence,
            missing="Finish the ladder, analyze outputs, and save the human `cgmd_label`.",
            safe_next_move="Open MD Validation and continue the safest visible ladder or review step.",
            next_click="MD Validation -> Guided ladder",
            open_view="MD Validation",
        )
    elif md_ready:
        md = _phase_row(
            "MD",
            status=STATUS_READY,
            progress=20,
            evidence=md_evidence,
            missing="Choose a candidate and launch the safe MD ladder.",
            safe_next_move="Use Peptides -> Candidate selection to select a traceable slate before launching MD.",
        )
    else:
        md = _phase_row(
            "MD",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=md_evidence,
            missing="No proposed or selected peptide is visible for MD yet.",
            safe_next_move="Generate or select candidates first.",
        )
    rows.append(md)

    ingest_done = bool(signals["import_runs"] or signals["already_ingested"])
    ingest_ready = bool(signals["ingest_csvs"] or signals["ready_for_ingest"])
    ingest_evidence = f"{signals['ingest_csvs']} ingest CSV(s), {signals['import_runs']} imported run(s), {signals['already_ingested']} already-ingested peptide(s)"
    ingest_issue = _phase_attention(action_debug_rows, "Ingest")
    if ingest_done:
        ingest = _phase_row(
            "Ingest",
            status=STATUS_DONE,
            progress=100,
            evidence=ingest_evidence,
            missing="No missing model-feedback handoff detected.",
            safe_next_move="Propose the next batch, continue another AL round, or move toward final evaluation.",
        )
    elif ingest_issue:
        ingest = _blocked_row("Ingest", ingest_issue, progress=60 if ingest_ready else 25, evidence=ingest_evidence, missing="A label ingest action needs attention before the model state can be trusted.")
    elif _active_action_count(actions, "Ingest"):
        ingest = _phase_row(
            "Ingest",
            status=STATUS_IN_PROGRESS,
            progress=70,
            evidence=ingest_evidence,
            missing="Wait for ingest/retrain to finish.",
            safe_next_move="Watch Model Workflow and Action debugger until the action succeeds.",
        )
    elif ingest_ready:
        ingest = _phase_row(
            "Ingest",
            status=STATUS_READY,
            progress=55,
            evidence=ingest_evidence,
            missing="Reviewed labels have not been fed back into the model yet.",
            safe_next_move="Run Ingest returned labels or Continue AL from reviewed peptides for the selected run.",
        )
    elif md["status"] == STATUS_DONE:
        ingest = _phase_row(
            "Ingest",
            status=STATUS_READY,
            progress=35,
            evidence=ingest_evidence,
            missing="Create `cgmd_ingest.csv` from the reviewed label first.",
            safe_next_move="Open MD Validation -> Review & ingest and create the ingest CSV.",
            next_click="MD Validation -> Review & ingest",
            open_view="MD Validation",
        )
    else:
        ingest = _phase_row(
            "Ingest",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=ingest_evidence,
            missing="Finish MD review before ingest.",
            safe_next_move="Complete the MD phase first.",
        )
    rows.append(ingest)

    freeze_done = bool(signals["freeze_count"])
    freeze_ready = bool(signals["final_metric_runs"] or ingest_done)
    freeze_evidence = f"{signals['final_metric_runs']} final-eval run(s), {signals['freeze_count']} final freeze(s)"
    freeze_issue = _phase_attention(action_debug_rows, "Freeze")
    if freeze_done:
        freeze = _phase_row(
            "Freeze",
            status=STATUS_DONE,
            progress=100,
            evidence=freeze_evidence,
            missing="Final freeze artifact is visible.",
            safe_next_move="Move to Results and export the thesis packet from frozen evidence.",
        )
    elif freeze_issue:
        freeze = _blocked_row("Freeze", freeze_issue, progress=65 if freeze_ready else 25, evidence=freeze_evidence, missing="The freeze/final-evaluation action needs attention.")
    elif _active_action_count(actions, "Freeze"):
        freeze = _phase_row(
            "Freeze",
            status=STATUS_IN_PROGRESS,
            progress=70,
            evidence=freeze_evidence,
            missing="Wait for the freeze bundle to finish.",
            safe_next_move="Monitor the action until `final_freeze/final_freeze.json` appears.",
        )
    elif freeze_ready:
        freeze = _phase_row(
            "Freeze",
            status=STATUS_READY,
            progress=55,
            evidence=freeze_evidence,
            missing="No final freeze bundle is visible yet.",
            safe_next_move="Run final evaluation if needed, then Freeze final thesis result.",
        )
    else:
        freeze = _phase_row(
            "Freeze",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=freeze_evidence,
            missing="Need final evaluation metrics or ingested thesis labels before freezing.",
            safe_next_move="Finish model feedback/final evaluation first.",
        )
    rows.append(freeze)

    export_done = bool(signals["packet_manifests"] and signals["figure_manifests"])
    export_evidence = f"{signals['packet_manifests']} thesis packet(s), {signals['figure_manifests']} figure bundle(s)"
    export_issue = _phase_attention(action_debug_rows, "Export")
    if export_done:
        export = _phase_row(
            "Export",
            status=STATUS_DONE,
            progress=100,
            evidence=export_evidence,
            missing="Packet and figure bundle are visible.",
            safe_next_move="Use the packet tables, captions, and manifests for thesis writing.",
        )
    elif export_issue:
        export = _blocked_row("Export", export_issue, progress=75 if signals["packet_manifests"] else 45, evidence=export_evidence, missing="The thesis export action needs attention.")
    elif _active_action_count(actions, "Export"):
        export = _phase_row(
            "Export",
            status=STATUS_IN_PROGRESS,
            progress=75,
            evidence=export_evidence,
            missing="Wait for the packet/figure export to finish.",
            safe_next_move="Monitor Results and Action debugger until the export artifacts appear.",
        )
    elif signals["packet_manifests"]:
        export = _phase_row(
            "Export",
            status=STATUS_READY,
            progress=75,
            evidence=export_evidence,
            missing="Build the thesis figures from the packet.",
            safe_next_move="Open Results -> Thesis output builder and build thesis figures for the selected packet.",
        )
    elif freeze["status"] == STATUS_DONE:
        export = _phase_row(
            "Export",
            status=STATUS_READY,
            progress=50,
            evidence=export_evidence,
            missing="Export a thesis packet from frozen evidence.",
            safe_next_move="Open Results -> Thesis output builder and export the thesis packet.",
        )
    else:
        export = _phase_row(
            "Export",
            status=STATUS_NOT_STARTED,
            progress=0,
            evidence=export_evidence,
            missing="Freeze the final thesis result first.",
            safe_next_move="Complete the Freeze phase first.",
        )
    rows.append(export)

    return rows


def build_thesis_phase_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    done = sum(1 for row in rows if row.get("status") == STATUS_DONE)
    blocked = sum(1 for row in rows if row.get("status") == STATUS_BLOCKED)
    in_progress = sum(1 for row in rows if row.get("status") == STATUS_IN_PROGRESS)
    ready = sum(1 for row in rows if row.get("status") == STATUS_READY)
    progress = round(sum(_safe_int(row.get("progress", 0)) for row in rows) / total) if total else 0
    next_row = next((row for row in rows if row.get("status") != STATUS_DONE), None)
    if blocked:
        verdict = "Blocked"
    elif done == total and total:
        verdict = "Complete"
    elif in_progress:
        verdict = "In progress"
    elif ready:
        verdict = "Ready"
    else:
        verdict = "Not started"
    return {
        "total": total,
        "done": done,
        "blocked": blocked,
        "in_progress": in_progress,
        "ready": ready,
        "progress": progress,
        "verdict": verdict,
        "next_phase": str(next_row.get("phase", "Complete")) if next_row else "Complete",
        "next_click": str(next_row.get("next_click", "-")) if next_row else "Results -> Thesis output builder",
        "safe_next_move": str(next_row.get("safe_next_move", "-")) if next_row else "Keep the frozen packet and figure manifests with the thesis evidence.",
        "next_status": str(next_row.get("status", STATUS_DONE)) if next_row else STATUS_DONE,
    }


def thesis_phase_display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "phase": row.get("phase", ""),
            "status": row.get("status", ""),
            "progress": f"{row.get('progress', 0)}%",
            "evidence": row.get("evidence", ""),
            "missing": row.get("missing", ""),
            "safe_next_move": row.get("safe_next_move", ""),
            "next_click": row.get("next_click", ""),
            "open_view": row.get("open_view", ""),
        }
        for row in rows
    ]


def build_thesis_phase_markdown(rows: list[dict[str, object]], summary: dict[str, object] | None = None) -> str:
    summary = summary or build_thesis_phase_summary(rows)
    lines = [
        "# Guided Thesis Phase Checklist",
        "",
        f"- Overall verdict: {summary.get('verdict', '')}",
        f"- Progress: {summary.get('progress', 0)}% ({summary.get('done', 0)}/{summary.get('total', 0)} phases done)",
        f"- Next phase: {summary.get('next_phase', '')}",
        f"- Next click: {summary.get('next_click', '')}",
        f"- Safe next move: {summary.get('safe_next_move', '')}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('phase', '')}",
                f"- Status: {row.get('status', '')} ({row.get('progress', 0)}%)",
                f"- Why it matters: {row.get('why_it_matters', '')}",
                f"- Evidence: {row.get('evidence', '')}",
                f"- Missing: {row.get('missing', '')}",
                f"- Safe next move: {row.get('safe_next_move', '')}",
                f"- Next click: {row.get('next_click', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
