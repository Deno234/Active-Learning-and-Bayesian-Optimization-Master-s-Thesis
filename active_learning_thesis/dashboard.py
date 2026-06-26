from __future__ import annotations

import ast
import csv
import html
import importlib.util
import json
import re
import subprocess
import sys
import uuid
from io import StringIO
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - dashboard still works in lean envs
    pd = None

from active_learning_thesis.dashboard_actions import (
    ACTIVE_ACTION_STATUSES,
    APPROVAL_PENDING_STATUSES,
    FINAL_ACTION_STATUSES,
    PAUSABLE_ACTION_STATUSES,
    approve_dashboard_action,
    cancel_dashboard_action,
    list_dashboard_actions,
    mark_manual_override,
    open_local_path,
    pause_dashboard_action,
    read_log_excerpt,
    rerun_dashboard_action,
    resume_dashboard_action,
    submit_dashboard_init_run_action,
    submit_export_md_source_batch_action,
    submit_finalize_md_stage_action,
    submit_ingest_round_action,
    submit_phase3_ingest_action,
    submit_phase3_make_ingest_action,
    submit_make_md_ingest_action,
    submit_bulk_make_md_ingest_action,
    submit_bulk_candidate_decision_action,
    submit_bulk_update_md_review_action,
    submit_continue_feedback_action,
    submit_freeze_final_action,
    submit_prepare_md_stage_action,
    submit_prepare_bura_md_benchmark_action,
    submit_prepare_manual_md_stage_action,
    submit_parse_bura_md_benchmark_action,
    submit_promote_reporting_md_campaign_action,
    submit_compare_studies_action,
    submit_run_study_action,
    submit_summarize_study_action,
    submit_thesis_canary_action,
    submit_thesis_figures_action,
    submit_thesis_packet_action,
    submit_update_md_review_action,
    submit_run_workflow_action,
)
from active_learning_thesis.dashboard_feedback import build_feedback_queue
from active_learning_thesis.dashboard_al_loop_simulator import (
    build_al_loop_simulation_rows,
    inject_simulated_review_labels,
    simulate_loop_ingest,
    simulate_loop_retrain_and_propose,
    start_al_loop_simulation,
)
from active_learning_thesis.dashboard_action_contracts import (
    list_dashboard_action_contracts,
)
from active_learning_thesis.dashboard_contract_ui import render_action_contract_compact
from active_learning_thesis.dashboard_action_debugger import (
    action_debug_display_rows,
    build_action_debug_packet_markdown,
    build_action_debug_rows,
    build_action_debug_summary,
)
from active_learning_thesis.dashboard_artifacts import (
    build_artifact_verification_rows,
    build_artifact_verification_summary,
)
from active_learning_thesis.dashboard_analytics import (
    METRIC_FIELDS,
    action_timeline_frame,
    discovery_frame,
    flatten_remote_jobs,
    md_ladder_summary_frame,
    remote_job_summary_frame,
    replay_curve_frame,
    run_metric_history,
)
from active_learning_thesis.dashboard_health import (
    check_all_cluster_health,
    check_cluster_health,
)
from active_learning_thesis.dashboard_md_batches import (
    find_run_md_source_batch,
    is_dashboard_md_source_batch,
    load_md_source_batch_row,
)
from active_learning_thesis.dashboard_md_slate import (
    SLATE_STAGE_CAPS,
    build_bura_utilization_summary,
    build_md_slate_monitor_rows,
    build_md_slate_launch_readiness,
    draft_md_slate_run_action,
    find_md_source_batch_csv,
    launch_md_slate_rehearsal_action,
    latest_run_md_slate,
    pause_md_slate,
    rebind_md_slate_peptide,
    recover_md_slate_peptide,
    resume_md_slate,
    retry_blocked_md_slate_peptide,
    stop_blocked_md_slate_peptide,
)
from active_learning_thesis.dashboard_md_recovery import (
    build_md_slate_exception_summary,
)
from active_learning_thesis.dashboard_readiness import (
    build_button_readiness,
    with_execution_readiness,
)
from active_learning_thesis.dashboard_layout import (
    _dashboard_ui_mode,
    inject_dashboard_theme,
    render_action_guidance as _render_action_guidance,
    render_badges as _render_badges,
    render_dashboard_hero,
    render_metric_cards as _render_metric_cards,
    render_page_guide as _render_page_guide,
    render_plan_checkpoint_table as _render_plan_checkpoint_table,
    render_recommended_card as _render_recommended_card,
    render_runner_memory_panel as _render_runner_memory_panel,
)
from active_learning_thesis.dashboard_md_view import (
    render_md_validation_view as _render_md_validation_view_impl,
)
from active_learning_thesis.dashboard_metadata import (
    BEGINNER_WORKFLOW_GUIDE,
    CANDIDATE_DECISION_TYPES,
    DECISION_TYPE_INFO,
    GUI_COVERAGE_GUIDE,
    HISTORICAL_RUN_MARKERS,
    MD_PROFILE_INFO,
    MD_STATUS_INFO,
    ML_STATUS_INFO,
    PEPTIDE_BUCKET_ORDER,
    PEPTIDE_DECISION_TYPES,
    REMOTE_SYNC_INFO,
    RUN_ACTION_INFO,
    RUN_DECISION_TYPES,
    VIEW_NAMES,
    WORKSPACE_SCOPES,
)
from active_learning_thesis.dashboard_operations_view import (
    render_operations_view as _render_operations_view_impl,
)
from active_learning_thesis.dashboard_peptides_view import (
    render_peptides_view as _render_peptides_view_impl,
)
from active_learning_thesis.dashboard_thesis_checklist import (
    build_thesis_phase_markdown,
    build_thesis_phase_rows,
    build_thesis_phase_summary,
    thesis_phase_display_rows,
)
from active_learning_thesis.dashboard_thesis_mode import render_guided_state_panel
from active_learning_thesis.dashboard_model_view import (
    render_model_workflow_view as _render_model_workflow_view_impl,
)
from active_learning_thesis.dashboard_curation import (
    hide_dashboard_run,
    ignore_dashboard_reconciliation_item,
    pin_dashboard_run,
    set_dashboard_run_label,
    show_dashboard_run,
    unpin_dashboard_run,
)
from active_learning_thesis.dashboard_decisions import (
    add_dashboard_decision,
)
from active_learning_thesis.dashboard_notifications import (
    acknowledge_dashboard_notifications,
    load_dashboard_notifications,
    save_dashboard_notifications,
)
from active_learning_thesis.dashboard_md_slate_state import (
    list_dashboard_md_slates,
)
from active_learning_thesis.dashboard_md_slate_planner_state import (
    load_dashboard_md_slate_planner,
    save_dashboard_md_slate_planner,
)
from active_learning_thesis.dashboard_progress import (
    record_dashboard_progress,
)
from active_learning_thesis.dashboard_profiles import (
    SUPPORTED_CLUSTERS,
    default_cluster_profile_path,
    get_cluster_profile,
    profile_rows,
)
from active_learning_thesis.dashboard_preferences import (
    DASHBOARD_APPROVAL_MODES,
    DASHBOARD_REFRESH_MODES,
    DASHBOARD_UI_MODES,
    DASHBOARD_WORKFLOW_MODES,
    DEFAULT_DASHBOARD_APPROVAL_MODE,
    DEFAULT_DASHBOARD_REFRESH_MODE,
    DEFAULT_DASHBOARD_UI_MODE,
    DEFAULT_DASHBOARD_WORKFLOW_MODE,
    load_dashboard_preferences,
    save_dashboard_preferences,
)
from active_learning_thesis.dashboard_results_view import (
    render_results_view as _render_results_view_impl,
)
from active_learning_thesis.dashboard_workflow_ownership import (
    canonical_navigation_hint,
    is_canonical_context,
    view_section_query_key,
    view_section_session_key,
)
from active_learning_thesis.dashboard_remote import (
    draft_bura_cancel_action,
    draft_bura_normalize_action,
    draft_bura_preflight_action,
    draft_bura_pull_package_action,
    draft_bura_submit_action,
    draft_bura_upload_campaign_action,
    draft_supek_cancel_action,
    draft_supek_pull_artifacts_action,
    draft_supek_submit_action,
    draft_supek_submit_study_array_action,
    draft_supek_submit_study_action,
    draft_supek_sync_repo_action,
    draft_supek_sync_run_action,
    draft_supek_verify_action,
    parse_qstat_output,
    parse_squeue_output,
    queue_bura_readiness_action,
    queue_bura_reconcile_campaign_action,
    queue_bura_fetch_logs_action,
    queue_bura_poll_action,
    queue_supek_preflight_action,
    queue_supek_fetch_logs_action,
    queue_supek_poll_action,
    submit_bura_full_autopilot_action,
)
from active_learning_thesis.dashboard_remote_state import (
    update_sync_status,
)
from active_learning_thesis.dashboard_remote_reconciliation import (
    build_remote_reconciliation_rows,
    build_remote_reconciliation_summary,
    filter_remote_reconciliation_rows,
)
from active_learning_thesis.dashboard_remote_watchdog import (
    build_remote_watchdog_rows,
    build_remote_watchdog_summary,
    filter_remote_watchdog_rows,
)
from active_learning_thesis.dashboard_run_setup import (
    RUN_SETUP_PRESETS,
    build_run_setup_readiness,
    coerce_positive_int,
    normalize_run_name,
    parse_strategy_list,
    run_setup_defaults,
)
from active_learning_thesis.dashboard_study_setup import (
    STUDY_PRESETS,
    discover_study_comparisons,
    discover_study_manifests,
    discover_study_summaries,
    normalize_study_name,
    parse_float_or_none,
    read_csv_rows,
    read_json_file,
    study_manifest_options,
)
from active_learning_thesis.dashboard_state_collectors import (
    discover_dashboard_run_summaries,
    load_dashboard_state_records,
)
from active_learning_thesis.md_review_evidence import (
    LABEL_CONFIDENCE_OPTIONS,
    LABEL_EVIDENCE_TAG_OPTIONS,
    LABEL_REVIEW_FIELDS,
    LABEL_RUBRIC_OPTIONS,
    default_rubric_for_label,
    label_for_rubric,
    normalize_evidence_tags,
    review_evidence_status,
    review_schema_for_row,
)
from active_learning_thesis.md_orchestrator import (
    NEXT_COMMANDS_FILENAME,
    STAGE_META_FILENAME,
)

ROUND_BATCH_PATTERN = re.compile(r"round_(\d+)_batch\.csv$")
ROUND_CANDIDATE_PATTERN = re.compile(r"round_(\d+)_scored\.csv$")
ROUND_IMPORT_PATTERN = re.compile(r"round_(\d+)_labels\.csv$")
DEFAULT_DASHBOARD_PORT = 8501
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
def _friendly_remote_sync(status: str) -> str:
    return REMOTE_SYNC_INFO.get(status, status or "-")


def _review_evidence_missing_text(status: dict[str, object]) -> str:
    missing = [str(item) for item in status.get("missing", []) if str(item).strip()]
    blockers = [str(item) for item in status.get("blockers", []) if str(item).strip()]
    issues = [*missing, *blockers]
    return ", ".join(issues) if issues else "-"


def _review_evidence_summary_row(item: dict[str, object]) -> dict[str, str]:
    status = review_evidence_status(item)
    return {
        "review_evidence": str(status.get("state", "")),
        "schema": str(status.get("schema", "")),
        "rubric": str(status.get("rubric", "")) or "-",
        "confidence": str(status.get("confidence", "")) or "-",
        "ap_sasa_200ns": str(item.get("ap_200ns", "")) or "-",
        "paper_ap_sasa_last10ns_mean": str(item.get("paper_ap_sasa_last10ns_mean", "")) or "-",
        "paper_ap_sasa_status": str(item.get("paper_ap_sasa_status", "")) or "-",
        "ap_contact_200ns": str(item.get("ap_contact_200ns", "")) or "-",
        "ap_contact_exact_paper_200ns": str(item.get("ap_contact_same_paper_formula_200ns", "")) or "-",
        "paper_path_ap_contact_200ns": str(item.get("paper_path_ap_contact_200ns", "")) or "-",
        "paper_path_ap_contact_last10ns_mean": str(item.get("paper_path_ap_contact_last10ns_mean", "")) or "-",
        "paper_path_ap_contact_last10ns_status": str(item.get("paper_path_ap_contact_last10ns_status", "")) or "-",
        "largest_cluster_200ns": str(item.get("cluster_largest_fraction_200ns", "")) or "-",
        "evidence_tags": str(status.get("evidence_tags", "")) or "-",
        "missing_or_blocked": _review_evidence_missing_text(status),
    }


def _candidate_priority_fields(
    *,
    candidate_status: str,
    source_labels: list[str],
    strategy_labels: list[str],
) -> tuple[int, str, str]:
    score = 0
    reasons: list[str] = []
    normalized_sources = {str(item).strip() for item in source_labels if str(item).strip()}
    normalized_strategies = {str(item).strip() for item in strategy_labels if str(item).strip()}

    if candidate_status == "Undecided":
        score += 5
        reasons.append("still undecided")
    elif candidate_status == "Selected for MD":
        score += 4
        reasons.append("already selected")
    elif candidate_status == "Already in MD":
        score += 2
        reasons.append("already in MD")
    elif candidate_status == "Deferred":
        score += 1
        reasons.append("explicitly deferred")
    elif candidate_status == "Rejected":
        score -= 2
        reasons.append("explicitly rejected")

    if {"Proposed next batch", "Discovery shortlist"}.issubset(normalized_sources):
        score += 4
        reasons.append("appears in both proposal and discovery")
    elif "Discovery shortlist" in normalized_sources:
        score += 3
        reasons.append("appears in discovery shortlist")
    elif "Proposed next batch" in normalized_sources:
        score += 2
        reasons.append("appears in proposed batch")

    if any("mi" in item.lower() or "mutual" in item.lower() for item in normalized_strategies):
        score += 1
        reasons.append("uncertainty-driven strategy")
    if any("ensemble" in item.lower() for item in normalized_strategies):
        score += 1
        reasons.append("ensemble-backed strategy")

    if candidate_status == "Rejected":
        band = "Do not advance"
    elif score >= 10:
        band = "Top priority"
    elif score >= 8:
        band = "High priority"
    elif score >= 6:
        band = "Worth reviewing soon"
    else:
        band = "Background"
    reason_text = ", ".join(reasons[:3]) if reasons else "general candidate pool"
    return score, band, reason_text


def _candidate_priority_rank(priority_band: str) -> int:
    return {
        "Top priority": 4,
        "High priority": 3,
        "Worth reviewing soon": 2,
        "Background": 1,
        "Do not advance": 0,
    }.get(priority_band, 0)


def _candidate_source_bucket(source_text: str) -> str:
    parts = {part.strip() for part in source_text.split("+") if part.strip()}
    if {"Proposed next batch", "Discovery shortlist"}.issubset(parts):
        return "Proposal + discovery"
    if "Discovery shortlist" in parts:
        return "Any discovery evidence"
    if "Proposed next batch" in parts:
        return "Proposal-only evidence"
    return "Other evidence"


def _candidate_focus_rows(
    rows: list[dict[str, object]],
    *,
    focus: str,
    source_focus: str,
    priority_focus: str,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        candidate_status = str(row.get("candidate_status", ""))
        source_bucket = _candidate_source_bucket(str(row.get("source", "")))
        priority_rank = _candidate_priority_rank(str(row.get("priority_band", "")))

        if source_focus == "Proposal + discovery" and source_bucket != "Proposal + discovery":
            continue
        if source_focus == "Any discovery evidence" and source_bucket not in {"Proposal + discovery", "Any discovery evidence"}:
            continue
        if source_focus == "Proposal-only evidence" and source_bucket != "Proposal-only evidence":
            continue

        if priority_focus == "Top priority only" and priority_rank < 4:
            continue
        if priority_focus == "High + Top" and priority_rank < 3:
            continue
        if priority_focus == "Review soon or higher" and priority_rank < 2:
            continue

        if focus == "Shortlist manager":
            if candidate_status == "Selected for MD":
                filtered.append(row)
                continue
            if candidate_status == "Undecided" and priority_rank >= 2:
                filtered.append(row)
                continue
            continue
        if focus == "Selected for MD next" and candidate_status != "Selected for MD":
            continue
        if focus == "Undecided pool" and candidate_status != "Undecided":
            continue
        if focus == "Discovery-backed candidates" and source_bucket not in {"Proposal + discovery", "Any discovery evidence"}:
            continue
        if focus == "Deferred / rejected" and candidate_status not in {"Deferred", "Rejected"}:
            continue
        if focus == "Already in MD" and candidate_status != "Already in MD":
            continue

        filtered.append(row)
    return filtered


def _candidate_shortlist_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        run_name = str(row.get("run", ""))
        item = summary.setdefault(
            run_name,
            {
                "run": run_name,
                "selected_for_md": 0,
                "top_undecided": 0,
                "already_in_md": 0,
                "held_back": 0,
            },
        )
        candidate_status = str(row.get("candidate_status", ""))
        priority_rank = _candidate_priority_rank(str(row.get("priority_band", "")))
        if candidate_status == "Selected for MD":
            item["selected_for_md"] = int(item["selected_for_md"]) + 1
        elif candidate_status == "Undecided" and priority_rank >= 3:
            item["top_undecided"] = int(item["top_undecided"]) + 1
        elif candidate_status == "Already in MD":
            item["already_in_md"] = int(item["already_in_md"]) + 1
        elif candidate_status in {"Deferred", "Rejected"}:
            item["held_back"] = int(item["held_back"]) + 1

    rows_out: list[dict[str, object]] = []
    for item in summary.values():
        if int(item["selected_for_md"]) > 0:
            next_focus = "Open MD Validation and start the selected peptides."
        elif int(item["top_undecided"]) > 0:
            next_focus = "Decide which high-priority candidates move into MD next."
        elif int(item["already_in_md"]) > 0:
            next_focus = "Track the existing MD ladders before adding more peptides."
        else:
            next_focus = "Keep this run as a deferred / historical candidate pool for now."
        rows_out.append({**item, "next_focus": next_focus})

    return sorted(
        rows_out,
        key=lambda row: (
            -int(row.get("selected_for_md", 0)),
            -int(row.get("top_undecided", 0)),
            str(row.get("run", "")),
        ),
    )


def _active_filter_caption(*, run_name: str, sequence: str, md_profile: str, status: str) -> str:
    parts: list[str] = []
    if run_name != "All":
        parts.append(f"run={run_name}")
    if sequence != "All":
        parts.append(f"peptide={sequence}")
    if md_profile != "All":
        parts.append(f"stage={_friendly_md_profile(md_profile)}")
    if status != "All":
        parts.append(f"status={status}")
    if not parts:
        return "No extra filters are active for this page."
    return "Active filters: " + ", ".join(parts)


def _run_next_step_copy(run: dict[str, object]) -> dict[str, str]:
    recommendation = run.get("recommended_next_step", {}) if isinstance(run.get("recommended_next_step"), dict) else {}
    title = str(recommendation.get("title", "Review run state"))
    if title == "Run the replay benchmark":
        return {
            "eyebrow": "Model workflow",
            "title": "Start with the replay benchmark",
            "summary": "This is the cleanest first thesis action for the run. It compares acquisition strategies on the initial labeled dataset without creating new peptides yet.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Local model actions below and run Replay benchmark (run-replay).",
            "next_after": "You can then propose the first real peptide batch with Propose next batch.",
        }
    if title == "Propose the next peptide batch":
        return {
            "eyebrow": "Model workflow",
            "title": "Generate the next peptide batch",
            "summary": "The baseline benchmark is already in place, so the run is ready to suggest real peptides for MD validation.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Local model actions below and run Propose next batch (propose-round).",
            "next_after": "Those peptides will then appear in Peptides and MD Validation for the guided MD ladder.",
        }
    if title == "Validate the proposed peptides":
        return {
            "eyebrow": "Model workflow",
            "title": "Move the suggested peptides into MD validation",
            "summary": "The model has already proposed peptides. The next production step is to validate them, not to propose another batch yet.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open MD Validation for one of the suggested peptides and prepare Quick package check (line_smoke).",
            "next_after": "After review and ingest, the model can run another propose/discovery/final-evaluation step.",
        }
    if title == "Ingest returned labels":
        return {
            "eyebrow": "Model workflow",
            "title": "Feed reviewed MD labels back into the model",
            "summary": "A reviewed `cgmd_ingest.csv` is available, so this run can close the active-learning loop and retrain on the newest labels.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Local model actions below, choose the reviewed ingest CSV, and run Ingest returned labels (ingest-round).",
            "next_after": "The refreshed run can then propose another batch, run discovery, or record a final evaluation.",
        }
    if title == "Continue AL from reviewed peptides":
        return {
            "eyebrow": "Model feedback",
            "title": "Close the full MD -> AL handoff",
            "summary": "The pending proposed batch is fully reviewed, so the cockpit can build the import CSV, run ingest, retrain the model, and optionally tee up the next batch.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Local model actions below and run Continue AL from reviewed peptides.",
            "next_after": "The run will return in a fresh post-ingest state, ready for the next batch, discovery, or final evaluation.",
        }
    if title == "Finish the MD feedback queue":
        return {
            "eyebrow": "Model feedback",
            "title": "Finish the pending MD feedback queue",
            "summary": str(recommendation.get("summary", "")),
            "why": str(recommendation.get("why", "")),
            "do_now": "Use Workflow summary below to see which proposed peptides still need review, promotion, or completed full-analysis outputs.",
            "next_after": "Once the whole pending batch is ready, Continue AL from reviewed peptides becomes available.",
        }
    if title == "Monitor the active SUPEK job":
        return {
            "eyebrow": "Remote SUPEK",
            "title": "Monitor the live SUPEK job",
            "summary": "A remote workflow is already active for this run, so the safest next step is monitoring rather than launching more work.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Remote SUPEK below and use Poll SUPEK queue.",
            "next_after": "Once the job finishes, pull the artifacts back into the dashboard staging area.",
        }
    if title == "Review discovery candidates or freeze a final evaluation":
        return {
            "eyebrow": "Model workflow",
            "title": "Decide whether discovery is for validation or reporting",
            "summary": "Discovery results already exist. This is the point where you choose whether to validate more peptides or freeze a thesis-ready final evaluation.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Use Workflow summary below to inspect the discovery shortlist, then run either Final evaluation or start MD on the chosen peptides.",
            "next_after": "That decision determines whether the run stays exploratory or becomes a reportable thesis result.",
        }
    if title == "Review and export thesis results":
        return {
            "eyebrow": "Thesis results",
            "title": "This run is ready for reporting",
            "summary": "The frozen holdout evaluation already exists, so this run has crossed from experimentation into thesis-results mode.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Use Workflow summary below for the curated metrics, then Technical details if you need raw files for plots or tables.",
            "next_after": "You can compare this run against other pinned thesis runs on the dashboard.",
        }
    if title == "Freeze final thesis result":
        return {
            "eyebrow": "Thesis freeze",
            "title": "Freeze this run before reporting it",
            "summary": "Final holdout metrics exist, but the reproducibility freeze, checks, and model card are still missing.",
            "why": str(recommendation.get("why", "")),
            "do_now": "Open Thesis freeze below and run Freeze final thesis result.",
            "next_after": "Then use Results -> Thesis output builder to export the packet and build thesis figures.",
        }
    return {
        "eyebrow": "Model workflow",
        "title": title,
        "summary": str(recommendation.get("summary", "")),
        "why": str(recommendation.get("why", "")),
        "do_now": "Use the sections below to inspect the run state, then act in either Local model actions or Remote SUPEK.",
        "next_after": "",
    }


def _recommended_run_workflow_command(run: dict[str, object]) -> str:
    ml_status = str(run.get("ml_status", "config-only"))
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
    if bool(feedback_queue.get("can_continue", False)):
        return "continue-feedback"
    if list(run.get("available_ingest_csvs", [])):
        return "ingest-round"
    if ml_status in {"config-only", "initialized"}:
        return "run-replay"
    if ml_status == "replay-complete":
        return "propose-round"
    if ml_status == "discovery-complete":
        return "evaluate-final"
    return ""


def _ladder_next_step_copy(ladder: dict[str, object]) -> dict[str, str]:
    next_step = ladder.get("next_step", {}) if isinstance(ladder.get("next_step"), dict) else {}
    title = str(next_step.get("title", "Review ladder state"))
    if title.startswith("Prepare "):
        return {
            "eyebrow": "MD validation",
            "title": title,
            "summary": "The next missing guided MD rung has not been prepared yet, so the ladder still needs a local campaign package before any BURA action.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Local MD actions below and prepare the next campaign locally.",
            "next_after": "That campaign can then be uploaded to BURA and sent through the remote chain.",
        }
    if title == "Upload the campaign to BURA":
        return {
            "eyebrow": "Remote BURA",
            "title": "Stage the campaign on BURA",
            "summary": "The campaign already exists locally. The next remote action is to copy it to BURA so normalization, preflight, and submit can happen there.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Remote BURA below and use Upload campaign to BURA.",
            "next_after": "After upload, normalize the scripts, run preflight, and then submit the chain.",
        }
    if title == "Normalize, preflight, then submit":
        return {
            "eyebrow": "Remote BURA",
            "title": "Run the BURA pre-submit checks",
            "summary": "The campaign is already on BURA. The next safe remote steps are script normalization, preflight, and chain submission.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Remote BURA below and go in order: Normalize BURA scripts, Run BURA preflight, then Submit BURA chain.",
            "next_after": "Once submitted, the next action becomes queue monitoring instead of more setup.",
        }
    if title == "Monitor the active BURA campaign":
        return {
            "eyebrow": "Remote BURA",
            "title": "Monitor the running BURA campaign",
            "summary": "A remote BURA job is already tracked for this peptide, so the safe action now is monitoring and eventual copy-back.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Remote BURA below and use Poll BURA queue until the chain finishes.",
            "next_after": "Then copy the outputs back and re-parse them locally.",
        }
    if title == "Re-parse the staged outputs locally":
        return {
            "eyebrow": "Local MD action",
            "title": "Re-parse the staged outputs",
            "summary": "The outputs are already downloaded into the dashboard staging area. The remaining step is to parse them into the local ladder state.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Local MD actions below and run the re-parse/finalize action.",
            "next_after": "The ladder will then either advance to the next stage or become review-ready.",
        }
    if title == "Finalize the copied-back outputs":
        return {
            "eyebrow": "Local MD action",
            "title": "Finalize the copied-back outputs",
            "summary": "The BURA outputs are already back in the campaign folder. The next local step is to parse and register them into the ladder state.",
            "why": str(next_step.get("why", "")),
            "do_now": "Open Local MD actions below and run Finalize local outputs.",
            "next_after": "That finalize step decides the next ladder rung or unlocks review.",
        }
    if title == "Review the full-analysis result":
        return {
            "eyebrow": "Review",
            "title": "Assign the human MD label",
            "summary": "The full analysis is finished. This peptide now needs a human review and a `cgmd_label` before it can be fed back into the model.",
            "why": str(next_step.get("why", "")),
            "do_now": "Use Review & ingest below to inspect the current campaign, then assign the `cgmd_label` in `md_review.csv`.",
            "next_after": "Once the label exists, the ingest CSV can be created locally.",
        }
    if title == "Create the ingest CSV":
        return {
            "eyebrow": "Model feedback",
            "title": "Create the ingest CSV for this peptide",
            "summary": "The full-analysis result already has a reviewed label, so this peptide is ready to become a model-ingest row.",
            "why": str(next_step.get("why", "")),
            "do_now": "Use Review & ingest below to confirm the label, then open Local MD actions and run Create ingest CSV.",
            "next_after": "That CSV can then be ingested from Model Workflow to retrain the model.",
        }
    return {
        "eyebrow": "MD validation",
        "title": title,
        "summary": str(next_step.get("summary", "")),
        "why": str(next_step.get("why", "")),
        "do_now": "Use the sections below to inspect the ladder and act in either Local MD actions or Remote BURA.",
        "next_after": "",
    }


def _recommended_ladder_macro_key(ladder: dict[str, object]) -> str:
    next_step = ladder.get("next_step", {}) if isinstance(ladder.get("next_step"), dict) else {}
    title = str(next_step.get("title", ""))
    if title.startswith("Prepare "):
        return "prepare-local"
    if title == "Upload the campaign to BURA":
        return "upload-bura"
    if title == "Normalize, preflight, then submit":
        return "normalize-bura"
    if title == "Monitor the active BURA campaign":
        return "poll-bura"
    if title in {"Re-parse the staged outputs locally", "Finalize the copied-back outputs"}:
        return "finalize-local"
    if title == "Create the ingest CSV":
        return "make-ingest"
    return ""


def _run_plan_rows(run: dict[str, object]) -> list[dict[str, str]]:
    run_dir_text = str(run.get("run_dir", "")).strip()
    run_dir = Path(run_dir_text) if run_dir_text else None
    freeze_ready = bool(run_dir and (run_dir / "final_freeze" / "final_freeze.json").exists())
    steps = [
        ("Replay benchmark", bool(_frame_records(replay_curve_frame(run)) or str(run.get("ml_status", "")) in {"replay-complete", "batch-proposed", "discovery-complete", "final-evaluated"})),
        ("Propose next batch", bool((run.get("latest_batch") or {}).get("rows", []))),
        ("Ingest returned labels", bool(list(run.get("import_rows", [])))),
        ("Run discovery", bool(_frame_records(discovery_frame(run)))),
        ("Final evaluation", bool(run.get("final_metrics"))),
        ("Freeze final thesis result", freeze_ready),
    ]
    current_title = str((run.get("recommended_next_step") or {}).get("title", ""))
    rows: list[dict[str, str]] = []
    current_marked = False
    for label, complete in steps:
        if complete:
            status = "complete"
        elif not current_marked:
            status = "current"
            current_marked = True
        else:
            status = "upcoming"
        if current_title and status == "current":
            detail = current_title
        else:
            detail = ""
        rows.append({"checkpoint": label, "status": status, "detail": detail})
    return rows


def _remote_run_plan_rows(run: dict[str, object]) -> list[dict[str, str]]:
    remote_status = str(run.get("remote_sync_status", "not_synced"))
    active_job = bool(str(run.get("remote_job_id", "")))
    rows = [
        {"checkpoint": "Stage run on SUPEK", "status": "complete" if remote_status != "not_synced" else "current", "detail": ""},
        {"checkpoint": "Submit next SUPEK workflow", "status": "complete" if remote_status in {"submitted", "running", "outputs_staged", "outputs_returned"} else ("current" if remote_status == "staged_remote" else "upcoming"), "detail": ""},
        {"checkpoint": "Monitor remote queue", "status": "complete" if remote_status in {"outputs_staged", "outputs_returned"} else ("current" if active_job and remote_status in {"submitted", "running"} else "upcoming"), "detail": str(run.get("remote_job_id", "")) or ""},
        {"checkpoint": "Pull artifacts back", "status": "complete" if remote_status in {"outputs_staged", "outputs_returned"} else ("current" if remote_status in {"submitted", "running"} and not active_job else "upcoming"), "detail": ""},
    ]
    return rows


def _ladder_plan_rows(ladder: dict[str, object]) -> list[dict[str, str]]:
    current = ladder.get("current")
    sync_status = str(ladder.get("sync_status", "not_synced"))
    ready_for_review = bool(ladder.get("ready_for_review"))
    full_item = ladder.get("full") or {}
    label_value = str(full_item.get("cgmd_label", "")).strip()
    ingest_exists = _ingest_csv_path(str(full_item.get("campaign_dir", ""))).exists() if full_item else False
    current_title = str((ladder.get("next_step") or {}).get("title", ""))
    steps = [
        ("Prepare local campaign", bool(current)),
        ("Upload to BURA", sync_status != "not_synced"),
        ("Normalize / preflight / submit", sync_status in {"submitted", "running", "outputs_staged", "outputs_returned", "finalized_local"}),
        ("Monitor remote queue", sync_status in {"running", "outputs_staged", "outputs_returned", "finalized_local"}),
        ("Copy outputs back", sync_status in {"outputs_staged", "outputs_returned", "finalized_local"}),
        ("Finalize locally", sync_status == "finalized_local" or ready_for_review),
        ("Review label", label_value in {"0", "1"}),
        ("Create ingest CSV", ingest_exists),
    ]
    rows: list[dict[str, str]] = []
    current_marked = False
    for label, complete in steps:
        if complete:
            status = "complete"
        elif not current_marked:
            status = "current"
            current_marked = True
        else:
            status = "upcoming"
        detail = current_title if status == "current" else ""
        rows.append({"checkpoint": label, "status": status, "detail": detail})
    return rows


def _progress_events_for_context(
    state: dict[str, object],
    *,
    scope: str,
    plan_kind: str,
    run_dir: str = "",
    sequence: str = "",
    campaign_dir: str = "",
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for entry in state.get("progress_events", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("scope", "")) != scope:
            continue
        if str(entry.get("plan_kind", "")) != plan_kind:
            continue
        if run_dir and str(entry.get("run_dir", "")) != run_dir:
            continue
        if sequence and str(entry.get("sequence", "")) != sequence:
            continue
        if campaign_dir and str(entry.get("campaign_dir", "")) != campaign_dir:
            continue
        events.append(entry)
    return events


def _record_runner_progress(
    run_root: Path,
    *,
    scope: str,
    plan_kind: str,
    checkpoint: str,
    action_label: str,
    action: dict[str, object],
    run_dir: str = "",
    sequence: str = "",
    campaign_dir: str = "",
    note: str = "",
) -> None:
    record_dashboard_progress(
        run_root,
        scope=scope,
        plan_kind=plan_kind,
        checkpoint=checkpoint,
        action_label=action_label,
        run_dir=run_dir,
        sequence=sequence,
        campaign_dir=campaign_dir,
        action_status=str(action.get("status", "")),
        action_id=str(action.get("id", "")),
        note=note,
    )


def _primary_dashboard_recommendation(state: dict[str, object]) -> dict[str, str]:
    overview = state.get("overview", {}) if isinstance(state.get("overview", {}), dict) else {}
    cluster_rows = [row for row in _cluster_health_rows(state) if str(row.get("status", "")) != "ok"]
    if cluster_rows:
        first = cluster_rows[0]
        cluster_name = str(first.get("cluster", "")).upper()
        return {
            "eyebrow": "Cluster readiness",
            "title": f"Resolve the {cluster_name} blocker first",
            "summary": str(first.get("summary", "")),
            "why": str(first.get("hint", "")),
            "do_now": f"Open Operations and re-check {cluster_name} health before relying on remote actions.",
            "next_after": "Once cluster health is green again, the remote workflow buttons become trustworthy.",
        }
    approvals = list(overview.get("approval_queue", []))
    if approvals:
        first = approvals[0]
        return {
            "eyebrow": "Approval queue",
            "title": f"Review the pending {first.get('scope', 'action')}",
            "summary": f"{first.get('title', 'A dashboard action')} is waiting for approval before it can run.",
            "why": "Mutating remote actions are intentionally draft-first so you always have a chance to sanity-check them.",
            "do_now": "Open Operations -> Approval queue and approve or pause the pending action.",
            "next_after": "After approval, monitoring and artifact-return actions become available.",
        }
    queue = list(overview.get("today_queue", []))
    if queue:
        first = queue[0]
        return {
            "eyebrow": str(first.get("category", "Today")),
            "title": str(first.get("action_now", "Review current state")),
            "summary": str(first.get("why", "")),
            "why": "This is the strongest next step inferred from the currently visible thesis workspace.",
            "do_now": f"Open {first.get('open_view', 'the relevant page')} and focus on {first.get('target', 'the highlighted item')}.",
            "next_after": "The queue below shows the other actions that are ready behind this one.",
        }
    return {
        "eyebrow": "Today",
        "title": "Review the current thesis workspace",
        "summary": "Nothing urgent was inferred automatically, so this is a good moment to inspect suggested peptides, ready-for-ingest items, and blocked ladders.",
        "why": "",
        "do_now": "Use the queue and peptide tables below to choose the next thesis action.",
        "next_after": "",
    }


def _today_queue_priority(*, category: str, target: str) -> int:
    category_order = {
        "Slate recovery": 5,
        "Remote reconciliation": 6,
        "Review": 10,
        "Ingest": 20,
        "AL promotion": 25,
        "Remote monitoring": 30,
        "MD preparation": 40,
        "Candidate selection": 45,
        "Model workflow": 50,
        "Reporting": 60,
    }
    return category_order.get(category, 100) * 1000 + sum(ord(char) for char in target[:32])


def _queue_row(*, category: str, target: str, action_now: str, why: str, open_view: str) -> dict[str, str]:
    return {
        "category": category,
        "target": target,
        "action_now": action_now,
        "why": why,
        "open_view": open_view,
    }


def _result_summary_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        baseline = run.get("baseline_metrics", {}) if isinstance(run.get("baseline_metrics", {}), dict) else {}
        final = run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {}
        row = {
            "run": str(run.get("run_display_name", run.get("run_name", ""))),
            "model_state": str(run.get("ml_status_label", run.get("ml_status", ""))),
            "latest_round": run.get("latest_round_id", 0),
            "proposed_peptides": len((run.get("latest_batch") or {}).get("rows", [])),
            "ingested_labels": len(run.get("import_rows", [])),
        }
        for metric_name in METRIC_FIELDS:
            row[f"baseline_{metric_name}"] = baseline.get(metric_name, "")
            row[f"final_{metric_name}"] = final.get(metric_name, "")
        rows.append(row)
    return rows


def _reporting_readiness_for_run(run: dict[str, object]) -> tuple[str, str]:
    if run.get("final_metrics"):
        return "Report-ready", "Frozen holdout metrics exist and can be used directly in thesis comparisons."
    if list(run.get("available_ingest_csvs", [])):
        return "Needs ingest", "Reviewed MD labels exist, but the run has not been updated with them yet."
    if list(run.get("import_rows", [])):
        return "Loop active", "Returned labels have already been ingested, but the final frozen evaluation is still missing."
    if str(run.get("ml_status", "")) == "discovery-complete":
        return "Discovery evidence only", "Discovery summaries exist, but there is no frozen final evaluation yet."
    if str(run.get("ml_status", "")) == "replay-complete":
        return "Benchmark only", "This run is still at the initial-dataset benchmarking stage."
    return "In progress", "This run still needs more workflow progress before it becomes thesis-report-ready."


def _result_scorecard_rows(runs: list[dict[str, object]], *, metric_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        baseline_value = _safe_float(run.get("baseline_metrics", {}).get(metric_name, "")) if isinstance(run.get("baseline_metrics", {}), dict) else None
        final_value = _safe_float(run.get("final_metrics", {}).get(metric_name, "")) if isinstance(run.get("final_metrics", {}), dict) else None
        readiness, note = _reporting_readiness_for_run(run)
        delta = final_value - baseline_value if baseline_value is not None and final_value is not None else None
        rows.append(
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "model_state": str(run.get("ml_status_label", run.get("ml_status", ""))),
                "reporting_readiness": readiness,
                f"baseline_{metric_name}": _format_float(baseline_value),
                f"final_{metric_name}": _format_float(final_value),
                f"delta_{metric_name}": _format_float(delta),
                "latest_round": run.get("latest_round_id", 0),
                "note": note,
            }
        )
    return rows


def _thesis_milestone_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        baseline_ready = bool(run.get("baseline_metrics"))
        replay_ready = bool(_frame_records(replay_curve_frame(run)))
        proposed_ready = bool((run.get("latest_batch") or {}).get("rows", []))
        md_feedback_ready = bool(list(run.get("available_ingest_csvs", [])))
        labels_ingested = bool(list(run.get("import_rows", [])))
        discovery_ready = bool(_frame_records(discovery_frame(run)))
        final_ready = bool(run.get("final_metrics"))
        readiness, note = _reporting_readiness_for_run(run)
        completed = sum(
            1
            for flag in (
                baseline_ready,
                replay_ready,
                proposed_ready,
                md_feedback_ready,
                labels_ingested,
                discovery_ready,
                final_ready,
            )
            if flag
        )
        next_focus = str((run.get("recommended_next_step") or {}).get("title", "Review run state"))
        rows.append(
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "baseline_ready": _friendly_bool(baseline_ready),
                "replay_ready": _friendly_bool(replay_ready),
                "batch_proposed": _friendly_bool(proposed_ready),
                "md_feedback_ready": _friendly_bool(md_feedback_ready),
                "labels_ingested": _friendly_bool(labels_ingested),
                "discovery_ready": _friendly_bool(discovery_ready),
                "final_evaluated": _friendly_bool(final_ready),
                "completed_milestones": completed,
                "reporting_readiness": readiness,
                "next_focus": next_focus,
                "note": note,
            }
        )
    readiness_priority = {
        "Needs ingest": 10,
        "In progress": 20,
        "Benchmark only": 30,
        "Discovery evidence only": 40,
        "Loop active": 50,
        "Report-ready": 60,
    }
    return sorted(
        rows,
        key=lambda row: (
            readiness_priority.get(str(row.get("reporting_readiness", "")), 100),
            -int(row.get("completed_milestones", 0) or 0),
            str(row.get("run", "")),
        ),
    )


def _focused_comparison_rows(runs: list[dict[str, object]], *, metric_name: str) -> list[dict[str, object]]:
    scorecards = {
        str(row.get("run", "")): row
        for row in _result_scorecard_rows(runs, metric_name=metric_name)
    }
    milestones = {
        str(row.get("run", "")): row
        for row in _thesis_milestone_rows(runs)
    }
    replay_best = {
        str(row.get("run", "")): row
        for row in _replay_best_strategy_rows(runs, metric_name=metric_name)
    }
    discovery_best = {
        str(row.get("run", "")): row
        for row in _discovery_highlight_rows(runs)
    }
    rows: list[dict[str, object]] = []
    for run in runs:
        run_name = str(run.get("run_display_name", run.get("run_name", "")))
        scorecard = scorecards.get(run_name, {})
        milestone = milestones.get(run_name, {})
        replay_row = replay_best.get(run_name, {})
        discovery_row = discovery_best.get(run_name, {})
        rows.append(
            {
                "run": run_name,
                "model_state": str(run.get("ml_status_label", run.get("ml_status", ""))),
                "reporting_readiness": str(scorecard.get("reporting_readiness", "")),
                f"baseline_{metric_name}": str(scorecard.get(f"baseline_{metric_name}", "")),
                f"final_{metric_name}": str(scorecard.get(f"final_{metric_name}", "")),
                f"delta_{metric_name}": str(scorecard.get(f"delta_{metric_name}", "")),
                "proposed_peptides": len((run.get("latest_batch") or {}).get("rows", [])),
                "ingested_labels": len(run.get("import_rows", [])),
                "completed_milestones": milestone.get("completed_milestones", 0),
                "replay_best_strategy": str(replay_row.get("best_strategy", "")) or "-",
                "discovery_best_strategy": str(discovery_row.get("strategy", "")) or "-",
                "next_focus": str((run.get("recommended_next_step") or {}).get("title", "")),
            }
        )
    return rows


def _selected_results_runs(st, runs: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    run_options = [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs]
    none_option = "None"
    primary_name = st.selectbox(
        "Primary thesis run",
        run_options,
        index=0,
        key="results_compare_primary",
    )
    secondary_options = [none_option, *run_options]
    secondary_default = run_options[1] if len(run_options) > 1 else none_option
    secondary_name = st.selectbox(
        "Compare against",
        secondary_options,
        index=(secondary_options.index(secondary_default) if secondary_default in secondary_options else 0),
        key="results_compare_secondary",
    )
    tertiary_name = st.selectbox(
        "Optional third run",
        secondary_options,
        index=0,
        key="results_compare_tertiary",
    )
    chosen_names: list[str] = []
    for name in [primary_name, secondary_name, tertiary_name]:
        if name == none_option or name in chosen_names:
            continue
        chosen_names.append(name)
    selected_runs = [
        run
        for run in runs
        if str(run.get("run_display_name", run.get("run_name", ""))) in chosen_names
    ]
    return selected_runs, chosen_names


def _replay_best_strategy_rows(runs: list[dict[str, object]], *, metric_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        replay_rows = _frame_records(replay_curve_frame(run))
        if not replay_rows:
            continue
        latest_per_strategy: dict[str, dict[str, object]] = {}
        for row in replay_rows:
            strategy = str(row.get("strategy", ""))
            round_id = int(row.get("round_id", 0) or 0)
            current = latest_per_strategy.get(strategy)
            if current is None or round_id >= int(current.get("round_id", 0) or 0):
                latest_per_strategy[strategy] = row
        best_strategy = max(
            latest_per_strategy.values(),
            key=lambda row: (_safe_float(row.get(metric_name)) or float("-inf")),
        )
        rows.append(
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "best_strategy": str(best_strategy.get("strategy", "")),
                metric_name: _format_float(_safe_float(best_strategy.get(metric_name))),
                "labeled_count": best_strategy.get("labeled_count", ""),
                "round_id": best_strategy.get("round_id", ""),
            }
        )
    return rows


def _discovery_highlight_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        discovery_rows = _frame_records(discovery_frame(run))
        if not discovery_rows:
            continue
        best = max(
            discovery_rows,
            key=lambda row: (_safe_float(row.get("top_batch_mean_utility_score")) or float("-inf")),
        )
        rows.append(
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "strategy": str(best.get("strategy", "")),
                "exported_count": best.get("exported_count", ""),
                "utility_score": _format_float(_safe_float(best.get("top_batch_mean_utility_score"))),
                "pred_std": _format_float(_safe_float(best.get("top_batch_mean_pred_std"))),
                "candidate_sequences": str(best.get("top_batch_sequences", "")),
            }
        )
    return rows


def _review_feedback_audit_rows(state: dict[str, object]) -> list[dict[str, object]]:
    ladders = list(state.get("peptides", []))
    actions = list(state.get("actions", []))
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    ingested_rows = list(inventory.get("already_ingested", [])) if isinstance(inventory.get("already_ingested", []), list) else []
    ingested_by_key = {
        (str(row.get("run", "")), str(row.get("sequence", ""))): row
        for row in ingested_rows
    }
    latest_review_save: dict[tuple[str, str, str], str] = {}
    for action in actions:
        if str(action.get("kind", "")) != "update-md-review":
            continue
        key = (
            str(action.get("related_run", "")),
            str(action.get("related_sequence", "")),
            str(action.get("related_campaign", "")),
        )
        timestamp = str(action.get("finished_at") or action.get("started_at") or action.get("created_at") or "")
        if not timestamp:
            continue
        current = latest_review_save.get(key, "")
        if not current or timestamp >= current:
            latest_review_save[key] = timestamp

    rows: list[dict[str, object]] = []
    for ladder in ladders:
        full_item = ladder.get("full")
        if not full_item:
            continue
        run_label = str(ladder.get("run_display_name", "") or _path_name(ladder.get("run_dir", "")))
        run_key = str(ladder.get("run_dir", ""))
        sequence = str(ladder.get("sequence", ""))
        campaign_dir = str(full_item.get("campaign_dir", ""))
        campaign = str(full_item.get("campaign", ""))
        label_value = str(full_item.get("cgmd_label", "")).strip()
        ingest_csv_path = _ingest_csv_path(campaign_dir)
        ingest_exists = ingest_csv_path.exists()
        ingest_supported = bool(ladder.get("ingest_supported", True))
        promotion_available = bool(ladder.get("promotion_available", False))
        promoted_at = str(full_item.get("promoted_to_real_batch_at", "")).strip()
        promoted_round = str(full_item.get("promoted_round_id", "")).strip()
        promoted_from_batch = str(full_item.get("promoted_from_source_batch_csv", "")).strip()
        current_source_batch = str(full_item.get("source_batch_csv", "")).strip()
        ingested_row = ingested_by_key.get((run_label, sequence))
        if promoted_at:
            promotion_state = f"Promoted into real batch (round {promoted_round})" if promoted_round else "Promoted into real batch"
        elif label_value in {"0", "1"} and promotion_available and not ingest_supported:
            promotion_state = (
                f"Can promote now (round {ladder.get('promotion_target_round_id', '')})"
                if str(ladder.get("promotion_target_round_id", "")).strip()
                else "Can promote now"
            )
        elif not ingest_supported:
            promotion_state = "Waiting for real proposed batch"
        else:
            promotion_state = "Not needed"
        if ingested_row:
            feedback_state = "Already ingested"
            next_step = "Use the updated run state for the next model step."
        elif label_value in {"0", "1"} and ingest_exists and ingest_supported:
            feedback_state = "Ready for model ingest"
            next_step = "Run Ingest returned labels in Model Workflow."
        elif label_value in {"0", "1"} and ingest_supported:
            feedback_state = "Needs ingest CSV"
            next_step = "Create cgmd_ingest.csv from the reviewed label."
        elif label_value in {"0", "1"} and promotion_available:
            feedback_state = "Can promote into real batch"
            next_step = "Promote this reporting-only MD result into the real proposed batch."
        elif label_value in {"0", "1"}:
            feedback_state = "Reporting only"
            next_step = "Wait until the peptide appears in a real proposed batch, then promote it."
        elif ladder.get("ready_for_review"):
            feedback_state = "Needs human review"
            next_step = "Assign cgmd_label in md_review.csv."
        else:
            feedback_state = "Full analysis incomplete"
            next_step = str((ladder.get("next_step") or {}).get("title", "Complete the full analysis rerun"))
        rows.append(
            {
                "run": run_label,
                "sequence": sequence,
                "campaign": campaign,
                "full_status": _friendly_md_status(str(full_item.get("job_root_status", ""))),
                "feedback_state": feedback_state,
                "current_label": label_value or "-",
                "review_notes": str(full_item.get("review_notes", "")) or "-",
                "ingest_csv": _path_name(ingest_csv_path) if ingest_exists else "-",
                "already_ingested": "yes" if ingested_row else "no",
                "promotion_state": promotion_state,
                "promotion_from_batch": _path_name(promoted_from_batch) or "-",
                "current_source_batch": _path_name(current_source_batch) or "-",
                "promoted_at": promoted_at or "-",
                "promoted_round": promoted_round or "-",
                "last_review_saved": latest_review_save.get((run_key, sequence, campaign_dir), "-"),
                "next_feedback_step": next_step,
            }
        )
    priority = {
        "Needs human review": 10,
        "Can promote into real batch": 20,
        "Needs ingest CSV": 30,
        "Ready for model ingest": 40,
        "Reporting only": 45,
        "Already ingested": 40,
        "Full analysis incomplete": 50,
    }
    return sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("feedback_state", "")), 100),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )


def _promotion_audit_rows(state: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ladder in list(state.get("peptides", [])):
        full_item = ladder.get("full")
        if not full_item:
            continue
        promoted_from_batch = str(full_item.get("promoted_from_source_batch_csv", "")).strip()
        promoted_at = str(full_item.get("promoted_to_real_batch_at", "")).strip()
        current_source_batch = str(full_item.get("source_batch_csv", "")).strip()
        source_batch_kind = str(full_item.get("source_batch_kind", ladder.get("source_batch_kind", ""))).strip()
        promotion_available = bool(ladder.get("promotion_available", False))
        if not (promoted_from_batch or source_batch_kind == "dashboard_generated" or promotion_available):
            continue
        sequence = str(ladder.get("sequence", ""))
        run_label = str(ladder.get("run_display_name", "") or _path_name(ladder.get("run_dir", "")))
        campaign = str(full_item.get("campaign", ""))
        label_value = str(full_item.get("cgmd_label", "")).strip() or "-"
        ingest_supported = bool(ladder.get("ingest_supported", True))
        promotion_target_batch = str(ladder.get("promotion_target_batch_csv", "")).strip()
        promotion_target_round = str(ladder.get("promotion_target_round_id", "")).strip()
        promoted_round = str(full_item.get("promoted_round_id", "")).strip()
        if promoted_at:
            promotion_state = f"Promoted into real batch (round {promoted_round})" if promoted_round else "Promoted into real batch"
            next_step = (
                "Create cgmd_ingest.csv or ingest the label now."
                if label_value in {"0", "1"}
                else "Assign cgmd_label, then continue with ingest."
            )
        elif promotion_available:
            promotion_state = (
                f"Can promote now (round {promotion_target_round})"
                if promotion_target_round
                else "Can promote now"
            )
            next_step = "Use the Promotion bridge in MD Validation -> Review & ingest."
        else:
            promotion_state = "Waiting for real proposed batch"
            next_step = "Keep this peptide as reporting-only until a real proposed batch contains it."
        rows.append(
            {
                "run": run_label,
                "sequence": sequence,
                "campaign": campaign,
                "promotion_state": promotion_state,
                "label": label_value,
                "source_batch_kind": _source_batch_kind_label(source_batch_kind),
                "original_source_batch": _path_name(promoted_from_batch or current_source_batch) or "-",
                "current_source_batch": _path_name(current_source_batch) or "-",
                "target_batch_csv": _path_name(promotion_target_batch or current_source_batch) or "-",
                "target_round": promotion_target_round or promoted_round or "-",
                "promoted_at": promoted_at or "-",
                "ingest_support": "AL-ingestable" if ingest_supported else "Reporting-only batch",
                "next_step": next_step,
            }
        )
    priority = {
        "Can promote now": 10,
        "Waiting for real proposed batch": 20,
        "Promoted into real batch": 30,
    }
    return sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("promotion_state", "")).split(" (", 1)[0], 100),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )


def _peptide_provenance_audit_rows(state: dict[str, object]) -> list[dict[str, object]]:
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    ledger_rows = list(inventory.get("ledger", [])) if isinstance(inventory.get("ledger", []), list) else []
    candidate_rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    review_rows = _review_feedback_audit_rows(state)
    promotion_rows = _promotion_audit_rows(state)
    runs = list(state.get("runs", []))
    ladders = list(state.get("peptides", []))
    slates = list(state.get("md_slates", []))
    decisions = list(state.get("decisions", []))

    candidate_by_key = {
        (str(row.get("run", "")), str(row.get("sequence", ""))): row
        for row in candidate_rows
    }
    review_by_key = {
        (str(row.get("run", "")), str(row.get("sequence", ""))): row
        for row in review_rows
    }
    promotion_by_key = {
        (str(row.get("run", "")), str(row.get("sequence", ""))): row
        for row in promotion_rows
    }
    run_label_by_dir = {
        _canonical_path(str(run.get("run_dir", ""))): str(run.get("run_display_name", run.get("run_name", "")))
        for run in runs
    }
    ladder_by_key = {
        (str(ladder.get("run_display_name", "") or _path_name(ladder.get("run_dir", ""))), str(ladder.get("sequence", ""))): ladder
        for ladder in ladders
    }
    import_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for run in runs:
        run_label = str(run.get("run_display_name", run.get("run_name", "")))
        for row in list(run.get("import_rows", [])):
            sequence = str(row.get("sequence", "")).strip()
            if not sequence:
                continue
            key = (run_label, sequence)
            current = import_by_key.get(key)
            if current is None or int(str(row.get("round_id", "0")) or 0) >= int(str(current.get("round_id", "0")) or 0):
                import_by_key[key] = row

    latest_decision_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("scope", "")).strip() not in {"candidate", "peptide"}:
            continue
        run_dir = _canonical_path(str(entry.get("run_dir", "")))
        run_label = run_label_by_dir.get(run_dir, "")
        sequence = str(entry.get("sequence", "")).strip()
        if not run_label or not sequence:
            continue
        key = (run_label, sequence)
        timestamp = str(entry.get("created_at", "")).strip()
        current = latest_decision_by_key.get(key)
        if current is None or timestamp >= str(current.get("created_at", "")).strip():
            latest_decision_by_key[key] = entry

    latest_slate_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for slate in sorted(
        slates,
        key=lambda item: (str(item.get("updated_at", "")), str(item.get("created_at", "")), str(item.get("slate_id", ""))),
    ):
        run_label = run_label_by_dir.get(_canonical_path(str(slate.get("run_dir", ""))), _path_name(str(slate.get("run_dir", ""))))
        for peptide in list(slate.get("peptides", [])):
            if not isinstance(peptide, dict):
                continue
            sequence = str(peptide.get("sequence", "")).strip()
            if not sequence:
                continue
            latest_slate_by_key[(run_label, sequence)] = {
                "slate_id": str(slate.get("slate_id", "")),
                "slate_status": str(slate.get("effective_status", slate.get("status", ""))),
                "peptide_status": str(peptide.get("status", "")),
                "stage_history": list(peptide.get("stage_history", [])) if isinstance(peptide.get("stage_history", []), list) else [],
                "remote_job_id": str(peptide.get("remote_job_id", "")),
                "waiting_reason": str(peptide.get("waiting_reason", "")),
                "failure_reason": str(peptide.get("failure_reason", "")),
            }

    def _stage_trace(ladder: dict[str, object] | None, slate_info: dict[str, object] | None) -> str:
        if isinstance(slate_info, dict):
            history = list(slate_info.get("stage_history", []))
            if history:
                ordered = sorted(
                    [item for item in history if isinstance(item, dict)],
                    key=lambda item: _md_profile_sort_key(str(item.get("md_profile", ""))),
                )
                parts = []
                for item in ordered:
                    profile = _friendly_md_profile(str(item.get("md_profile", "")), short=True)
                    status = str(item.get("status", "")).strip() or "-"
                    step = str(item.get("step", "")).strip()
                    if step and status not in {"completed", "blocked"}:
                        parts.append(f"{profile}: {status} ({step})")
                    else:
                        parts.append(f"{profile}: {status}")
                if parts:
                    return " | ".join(parts)
        if isinstance(ladder, dict):
            campaigns = list(ladder.get("campaigns", []))
            if campaigns:
                return " | ".join(
                    f"{_friendly_md_profile(str(item.get('md_profile', '')), short=True)}: {_friendly_md_status(str(item.get('job_root_status', '')))}"
                    for item in campaigns
                )
        return "-"

    rows: list[dict[str, object]] = []
    for ledger in ledger_rows:
        run_label = str(ledger.get("run", ""))
        sequence = str(ledger.get("sequence", ""))
        key = (run_label, sequence)
        candidate = candidate_by_key.get(key, {})
        review = review_by_key.get(key, {})
        promotion = promotion_by_key.get(key, {})
        ladder = ladder_by_key.get(key, {})
        latest_import = import_by_key.get(key, {})
        latest_decision = latest_decision_by_key.get(key, {})
        slate_info = latest_slate_by_key.get(key, {})

        campaigns = []
        remote_job_ids = set()
        if isinstance(ladder, dict):
            for item in list(ladder.get("campaigns", [])):
                if not isinstance(item, dict):
                    continue
                campaign_name = str(item.get("campaign", "")).strip()
                if campaign_name:
                    campaigns.append(campaign_name)
                remote_job_id = str(item.get("remote_job_id", "")).strip()
                if remote_job_id:
                    remote_job_ids.add(remote_job_id)
        if isinstance(slate_info, dict):
            remote_job_id = str(slate_info.get("remote_job_id", "")).strip()
            if remote_job_id:
                remote_job_ids.add(remote_job_id)

        lifecycle_state = str(ledger.get("lifecycle_state", ""))
        feedback_state = str(review.get("feedback_state", "")) or lifecycle_state or "-"
        promotion_state = str(review.get("promotion_state", "")) or str(promotion.get("promotion_state", "")) or "-"
        source_batch_kind = (
            str(candidate.get("source_batch_kind", "")).strip()
            or str((ladder.get("source_batch_kind_label", "") if isinstance(ladder, dict) else ""))
            or str(review.get("source_batch_kind", "")).strip()
            or str(promotion.get("source_batch_kind", "")).strip()
            or "-"
        )
        current_source_batch = str(review.get("current_source_batch", "")).strip() or str(promotion.get("current_source_batch", "")).strip()
        original_source_batch = str(promotion.get("original_source_batch", "")).strip() or str(review.get("promotion_from_batch", "")).strip()
        review_label = str(review.get("current_label", "")).strip() or str(ledger.get("review_label", "")).strip()
        ingest_csv = str(review.get("ingest_csv", "")).strip() or "-"
        candidate_status = str(candidate.get("candidate_status", "")).strip() or "-"
        decision_title = str(latest_decision.get("title", "")).strip()
        decision_label = _decision_type_label(str(latest_decision.get("decision_type", "")).strip()) if latest_decision else "-"
        decision_summary = decision_title or decision_label
        integrity_flags: list[str] = []
        if candidate_status == "Selected for MD" and str(candidate.get("launch_ready", "")).strip() == "no":
            integrity_flags.append("selected for MD but missing a launch-ready source batch")
        if ingest_csv not in {"", "-"} and review_label not in {"0", "1"}:
            integrity_flags.append("ingest CSV exists without a final human label")
        if promotion_state.startswith("Promoted into real batch") and original_source_batch and current_source_batch and original_source_batch == current_source_batch:
            integrity_flags.append("promotion was recorded but the source batch still looks unchanged")
        if str(review.get("already_ingested", "")).strip() == "yes" and not latest_import:
            integrity_flags.append("review audit says already ingested, but no import row was found")
        if lifecycle_state == "MD in progress" and not (str(slate_info.get("slate_id", "")).strip() or remote_job_ids):
            integrity_flags.append("MD progress is recorded, but no slate/job trace is visible")
        rows.append(
            {
                "run": run_label,
                "sequence": sequence,
                "lifecycle_state": lifecycle_state or "-",
                "feedback_state": feedback_state,
                "origin": str(candidate.get("source", "")).strip() or str(ledger.get("suggested_via", "")).strip() or "-",
                "proposal_round": str(candidate.get("proposal_round", "")).strip() or str(ledger.get("proposal_round", "")).strip() or "-",
                "candidate_status": candidate_status,
                "source_batch_kind": source_batch_kind,
                "current_source_batch": _path_name(current_source_batch) or "-",
                "original_source_batch": _path_name(original_source_batch) or "-",
                "md_slate_id": str(slate_info.get("slate_id", "")).strip() or "-",
                "md_slate_status": str(slate_info.get("slate_status", "")).strip() or "-",
                "md_peptide_status": str(slate_info.get("peptide_status", "")).strip() or "-",
                "campaigns_seen": ", ".join(dict.fromkeys(campaigns)) if campaigns else "-",
                "md_stage_trace": _stage_trace(ladder if isinstance(ladder, dict) else None, slate_info if isinstance(slate_info, dict) else None),
                "remote_job_ids": ", ".join(sorted(remote_job_ids)) if remote_job_ids else "-",
                "review_label": review_label or "-",
                "review_notes": str(review.get("review_notes", "")).strip() or "-",
                "promotion_state": promotion_state,
                "ingest_csv": ingest_csv,
                "ingested_round": str(latest_import.get("round_id", "")).strip() or "-",
                "latest_decision": decision_summary or "-",
                "decision_rationale": str(latest_decision.get("rationale", "")).strip() or "-",
                "next_step": str(review.get("next_feedback_step", "")).strip() or str(ledger.get("next_action", "")).strip() or str(candidate.get("next_action", "")).strip() or "-",
                "integrity_state": "Attention needed" if integrity_flags else "OK",
                "integrity_flags": "; ".join(integrity_flags) if integrity_flags else "-",
            }
        )

    lifecycle_priority = {label: index for index, label in enumerate(PEPTIDE_BUCKET_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("integrity_state", "")) == "Attention needed" else 1,
            lifecycle_priority.get(str(row.get("lifecycle_state", "")), len(lifecycle_priority)),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )


def _today_queue_sections(overview: dict[str, object]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    queue = list(overview.get("today_queue", []))
    act_now_categories = {"Review", "Ingest", "AL promotion", "Remote monitoring", "MD preparation"}
    act_now = [row for row in queue if str(row.get("category", "")) in act_now_categories]
    keep_moving = [row for row in queue if str(row.get("category", "")) not in act_now_categories]
    return act_now[:8], keep_moving[:10]


def _bar_chart_rows(rows: list[dict[str, object]], *, label_key: str, value_key: str) -> dict[str, dict[object, object]]:
    chart: dict[str, dict[object, object]] = {}
    for row in rows:
        label = str(row.get(label_key, ""))
        value = _safe_float(row.get(value_key))
        if not label or value is None:
            continue
        chart.setdefault(value_key, {})[label] = value
    return chart


def _reporting_readiness_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for run in runs:
        readiness, note = _reporting_readiness_for_run(run)
        counts[readiness] = counts.get(readiness, 0) + 1
        rows.append(
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "reporting_readiness": readiness,
                "note": note,
            }
        )
    return rows


def _reporting_readiness_counts(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for run in runs:
        readiness, _ = _reporting_readiness_for_run(run)
        counts[readiness] = counts.get(readiness, 0) + 1
    return [{"readiness": key, "count": value} for key, value in sorted(counts.items())]


def _figure_caption_rows(runs: list[dict[str, object]], *, metric_name: str) -> list[dict[str, str]]:
    caption_rows: list[dict[str, str]] = []
    scorecard_rows = _result_scorecard_rows(runs, metric_name=metric_name)
    final_chart = _bar_chart_rows(scorecard_rows, label_key="run", value_key=f"final_{metric_name}")
    delta_chart = _bar_chart_rows(scorecard_rows, label_key="run", value_key=f"delta_{metric_name}")
    replay_best = _replay_best_strategy_rows(runs, metric_name=metric_name)
    replay_chart = _bar_chart_rows(replay_best, label_key="run", value_key=metric_name)
    discovery_best = _discovery_highlight_rows(runs)
    discovery_chart = _bar_chart_rows(discovery_best, label_key="run", value_key="utility_score")
    if final_chart:
        caption_rows.append(
            {
                "figure": f"Frozen final {metric_name} by run",
                "suggested_caption": f"Comparison of frozen final {metric_name} across visible thesis runs.",
            }
        )
    if delta_chart:
        caption_rows.append(
            {
                "figure": f"Improvement from baseline to final ({metric_name})",
                "suggested_caption": f"Improvement in {metric_name} from the baseline starting point to the frozen final evaluation.",
            }
        )
    if replay_chart:
        caption_rows.append(
            {
                "figure": "Best replay strategy score by run",
                "suggested_caption": f"Best replay-benchmark {metric_name} per run on the initial labeled dataset.",
            }
        )
    if discovery_chart:
        caption_rows.append(
            {
                "figure": "Discovery shortlist utility by run",
                "suggested_caption": "Best discovery shortlist utility score per run.",
            }
        )
    return caption_rows


def _selected_review_audit_rows(state: dict[str, object], selected_run_names: list[str]) -> list[dict[str, object]]:
    if not selected_run_names:
        return []
    selected = set(selected_run_names)
    return [
        row
        for row in _review_feedback_audit_rows(state)
        if str(row.get("run", "")) in selected
    ]


def _thesis_narrative_markdown(
    runs: list[dict[str, object]],
    *,
    metric_name: str,
    state: dict[str, object],
) -> str:
    if not runs:
        return (
            "## Thesis narrative draft\n"
            "- Select at least one thesis run to generate a draft comparison narrative.\n"
        )
    scorecards = _result_scorecard_rows(runs, metric_name=metric_name)
    replay_rows = _replay_best_strategy_rows(runs, metric_name=metric_name)
    discovery_rows = _discovery_highlight_rows(runs)
    review_rows = _selected_review_audit_rows(
        state,
        [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs],
    )
    readiness_counts = _reporting_readiness_counts(runs)
    best_final = max(
        scorecards,
        key=lambda row: (_safe_float(row.get(f"final_{metric_name}")) or float("-inf")),
    )
    best_delta = max(
        scorecards,
        key=lambda row: (_safe_float(row.get(f"delta_{metric_name}")) or float("-inf")),
    )
    readiness_text = ", ".join(
        f"{row['readiness']}: {row['count']}"
        for row in readiness_counts
    ) or "No reporting-readiness signal yet."
    review_ready_count = sum(1 for row in review_rows if str(row.get("feedback_state", "")) == "Ready for model ingest")
    review_pending_count = sum(1 for row in review_rows if str(row.get("feedback_state", "")) == "Needs human review")
    replay_sentence = (
        "; ".join(
            f"{row['run']} favored {row['best_strategy']} ({metric_name}={row[metric_name]})"
            for row in replay_rows
        )
        if replay_rows
        else "No replay benchmark evidence is available for the selected runs."
    )
    discovery_sentence = (
        "; ".join(
            f"{row['run']} surfaced {row['strategy']} with utility {row['utility_score']}"
            for row in discovery_rows
        )
        if discovery_rows
        else "No discovery shortlist evidence is available for the selected runs."
    )
    primary_storyline = (
        f"{best_final['run']} currently provides the strongest frozen final {metric_name} "
        f"({best_final.get(f'final_{metric_name}', '-')})."
        if _safe_float(best_final.get(f"final_{metric_name}")) is not None
        else "No selected run has a frozen final metric yet, so this comparison is still about progress rather than final reporting."
    )
    improvement_storyline = (
        f"The biggest baseline-to-final gain belongs to {best_delta['run']} "
        f"({best_delta.get(f'delta_{metric_name}', '-')})."
        if _safe_float(best_delta.get(f"delta_{metric_name}")) is not None
        else "Baseline-to-final deltas are not available yet across the selected runs."
    )
    next_focus = "; ".join(
        f"{row['run']}: {row['note']}"
        for row in scorecards
    )
    return (
        "## Thesis narrative draft\n"
        f"- Selected runs: {', '.join(str(run.get('run_display_name', run.get('run_name', ''))) for run in runs)}\n"
        f"- Main comparison claim: {primary_storyline}\n"
        f"- Improvement claim: {improvement_storyline}\n"
        f"- Reporting readiness: {readiness_text}\n"
        f"- Replay evidence: {replay_sentence}\n"
        f"- Discovery evidence: {discovery_sentence}\n"
        f"- Review / ingest handoff: {review_pending_count} peptides still need human review, and {review_ready_count} are ready for ingest.\n"
        "\n"
        "### Draft interpretation paragraph\n"
        f"The selected runs represent different stages of the thesis loop. {primary_storyline} {improvement_storyline} "
        f"Replay benchmarking suggests that {replay_sentence.lower()} Discovery evidence shows that {discovery_sentence.lower()} "
        f"This leaves the following immediate thesis focus: {next_focus}.\n"
    )


def _thesis_narrative_callout_rows(
    runs: list[dict[str, object]],
    *,
    metric_name: str,
    state: dict[str, object],
) -> list[dict[str, object]]:
    if not runs:
        return []
    selected_names = [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs]
    scorecards = _result_scorecard_rows(runs, metric_name=metric_name)
    replay_rows = _replay_best_strategy_rows(runs, metric_name=metric_name)
    discovery_rows = _discovery_highlight_rows(runs)
    review_rows = _selected_review_audit_rows(state, selected_names)
    rows: list[dict[str, object]] = []
    if scorecards:
        rows.append(
            {
                "artifact_type": "Table",
                "title": "Selected-run comparison table",
                "where_to_get_it": "Results -> Compare selected runs",
                "why_it_matters": "Shows the chosen runs side by side for final metric, improvement, readiness, and next thesis focus.",
                "draft_callout": f"Table X compares the selected runs using {metric_name}, reporting readiness, milestone completion, and next recommended step.",
            }
        )
        rows.append(
            {
                "artifact_type": "Figure",
                "title": f"Selected-run final {metric_name}",
                "where_to_get_it": "Results -> Compare selected runs",
                "why_it_matters": f"Shows which selected run currently leads on the frozen final {metric_name}.",
                "draft_callout": f"Figure X compares frozen final {metric_name} across the selected thesis runs.",
            }
        )
    if replay_rows:
        rows.append(
            {
                "artifact_type": "Figure",
                "title": "Replay benchmark comparison",
                "where_to_get_it": "Results -> Figure-ready comparisons",
                "why_it_matters": "Justifies the strategy choice on the initial dataset before real peptide feedback.",
                "draft_callout": "Figure Y summarizes the strongest replay-benchmark strategy observed for each selected run.",
            }
        )
    if discovery_rows:
        rows.append(
            {
                "artifact_type": "Table",
                "title": "Discovery shortlist evidence",
                "where_to_get_it": "Results -> Discovery evidence",
                "why_it_matters": "Documents which discovery strategy surfaced the most promising shortlist for each selected run.",
                "draft_callout": "Table Y summarizes the best discovery shortlist per selected run, including the surfaced candidate sequences.",
            }
        )
    if review_rows:
        rows.append(
            {
                "artifact_type": "Table",
                "title": "Review / feedback audit",
                "where_to_get_it": "Results -> Review / feedback audit",
                "why_it_matters": "Shows which MD review decisions are still blocking ingestion back into the model.",
                "draft_callout": "Table Z records the current human-review and model-feedback status for the selected full-analysis peptides.",
            }
        )
    return rows


def _thesis_packet_markdown(
    runs: list[dict[str, object]],
    *,
    metric_name: str,
    state: dict[str, object],
) -> str:
    best_final_run = _best_final_run(runs, metric_name=metric_name)
    readiness_counts = _reporting_readiness_counts(runs)
    readiness_text = ", ".join(
        f"{row['readiness']}: {row['count']}"
        for row in readiness_counts
    ) or "No visible runs."
    audit_rows = _review_feedback_audit_rows(state)
    audit_counts: dict[str, int] = {}
    for row in audit_rows:
        key = str(row.get("feedback_state", ""))
        audit_counts[key] = audit_counts.get(key, 0) + 1
    audit_text = ", ".join(f"{key}: {value}" for key, value in sorted(audit_counts.items())) or "No full-analysis review rows."
    best_run_name = str(best_final_run.get("run_display_name", best_final_run.get("run_name", ""))) if best_final_run else "No frozen final run yet"
    best_run_value = _format_float(_safe_float(best_final_run.get("final_metrics", {}).get(metric_name))) if best_final_run else "-"
    replay_runs = len([run for run in runs if _frame_records(replay_curve_frame(run))])
    discovery_runs = len([run for run in runs if _frame_records(discovery_frame(run))])
    return (
        "## Thesis packet summary\n"
        f"- Visible runs: {len(runs)}\n"
        f"- Best frozen final run for {metric_name}: {best_run_name} ({best_run_value})\n"
        f"- Reporting readiness breakdown: {readiness_text}\n"
        f"- Runs with replay evidence: {replay_runs}\n"
        f"- Runs with discovery evidence: {discovery_runs}\n"
        f"- Review / feedback audit: {audit_text}\n"
        "- Use the tables below as the copy/paste source for thesis comparison notes, figure captions, and human-review traceability.\n"
    )


def _appendix_packet_markdown(
    runs: list[dict[str, object]],
    *,
    metric_name: str,
    selected_runs_only: bool,
    report_ready_only: bool,
    comparison_runs: list[dict[str, object]],
    review_rows: list[dict[str, object]],
    promotion_rows: list[dict[str, object]],
    provenance_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    caption_rows: list[dict[str, object]],
) -> str:
    run_names = [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs]
    comparison_names = [str(run.get("run_display_name", run.get("run_name", ""))) for run in comparison_runs]
    readiness_counts = _reporting_readiness_counts(runs)
    readiness_text = ", ".join(
        f"{row['readiness']}: {row['count']}"
        for row in readiness_counts
    ) or "No runs in the appendix packet."
    best_final_run = _best_final_run(runs, metric_name=metric_name)
    best_run_name = str(best_final_run.get("run_display_name", best_final_run.get("run_name", ""))) if best_final_run else "No frozen final run yet"
    best_run_value = _format_float(_safe_float(best_final_run.get("final_metrics", {}).get(metric_name))) if best_final_run else "-"
    run_name_text = ", ".join(run_names) if run_names else "None"
    comparison_text = ", ".join(comparison_names) if comparison_names else "None"
    return (
        "## Thesis appendix packet\n"
        f"- Metric focus: {metric_name}\n"
        f"- Runs included ({len(runs)}): {run_name_text}\n"
        f"- Comparison focus runs ({len(comparison_runs)}): {comparison_text}\n"
        f"- Selected runs only: {'yes' if selected_runs_only else 'no'}\n"
        f"- Report-ready only: {'yes' if report_ready_only else 'no'}\n"
        f"- Best frozen final run for {metric_name}: {best_run_name} ({best_run_value})\n"
        f"- Reporting readiness breakdown: {readiness_text}\n"
        f"- Review / feedback audit rows: {len(review_rows)}\n"
        f"- Promotion audit rows: {len(promotion_rows)}\n"
        f"- Peptide provenance rows: {len(provenance_rows)}\n"
        f"- Thesis decision rows: {len(decision_rows)}\n"
        f"- Figure caption rows: {len(caption_rows)}\n"
        "- Use the appendix packet export blocks below as the copy/paste source for thesis appendix tables and reporting notes.\n"
    )


def _rows_to_csv_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    output = StringIO()
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output.getvalue()


def _rows_to_markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _best_final_run(runs: list[dict[str, object]], *, metric_name: str) -> dict[str, object] | None:
    candidates = [
        run
        for run in runs
        if isinstance(run.get("final_metrics", {}), dict) and _safe_float(run.get("final_metrics", {}).get(metric_name)) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda run: _safe_float(run.get("final_metrics", {}).get(metric_name)) or float("-inf"),
    )


def _render_export_pack(st, *, title: str, description: str, rows: list[dict[str, object]], key_prefix: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(description)
    if not rows:
        st.info("No rows are available for this export block yet.")
        return
    st.dataframe(rows)
    csv_text = _rows_to_csv_text(rows)
    markdown_text = _rows_to_markdown_table(rows)
    if csv_text:
        st.write("Copy-friendly CSV")
        st.code(csv_text, language="text")
    if markdown_text:
        st.write("Copy-friendly Markdown table")
        st.code(markdown_text, language="markdown")


def _safe_read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))



def _safe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def _iso_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")



def _round_file(directory: Path, pattern: re.Pattern[str]) -> dict[str, object] | None:
    if not directory.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    round_id, path = max(candidates, key=lambda item: item[0])
    rows = _safe_read_csv(path)
    return {
        "round_id": round_id,
        "path": str(path),
        "count": len(rows),
        "rows": rows,
    }



def _safe_float(value: str | float | int | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_decimal_text(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("."):
        return f"0{text}"
    if text.startswith("-."):
        return f"-0{text[1:]}"
    if text.startswith("+."):
        return f"+0{text[1:]}"
    return text



def _status_rank(status: str) -> int:
    order = {
        "analysis_complete": 5,
        "sasa_complete": 4,
        "dynamics_complete": 3,
        "package_prepared": 2,
        "pdb_missing": 1,
    }
    return order.get(status, 0)



def _expected_terminal_status(md_profile: str) -> str:
    if md_profile == "full":
        return "analysis_complete"
    return "dynamics_complete"



def _quote_path(path: str | Path) -> str:
    return f'"{Path(path)}"'



def _canonical_path(value: str | Path | None) -> str:
    if not value:
        return ""
    return str(Path(value).resolve())



def _path_name(value: str | Path | None) -> str:
    if not value:
        return ""
    return Path(value).name


def _source_batch_kind(run_root: Path, source_batch_csv: str | Path | None) -> str:
    if not source_batch_csv:
        return ""
    return "dashboard_generated" if is_dashboard_md_source_batch(run_root, source_batch_csv) else "run_batch"


def _source_batch_kind_label(kind: str) -> str:
    if kind == "dashboard_generated":
        return "Dashboard-local MD batch"
    if kind == "run_batch":
        return "Run proposed batch"
    return "Unknown"


def _source_batch_ingest_supported(kind: str) -> bool:
    return kind != "dashboard_generated"


def _source_batch_ingest_blocker(kind: str) -> str:
    if kind != "dashboard_generated":
        return ""
    return (
        "This peptide came from a dashboard-local MD source batch for discovery/reporting support. "
        "Use the returned MD evidence in the thesis cockpit, but do not treat it as `ingest-round` ready "
        "unless the peptide later appears in a real proposed batch."
    )


def _promotion_target_batch_csv(run_dir: str | Path | None, sequence: str) -> str:
    if not run_dir or not sequence:
        return ""
    return find_run_md_source_batch(Path(str(run_dir)), sequence)


def _promotion_round_id(batch_csv: str | Path | None, sequence: str) -> str:
    if not batch_csv:
        return ""
    try:
        row = load_md_source_batch_row(Path(str(batch_csv)), sequence)
    except Exception:
        return ""
    return str(row.get("round_id", "")).strip()



def _path_is_within(value: str | Path | None, root: str | Path | None) -> bool:
    if not value or not root:
        return False
    try:
        Path(value).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False



def _all_round_rows(directory: Path, pattern: re.Pattern[str]) -> list[dict[str, object]]:
    if not directory.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted((item for item in directory.iterdir() if item.is_file()), key=lambda item: item.name):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        round_id = int(match.group(1))
        for row in _safe_read_csv(path):
            rows.append({"round_id": round_id, "path": str(path), **row})
    return rows



def _parse_sequence_list(value: object) -> list[str]:
    if value in {None, ""}:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]



def _friendly_md_profile(profile: str, *, short: bool = False) -> str:
    info = MD_PROFILE_INFO.get(profile, {})
    if short:
        return str(info.get("short_label", profile or "-"))
    return str(info.get("label", profile or "-"))



def _friendly_md_status(status: str) -> str:
    return MD_STATUS_INFO.get(status, status or "-")



def _friendly_ml_status(status: str) -> str:
    info = ML_STATUS_INFO.get(status, {})
    return str(info.get("label", status or "-"))



def _friendly_bool(value: bool) -> str:
    return "yes" if value else "no"



def _md_profile_sort_key(profile: str) -> int:
    return {"line_smoke": 0, "production_smoke": 1, "full": 2}.get(profile, 99)



def _is_historical_candidate_run(run_summary: dict[str, object]) -> bool:
    run_slug = str(run_summary.get("run_slug", "") or Path(str(run_summary.get("run_dir", ""))).name).lower()
    configured = str(run_summary.get("run_name", "")).lower()
    searchable = f"{run_slug} {configured}"
    return any(marker in searchable for marker in HISTORICAL_RUN_MARKERS)



def _run_matches_selector(run: dict[str, object], selector: str) -> bool:
    if selector == "All":
        return True
    candidates = {
        str(run.get("run_display_name", "")),
        str(run.get("run_name", "")),
        str(run.get("run_slug", "")),
        str(run.get("run_dir_key", "")),
    }
    return selector in candidates



def _run_scope_matches(run: dict[str, object], workspace_scope: str, *, has_pins: bool) -> bool:
    if workspace_scope == "All Runs":
        return not bool(run.get("is_hidden", False))
    if workspace_scope == "Historical / Test":
        return bool(run.get("is_hidden", False)) or bool(run.get("is_historical_candidate", False))
    if bool(run.get("is_hidden", False)):
        return False
    if has_pins:
        return bool(run.get("is_pinned", False))
    if bool(run.get("is_historical_candidate", False)):
        return False
    return True



def _render_identity(run_name: str, run_slug: str) -> str:
    if not run_slug or run_slug == run_name:
        return run_name
    return f"{run_name} [{run_slug}]"


def _decision_type_label(decision_type: str) -> str:
    return str(DECISION_TYPE_INFO.get(decision_type, {}).get("label", decision_type or "-"))


def _decision_scope_options(scope: str) -> list[str]:
    if scope == "run":
        return list(RUN_DECISION_TYPES)
    if scope == "peptide":
        return list(PEPTIDE_DECISION_TYPES)
    if scope == "candidate":
        return list(CANDIDATE_DECISION_TYPES)
    return sorted(str(key) for key in DECISION_TYPE_INFO)


def _decision_target(entry: dict[str, object]) -> str:
    sequence = str(entry.get("sequence", "")).strip()
    run_name = str(entry.get("run_name", "")).strip()
    campaign_dir = str(entry.get("campaign_dir", "")).strip()
    if sequence and run_name:
        return f"{sequence} ({run_name})"
    if sequence:
        return sequence
    if run_name:
        return run_name
    if campaign_dir:
        return _path_name(campaign_dir)
    return str(entry.get("scope", "global")).title()


def _decision_log_rows(
    state: dict[str, object],
    *,
    run_names: set[str] | None = None,
    sequence: str = "",
    scope: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in list(state.get("decisions", [])):
        if not isinstance(entry, dict):
            continue
        entry_scope = str(entry.get("scope", "")).strip()
        entry_run = str(entry.get("run_name", "")).strip()
        entry_sequence = str(entry.get("sequence", "")).strip()
        if scope and entry_scope != scope:
            continue
        if run_names is not None and entry_run not in run_names:
            continue
        if sequence and entry_sequence != sequence:
            continue
        rows.append(
            {
                "logged_at": str(entry.get("created_at", "")),
                "scope": entry_scope or "-",
                "decision": _decision_type_label(str(entry.get("decision_type", ""))),
                "target": _decision_target(entry),
                "title": str(entry.get("title", "")),
                "rationale": str(entry.get("rationale", "")),
                "evidence": str(entry.get("evidence", "")),
                "next_step": str(entry.get("next_step", "")),
            }
        )
    return rows


def _render_decision_workspace(
    st,
    state: dict[str, object],
    *,
    scope: str,
    run: dict[str, object] | None = None,
    ladder: dict[str, object] | None = None,
) -> None:
    run_root = Path(str(state["run_root"]))
    run_name = ""
    run_dir = ""
    sequence = ""
    campaign_dir = ""
    context_label = ""
    if run is not None:
        run_name = str(run.get("run_display_name", run.get("run_name", "")))
        run_dir = str(run.get("run_dir", ""))
        context_label = run_name or "selected run"
    elif ladder is not None:
        run_name = str(ladder.get("run_display_name", ladder.get("run_name", "")))
        run_dir = str(ladder.get("run_dir", ""))
        sequence = str(ladder.get("sequence", ""))
        current = ladder.get("current") or ladder.get("full") or {}
        if isinstance(current, dict):
            campaign_dir = str(current.get("campaign_dir", ""))
        context_label = f"{sequence} ({run_name})" if run_name else sequence
    else:
        context_label = "current thesis workspace"

    st.subheader("Thesis decision log")
    st.caption("Use this to capture why a thesis choice was made. These notes are local-only, survive restarts, and show up later in Results and export-ready tables.")
    recent_rows = _decision_log_rows(
        state,
        run_names={run_name} if run_name else None,
        sequence=sequence,
        scope=scope,
    )
    _render_metric_cards(
        st,
        [
            ("Context", context_label),
            ("Recorded decisions here", len(recent_rows)),
        ],
    )
    if recent_rows:
        st.markdown("#### Recent decisions in this context")
        st.dataframe(recent_rows[:8])
    else:
        st.info("No thesis decision has been recorded for this context yet.")

    decision_types = _decision_scope_options(scope)
    option_labels = [_decision_type_label(item) for item in decision_types]
    option_map = dict(zip(option_labels, decision_types))
    key_prefix = f"decision_{scope}_{sequence or _path_name(run_dir) or 'global'}"
    selected_label = st.selectbox(
        "Decision type",
        option_labels,
        index=0,
        key=f"{key_prefix}_type",
    )
    decision_type = option_map[selected_label]
    spec = DECISION_TYPE_INFO.get(decision_type, {})
    title = st.text_input(
        "Short decision title",
        value=str(spec.get("default_title", "")),
        key=f"{key_prefix}_title",
    )
    rationale = st.text_area(
        "Why are we making this decision?",
        value="",
        key=f"{key_prefix}_rationale",
        height=110,
    )
    evidence = st.text_input(
        "Evidence / metric / file note",
        value="",
        key=f"{key_prefix}_evidence",
    )
    next_step = st.text_input(
        "What should happen next?",
        value=str(spec.get("default_next_step", "")),
        key=f"{key_prefix}_next",
    )
    if st.button("Save thesis decision", key=f"{key_prefix}_save"):
        if not title.strip() or not rationale.strip():
            st.warning("Add both a short title and a rationale before saving the thesis decision.")
        else:
            add_dashboard_decision(
                run_root,
                scope=scope,
                decision_type=decision_type,
                title=title,
                rationale=rationale,
                run_dir=run_dir,
                run_name=run_name,
                sequence=sequence,
                campaign_dir=campaign_dir,
                evidence=evidence,
                next_step=next_step,
            )
            st.success("Saved thesis decision to the local dashboard decision log.")
            st.rerun()


def _next_action_for_run(run: dict[str, object]) -> dict[str, str]:
    ml_status = str(run.get("ml_status", "config-only"))
    remote_sync_status = str(run.get("remote_sync_status", "not_synced"))
    remote_job_id = str(run.get("remote_job_id", ""))
    latest_import_rows = list(run.get("import_rows", []))
    latest_batch_rows = list((run.get("latest_batch") or {}).get("rows", []))
    available_ingest_csvs = list(run.get("available_ingest_csvs", []))
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}

    if remote_job_id and remote_sync_status in {"submitted", "running"}:
        return {
            "title": "Monitor the active SUPEK job",
            "summary": "A remote SUPEK workflow is already active for this run. Poll the queue and pull artifacts back when it finishes.",
            "why": "Remote execution is already in flight, so monitoring is the next safe action.",
        }
    if bool(feedback_queue.get("can_continue", False)):
        return {
            "title": "Continue AL from reviewed peptides",
            "summary": str(feedback_queue.get("summary", "")) or "The pending proposed batch is fully reviewed, so the model-feedback handoff can run now.",
            "why": "This is the cleanest way to close the full MD -> ingest -> retrain loop for the current proposed batch.",
        }
    if str(feedback_queue.get("status", "")) == "blocked" and str(feedback_queue.get("pending_round_id", "")).strip():
        return {
            "title": "Finish the MD feedback queue",
            "summary": str(feedback_queue.get("summary", "")) or "Some proposed peptides still need review, promotion, or full-analysis completion before ingest can run.",
            "why": "The next AL retraining step depends on the whole pending proposed batch being ready at once.",
        }
    if available_ingest_csvs:
        return {
            "title": "Ingest returned labels",
            "summary": "A reviewed `cgmd_ingest.csv` is available for this run, so you can feed the newest MD labels back into the model.",
            "why": "This is the step that closes the active-learning loop after MD review.",
        }
    if latest_batch_rows and not latest_import_rows:
        return {
            "title": "Validate the proposed peptides",
            "summary": "This run already proposed a peptide batch. The next meaningful step is to send those peptides through MD and eventually review / ingest the returned labels.",
            "why": "Proposing again before feedback would skip the current batch of candidates.",
        }
    if ml_status in {"config-only", "initialized"}:
        return {
            "title": "Run the replay benchmark",
            "summary": "Start by benchmarking acquisition strategies on the initial dataset only, without proposing new peptides yet.",
            "why": "This gives you a clean thesis baseline before the real active-learning loop begins.",
        }
    if ml_status == "replay-complete":
        return {
            "title": "Propose the next peptide batch",
            "summary": "The initial benchmark is already present. The next production step is usually generating the first real validation batch.",
            "why": "The run is ready to leave benchmarking mode and enter the validation loop.",
        }
    if ml_status == "discovery-complete":
        return {
            "title": "Review discovery candidates or freeze a final evaluation",
            "summary": "Discovery results exist for this run. Review them and decide whether to validate more peptides or record a frozen holdout result.",
            "why": "Discovery is exploratory, so the next step depends on which thesis question you want to answer next.",
        }
    if ml_status == "final-evaluated":
        run_dir_text = str(run.get("run_dir", "")).strip()
        freeze_json = Path(run_dir_text) / "final_freeze" / "final_freeze.json" if run_dir_text else None
        if freeze_json is None or not freeze_json.exists():
            return {
                "title": "Freeze final thesis result",
                "summary": "This run has final holdout metrics, but it has not yet been frozen into the thesis-safe manifest and model card.",
                "why": "The final freeze is the reproducibility handoff between experimentation and reporting.",
            }
        return {
            "title": "Review and export thesis results",
            "summary": "This run already has final holdout metrics, so it is ready for comparison tables, plots, and discussion.",
            "why": "The frozen evaluation already exists, so the next step is interpretation rather than more training.",
        }
    return {
        "title": "Review run state",
        "summary": "The dashboard could not infer a stronger recommendation yet. Inspect the latest batch, imports, and MD status for this run.",
        "why": "This run is in a mixed state that needs a quick human check.",
    }



def _infer_source_batch_csv(run_dir: Path, round_id: str) -> str:
    try:
        numeric_round = int(round_id)
    except (TypeError, ValueError):
        return ""
    candidate = run_dir / "batches" / f"round_{numeric_round:03d}_batch.csv"
    return str(candidate) if candidate.exists() else ""



def _load_stage_meta(campaign_dir: Path) -> dict[str, str] | None:
    meta_path = campaign_dir / STAGE_META_FILENAME
    if not meta_path.exists():
        return None
    payload = _safe_read_json(meta_path)
    return payload if isinstance(payload, dict) else None


def _replay_root_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "replay",
        run_dir.parent / "_dashboard_remote_state" / "downloads" / "supek" / run_dir.name / "replay",
    ]


def _first_existing_replay_root(run_dir: Path) -> Path | None:
    for replay_root in _replay_root_candidates(run_dir):
        if replay_root.exists() and any(path.is_dir() for path in replay_root.iterdir()):
            return replay_root
    return None



def _collect_replay_summary(run_dir: Path) -> dict[str, object]:
    strategies: list[dict[str, object]] = []
    replay_root = _first_existing_replay_root(run_dir)
    if replay_root is not None:
        for strategy_dir in sorted(path for path in replay_root.iterdir() if path.is_dir()):
            summary = _safe_read_json(strategy_dir / "summary.json")
            metrics = summary if isinstance(summary, list) else []
            latest = metrics[-1] if metrics else {}
            strategies.append(
                {
                    "strategy": strategy_dir.name,
                    "rounds": len(metrics),
                    "latest_labeled_count": latest.get("labeled_count", ""),
                    "latest_f1": _format_float(_safe_float(latest.get("f1"))),
                    "summary_path": str(strategy_dir / "summary.json"),
                }
            )
    return {
        "count": len(strategies),
        "strategies": strategies,
    }



def _collect_discovery_summary(run_dir: Path) -> dict[str, object]:
    discovery_root = run_dir / "discovery"
    aggregate_path = discovery_root / "aggregate_summary.csv"
    rows = _safe_read_csv(aggregate_path)
    return {
        "count": len(rows),
        "rows": rows,
        "aggregate_path": str(aggregate_path) if aggregate_path.exists() else "",
    }



def _collect_reference_commands(run_summary: dict[str, object]) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    latest_batch = run_summary.get("latest_batch") or {}
    batch_path = latest_batch.get("path", "")
    run_dir = run_summary["run_dir"]
    if batch_path:
        commands.append(
            {
                "label": "Prepare first guided MD stage",
                "command": (
                    "python -m active_learning_thesis prepare-md-stage "
                    f"--run-dir {_quote_path(run_dir)} "
                    f"--batch-csv {_quote_path(batch_path)} "
                    "--sequence <PEPTIDE> "
                    "--campaign <CAMPAIGN_NAME> "
                    "--md-profile line_smoke "
                    "--cluster bura"
                ),
            }
        )
    commands.append(
        {
            "label": "Check one peptide ladder",
            "command": (
                "python -m active_learning_thesis md-ladder-status "
                f"--run-dir {_quote_path(run_dir)} "
                "--sequence <PEPTIDE>"
            ),
        }
    )
    return commands



def _collect_md_campaigns(run_root: Path, run_dir: Path) -> list[dict[str, object]]:
    campaigns_root = run_dir / "md_campaigns"
    campaigns: list[dict[str, object]] = []
    if not campaigns_root.exists():
        return campaigns

    for campaign_dir in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
        meta = _load_stage_meta(campaign_dir)
        manifest_rows = _safe_read_csv(campaign_dir / "manifest.csv")
        review_rows = {row["sequence"]: row for row in _safe_read_csv(campaign_dir / "md_review.csv")}
        next_commands_path = campaign_dir / NEXT_COMMANDS_FILENAME
        next_commands = next_commands_path.read_text(encoding="utf-8") if next_commands_path.exists() else ""

        for manifest_row in manifest_rows:
            sequence = manifest_row["sequence"]
            review_row = review_rows.get(sequence, {})
            label_review_schema = review_schema_for_row(review_row)
            md_profile = (meta or {}).get("md_profile", manifest_row.get("md_profile", ""))
            round_id = manifest_row.get("round_id", "")
            package_dir = campaign_dir / "packages" / sequence
            source_batch_csv = (meta or {}).get("source_batch_csv", _infer_source_batch_csv(run_dir, round_id))
            source_batch_kind = _source_batch_kind(run_root, source_batch_csv)
            campaign = {
                "campaign": campaign_dir.name,
                "campaign_dir": str(campaign_dir),
                "run_dir": str(run_dir),
                "sequence": sequence,
                "round_id": round_id,
                "md_profile": md_profile,
                "legacy": meta is None,
                "cluster": (meta or {}).get("cluster", manifest_row.get("cluster", "")),
                "job_root_status": review_row.get("job_root_status", "package_prepared"),
                "review_notes": review_row.get("review_notes", ""),
                "cgmd_label": review_row.get("cgmd_label", ""),
                "label_review_schema": label_review_schema,
                **{field: review_row.get(field, "") for field in LABEL_REVIEW_FIELDS},
                "review_path": str(campaign_dir / "md_review.csv"),
                "manifest_path": str(campaign_dir / "manifest.csv"),
                "package_path": str(package_dir),
                "next_commands_path": str(next_commands_path) if next_commands_path.exists() else "",
                "next_commands": next_commands,
                "expected_terminal_status": (meta or {}).get(
                    "expected_terminal_status",
                    _expected_terminal_status(md_profile),
                ),
                "next_profile_on_success": (meta or {}).get("next_profile_on_success", ""),
                "source_batch_csv": source_batch_csv,
                "source_batch_kind": source_batch_kind,
                "promoted_from_source_batch_csv": (meta or {}).get("promoted_from_source_batch_csv", ""),
                "promoted_to_real_batch_at": (meta or {}).get("promoted_to_real_batch_at", ""),
                "promoted_round_id": (meta or {}).get("promoted_round_id", ""),
                "promotion_source": (meta or {}).get("promotion_source", ""),
                "selected_batch_csv": (meta or {}).get("selected_batch_csv", ""),
                "reuse_pdb_from": (meta or {}).get("reuse_pdb_from", ""),
                "exclude_nodes": (meta or {}).get("exclude_nodes", ""),
                "sasa_file": review_row.get("sasa_file", ""),
                "ap_file": review_row.get("ap_file", ""),
                "ap_contact_file": review_row.get("ap_contact_file", ""),
                "ap_contact_same_paper_formula_file": review_row.get("ap_contact_same_paper_formula_file", ""),
                "paper_path_ap_contact_file": review_row.get("paper_path_ap_contact_file", ""),
                "paper_path_ap_contact_last10ns_file": review_row.get("paper_path_ap_contact_last10ns_file", ""),
                "paper_path_ap_contact_last10ns_script": review_row.get("paper_path_ap_contact_last10ns_script", ""),
                "paper_path_ap_contact_last10ns_status_file": review_row.get("paper_path_ap_contact_last10ns_status_file", ""),
                "paper_ap_sasa_last10ns_file": review_row.get("paper_ap_sasa_last10ns_file", ""),
                "paper_ap_sasa_recompute_script": review_row.get("paper_ap_sasa_recompute_script", ""),
                "paper_ap_sasa_status_file": review_row.get("paper_ap_sasa_status_file", ""),
                "aggregate_summary_file": review_row.get("aggregate_summary_file", ""),
                "ap_5ns": review_row.get("ap_5ns", ""),
                "ap_12ns": review_row.get("ap_12ns", ""),
                "ap_25ns": review_row.get("ap_25ns", ""),
                "ap_50ns": review_row.get("ap_50ns", ""),
                "ap_100ns": review_row.get("ap_100ns", ""),
                "ap_200ns": review_row.get("ap_200ns", ""),
                "ap_contact_5ns": review_row.get("ap_contact_5ns", ""),
                "ap_contact_12ns": review_row.get("ap_contact_12ns", ""),
                "ap_contact_25ns": review_row.get("ap_contact_25ns", ""),
                "ap_contact_50ns": review_row.get("ap_contact_50ns", ""),
                "ap_contact_100ns": review_row.get("ap_contact_100ns", ""),
                "ap_contact_200ns": review_row.get("ap_contact_200ns", ""),
                "ap_contact_same_paper_formula_5ns": review_row.get("ap_contact_same_paper_formula_5ns", ""),
                "ap_contact_same_paper_formula_12ns": review_row.get("ap_contact_same_paper_formula_12ns", ""),
                "ap_contact_same_paper_formula_25ns": review_row.get("ap_contact_same_paper_formula_25ns", ""),
                "ap_contact_same_paper_formula_50ns": review_row.get("ap_contact_same_paper_formula_50ns", ""),
                "ap_contact_same_paper_formula_100ns": review_row.get("ap_contact_same_paper_formula_100ns", ""),
                "ap_contact_same_paper_formula_200ns": review_row.get("ap_contact_same_paper_formula_200ns", ""),
                "paper_path_ap_contact_5ns": review_row.get("paper_path_ap_contact_5ns", ""),
                "paper_path_ap_contact_12ns": review_row.get("paper_path_ap_contact_12ns", ""),
                "paper_path_ap_contact_25ns": review_row.get("paper_path_ap_contact_25ns", ""),
                "paper_path_ap_contact_50ns": review_row.get("paper_path_ap_contact_50ns", ""),
                "paper_path_ap_contact_100ns": review_row.get("paper_path_ap_contact_100ns", ""),
                "paper_path_ap_contact_200ns": review_row.get("paper_path_ap_contact_200ns", ""),
                "paper_path_ap_contact_last10ns_mean": review_row.get("paper_path_ap_contact_last10ns_mean", ""),
                "paper_path_ap_contact_last10ns_sd": review_row.get("paper_path_ap_contact_last10ns_sd", ""),
                "paper_path_ap_contact_last10ns_n_frames": review_row.get("paper_path_ap_contact_last10ns_n_frames", ""),
                "paper_path_ap_contact_last10ns_status": review_row.get("paper_path_ap_contact_last10ns_status", ""),
                "paper_ap_sasa_last10ns_mean": review_row.get("paper_ap_sasa_last10ns_mean", ""),
                "paper_ap_sasa_last10ns_sd": review_row.get("paper_ap_sasa_last10ns_sd", ""),
                "paper_ap_sasa_last10ns_n_frames": review_row.get("paper_ap_sasa_last10ns_n_frames", ""),
                "paper_ap_sasa_initial_sasa": review_row.get("paper_ap_sasa_initial_sasa", ""),
                "paper_ap_sasa_initial_source": review_row.get("paper_ap_sasa_initial_source", ""),
                "paper_ap_sasa_final10_mean_sasa": review_row.get("paper_ap_sasa_final10_mean_sasa", ""),
                "paper_ap_sasa_status": review_row.get("paper_ap_sasa_status", ""),
                "paper_ap_sasa_method": review_row.get("paper_ap_sasa_method", ""),
                "paper_ap_sasa_group_selection": review_row.get("paper_ap_sasa_group_selection", ""),
                "cluster_largest_fraction_200ns": review_row.get("cluster_largest_fraction_200ns", ""),
                "cluster_count_200ns": review_row.get("cluster_count_200ns", ""),
                "cluster_singleton_fraction_200ns": review_row.get("cluster_singleton_fraction_200ns", ""),
                "cluster_mean_contacts_200ns": review_row.get("cluster_mean_contacts_200ns", ""),
                "md_runtime_wall_hms": review_row.get("md_runtime_wall_hms", ""),
                "md_runtime_wall_seconds": review_row.get("md_runtime_wall_seconds", ""),
                "md_runtime_core_seconds": review_row.get("md_runtime_core_seconds", ""),
                "md_runtime_ns_per_day": review_row.get("md_runtime_ns_per_day", ""),
                "sync_status": "not_synced",
                "remote_job_id": "",
                "remote_path": "",
                "local_stage_path": "",
            }
            campaigns.append(campaign)
    return campaigns



def _ml_status(run_dir: Path) -> str:
    final_holdout = run_dir / "metrics" / "final_holdout.json"
    if final_holdout.exists():
        return "final-evaluated"
    if (run_dir / "discovery" / "aggregate_summary.csv").exists():
        return "discovery-complete"
    if _first_existing_replay_root(run_dir) is not None:
        return "replay-complete"
    if (run_dir / "batches").exists() and any(path.is_file() for path in (run_dir / "batches").iterdir()):
        return "batch-proposed"
    if (run_dir / "metrics" / "baseline_round_000.json").exists():
        return "initialized"
    return "config-only"



def summarize_run(run_root: Path, run_dir: Path) -> dict[str, object]:
    config = _safe_read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        raise FileNotFoundError(f"Missing run config: {run_dir / 'config.json'}")

    baseline_metrics = _safe_read_json(run_dir / "metrics" / "baseline_round_000.json")
    final_metrics = _safe_read_json(run_dir / "metrics" / "final_holdout.json")
    latest_batch = _round_file(run_dir / "batches", ROUND_BATCH_PATTERN)
    latest_candidates = _round_file(run_dir / "candidates", ROUND_CANDIDATE_PATTERN)
    latest_import = _round_file(run_dir / "imports", ROUND_IMPORT_PATTERN)
    import_rows = _all_round_rows(run_dir / "imports", ROUND_IMPORT_PATTERN)
    md_campaigns = _collect_md_campaigns(run_root, run_dir)
    latest_round_id = max(
        [
            item["round_id"]
            for item in (latest_batch, latest_candidates, latest_import)
            if item is not None
        ],
        default=0,
    )
    available_ingest_csvs = [
        str(path)
        for path in sorted((run_dir / "md_campaigns").glob("*/cgmd_ingest.csv"))
        if path.is_file()
    ] if (run_dir / "md_campaigns").exists() else []
    discovery_summary = _collect_discovery_summary(run_dir)
    discovery_sequences: list[dict[str, str]] = []
    for row in discovery_summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        for sequence in _parse_sequence_list(row.get("top_batch_sequences", "")):
            discovery_sequences.append(
                {
                    "sequence": sequence,
                    "strategy": str(row.get("strategy", "")),
                    "surrogate_stage": str(row.get("surrogate_stage", "")),
                }
            )

    phase3_inventory_status = {}
    inventory_path = run_root / "md_inventory" / "md_inventory.csv"
    if inventory_path.exists():
        for row in _safe_read_csv(inventory_path):
            sequence = str(row.get("sequence", "")).strip()
            if sequence:
                phase3_inventory_status[sequence] = row
    phase3_round_status = {}
    phase3_ingest_status = {}
    phase3_continuation_status = {}
    if config.get("phase") == "phase3_real_al" and config.get("branch_strategy"):
        try:
            from active_learning_thesis.phase3_real_al import (
                detect_round_status,
                summarize_phase3_branch_continuation,
                summarize_phase3_ingest_status,
            )

            phase3_continuation_status = summarize_phase3_branch_continuation(
                run_root,
                str(config.get("branch_strategy", "")),
            )
            display_round = int(phase3_continuation_status.get("next_round_number", 1) or 1)
            phase3_round_status = detect_round_status(
                run_root,
                str(config.get("branch_strategy", "")),
                display_round,
            )
            phase3_ingest_status = summarize_phase3_ingest_status(
                run_root,
                str(config.get("branch_strategy", "")),
                display_round,
            )
        except Exception as exc:  # pragma: no cover - dashboard should keep rendering
            phase3_round_status = {"status": "unknown", "error": str(exc)}
            phase3_ingest_status = {"status": "unknown", "error": str(exc)}
            phase3_continuation_status = {"status": "unknown", "error": str(exc)}

    summary = {
        "run_name": config.get("run_name", run_dir.name),
        "configured_run_name": config.get("run_name", run_dir.name),
        "run_slug": run_dir.name,
        "run_dir": str(run_dir),
        "run_dir_key": _canonical_path(run_dir),
        "phase": config.get("phase", ""),
        "branch_strategy": config.get("branch_strategy", ""),
        "last_modified": _iso_timestamp(run_dir),
        "ml_status": _ml_status(run_dir),
        "latest_round_id": latest_round_id,
        "baseline_metrics": baseline_metrics if isinstance(baseline_metrics, dict) else {},
        "final_metrics": final_metrics if isinstance(final_metrics, dict) else {},
        "latest_batch": latest_batch or {},
        "latest_candidates": latest_candidates or {},
        "latest_import": latest_import or {},
        "import_rows": import_rows,
        "discovery": discovery_summary,
        "discovery_sequences": discovery_sequences,
        "replay": _collect_replay_summary(run_dir),
        "md_campaigns": md_campaigns,
        "config": config,
        "remote_sync_status": "not_synced",
        "remote_job_id": "",
        "remote_path": "",
        "local_stage_path": "",
        "available_ingest_csvs": available_ingest_csvs,
        "phase3_inventory_status": phase3_inventory_status,
        "phase3_round_status": phase3_round_status,
        "phase3_ingest_status": phase3_ingest_status,
        "phase3_continuation_status": phase3_continuation_status,
        "feedback_queue": build_feedback_queue(run_dir),
        "is_historical_candidate": False,
        "is_pinned": False,
        "is_hidden": False,
        "user_label": "",
        "run_display_name": (
            f"Phase 3 branch: {config.get('branch_strategy')}"
            if config.get("phase") == "phase3_real_al" and config.get("branch_strategy")
            else run_dir.name
        ),
    }
    summary["local_commands"] = _collect_reference_commands(summary)
    summary["recommended_next_step"] = _next_action_for_run(summary)
    return summary



def _sync_record_for_target(sync_records: list[dict[str, object]], *, cluster: str, target: str) -> dict[str, object] | None:
    target_key = _canonical_path(target)
    for record in sync_records:
        if str(record.get("cluster", "")) != cluster:
            continue
        candidates = [
            _canonical_path(str(record.get("related_campaign", ""))),
            _canonical_path(str(record.get("related_run", ""))),
            _canonical_path(str(record.get("target_key", ""))),
        ]
        if target_key and target_key in {candidate for candidate in candidates if candidate}:
            return record
    return None



def _attach_remote_state(run_summaries: list[dict[str, object]], sync_records: list[dict[str, object]]) -> None:
    for run in run_summaries:
        supek_record = _sync_record_for_target(sync_records, cluster="supek", target=str(run["run_dir"]))
        if supek_record:
            run["remote_sync_status"] = str(supek_record.get("status", "not_synced"))
            run["remote_job_id"] = str(supek_record.get("remote_job_id", ""))
            run["remote_path"] = str(supek_record.get("remote_path", ""))
            run_metadata = supek_record.get("metadata", {}) if isinstance(supek_record.get("metadata"), dict) else {}
            run["local_stage_path"] = str(run_metadata.get("local_stage_path", ""))
            run["remote_stdout"] = str(run_metadata.get("remote_stdout", ""))
            run["remote_stderr"] = str(run_metadata.get("remote_stderr", ""))
            run["remote_wrapper"] = str(run_metadata.get("remote_wrapper", ""))
        run["recommended_next_step"] = _next_action_for_run(run)
        for campaign in run["md_campaigns"]:
            bura_record = _sync_record_for_target(sync_records, cluster="bura", target=str(campaign["campaign_dir"]))
            if bura_record:
                campaign["sync_status"] = str(bura_record.get("status", "not_synced"))
                campaign["remote_job_id"] = str(bura_record.get("remote_job_id", ""))
                campaign["remote_path"] = str(bura_record.get("remote_path", ""))
                campaign_metadata = bura_record.get("metadata", {}) if isinstance(bura_record.get("metadata"), dict) else {}
                campaign["local_stage_path"] = str(campaign_metadata.get("local_stage_path", ""))
                campaign["remote_job_ids"] = list(campaign_metadata.get("remote_job_ids", [])) if isinstance(campaign_metadata.get("remote_job_ids", []), list) else []



def _decorate_runs_with_curation(run_summaries: list[dict[str, object]], curation: dict[str, object]) -> None:
    duplicate_counts: dict[str, int] = {}
    for run in run_summaries:
        duplicate_counts[str(run.get("run_name", ""))] = duplicate_counts.get(str(run.get("run_name", "")), 0) + 1

    pinned = {str(item) for item in curation.get("pinned_runs", [])}
    hidden = {str(item) for item in curation.get("hidden_runs", [])}
    labels = dict(curation.get("labels", {})) if isinstance(curation.get("labels"), dict) else {}

    for run in run_summaries:
        run_key = str(run.get("run_dir_key", ""))
        run_slug = str(run.get("run_slug", ""))
        configured_name = str(run.get("run_name", ""))
        user_label = str(labels.get(run_key, ""))
        primary_name = user_label or configured_name
        display_name = _render_identity(primary_name, run_slug)
        if duplicate_counts.get(configured_name, 0) <= 1 and not user_label:
            display_name = _render_identity(configured_name, run_slug)
        if str(run.get("phase", "")) == "phase3_real_al" and str(run.get("branch_strategy", "")).strip():
            display_name = f"Phase 3 branch: {run.get('branch_strategy', '')}"
        run["is_pinned"] = run_key in pinned
        run["is_hidden"] = run_key in hidden
        run["is_historical_candidate"] = _is_historical_candidate_run(run)
        run["user_label"] = user_label
        run["run_display_name"] = display_name
        run["run_identity"] = f"Configured name: {configured_name} | Folder: {run_slug}"
        run["ml_status_label"] = _friendly_ml_status(str(run.get("ml_status", "")))
        run["ml_status_summary"] = str(ML_STATUS_INFO.get(str(run.get("ml_status", "")), {}).get("summary", ""))
        run["recommended_next_step"] = _next_action_for_run(run)
        for campaign in run.get("md_campaigns", []):
            campaign["run_display_name"] = run["run_display_name"]



def _campaign_sort_key(item: dict[str, object]) -> tuple[int, int, int, str]:
    status_rank = _status_rank(str(item["job_root_status"]))
    sync_rank = {
        "finalized_local": 7,
        "outputs_returned": 6,
        "outputs_staged": 5,
        "running": 4,
        "submitted": 3,
        "staged_remote": 2,
        "not_synced": 1,
    }.get(str(item.get("sync_status", "not_synced")), 0)
    legacy_rank = 0 if not item["legacy"] else -1
    return status_rank, sync_rank, legacy_rank, str(item["campaign"])



def _next_action_for_ladder(ladder: dict[str, object]) -> dict[str, str]:
    if ladder["ready_for_review"]:
        current = ladder.get("full")
        if current and current.get("cgmd_label") in {"0", "1"}:
            return {
                "title": "Create the ingest CSV",
                "summary": "The full analysis run has already been reviewed and labeled, so you can now convert it into `cgmd_ingest.csv`.",
                "why": "That CSV is the bridge back into the active-learning model update step.",
            }
        return {
            "title": "Review the full-analysis result",
            "summary": "The full MD analysis has finished and is ready for a human review plus `cgmd_label` assignment.",
            "why": "A reviewed label is required before the peptide can be fed back into the model.",
        }

    full_item = ladder.get("full")
    if full_item and full_item.get("job_root_status") == "sasa_complete":
        return {
            "title": "Complete the full analysis rerun",
            "summary": "The full analysis stopped after SASA extraction, so it is still not ingest-ready.",
            "why": "A full-analysis peptide must reach `analysis_complete` before review and ingest.",
        }

    current = ladder.get("current")
    if current:
        sync_status = str(current.get("sync_status", "not_synced"))
        if sync_status == "not_synced":
            return {
                "title": "Upload the campaign to BURA",
                "summary": "This guided MD campaign exists locally but has not been staged on BURA yet.",
                "why": "Remote staging is the prerequisite for normalization, preflight, and submit.",
            }
        if sync_status == "staged_remote":
            return {
                "title": "Normalize, preflight, then submit",
                "summary": "The campaign is already on BURA. The next remote steps are script normalization, preflight, and then chain submission.",
                "why": "Those steps validate the package before real queue time is consumed.",
            }
        if sync_status in {"submitted", "running"}:
            return {
                "title": "Monitor the active BURA campaign",
                "summary": "The remote MD chain is active on BURA. Poll the queue and pull outputs back when the run is done.",
                "why": "There is already a tracked remote job, so the next safe action is monitoring.",
            }
        if sync_status == "outputs_staged":
            return {
                "title": "Re-parse the staged outputs locally",
                "summary": "The remote outputs have been copied into the dashboard staging area and now need a local finalize / parse step.",
                "why": "This updates the review CSV and unlocks the next ladder decision.",
            }
        if sync_status == "outputs_returned":
            return {
                "title": "Finalize the copied-back outputs",
                "summary": "The remote outputs are back in the campaign directory, so the next step is finalizing the local guided stage.",
                "why": "Finalize updates the review status and computes the next ladder recommendation.",
            }

    next_profile = ladder.get("next_profile", "")
    if next_profile:
        return {
            "title": f"Prepare {_friendly_md_profile(next_profile, short=True)}",
            "summary": f"The next missing ladder rung is {_friendly_md_profile(next_profile)}. Prepare that campaign locally before any BURA action.",
            "why": "Guided MD progresses in sequence from quick package check to short dynamics validation to full analysis.",
        }
    return {
        "title": "Review ladder state",
        "summary": "The dashboard could not infer a stronger next step for this peptide yet.",
        "why": "The ladder is in a mixed state that needs a quick manual check.",
    }



def _prepare_stage_command(*, ladder: dict[str, object], next_profile: str) -> str:
    source_batch_csv = ladder.get("source_batch_csv", "")
    run_dir = ladder.get("run_dir", "")
    if not source_batch_csv or not run_dir:
        return ""

    command = (
        "python -m active_learning_thesis prepare-md-stage "
        f"--run-dir {_quote_path(run_dir)} "
        f"--batch-csv {_quote_path(source_batch_csv)} "
        f"--sequence {ladder['sequence']} "
        "--campaign <CAMPAIGN_NAME> "
        f"--md-profile {next_profile} "
        f"--cluster {ladder.get('cluster', 'bura') or 'bura'}"
    )

    reuse_pdb_from = ladder.get("reuse_pdb_from", "")
    if reuse_pdb_from:
        command += f" --reuse-pdb-from {_quote_path(reuse_pdb_from)}"
    exclude_nodes = ladder.get("exclude_nodes", "")
    if exclude_nodes:
        command += f" --exclude-nodes {exclude_nodes}"
    return command



def _make_ingest_command(campaign_dir: str, review_path: str) -> str:
    return (
        "python -m active_learning_thesis make-md-ingest-csv "
        f"--campaign-dir {_quote_path(campaign_dir)} "
        f"--review-csv {_quote_path(review_path)}"
    )


def _ingest_csv_path(campaign_dir: str | Path) -> Path:
    return Path(campaign_dir) / "cgmd_ingest.csv"



def _cluster_profile_warning(state: dict[str, object], cluster_name: str) -> str:
    profile_path = default_cluster_profile_path()
    for row in state.get("profile_rows", []):
        if str(row.get("cluster", "")) != cluster_name:
            continue
        if str(row.get("configured", "no")) != "yes":
            return f"{cluster_name.title()} profile is not configured. Add it at `{profile_path}` and refresh the dashboard."
        if str(row.get("enabled", "no")) != "yes":
            return f"{cluster_name.title()} profile is disabled in `{profile_path}`. Enable it and refresh the dashboard."
        missing = str(row.get("missing_fields", "")).strip()
        if missing:
            return f"{cluster_name.title()} profile is incomplete: {missing}. Update `{profile_path}` and refresh the dashboard."
    return f"{cluster_name.title()} profile is unavailable. Check `{profile_path}` and refresh the dashboard."



def _default_health_entry(cluster_name: str) -> dict[str, object]:
    cluster_label = cluster_name.upper()
    return {
        "cluster": cluster_name,
        "checked_at": "",
        "overall_status": "unknown",
        "local_auth_status": {},
        "remote_status": {},
        "summary": f"{cluster_label} health has not been checked yet.",
        "hint": f"Use Check {cluster_label} health in Operations before relying on remote actions.",
        "details": [],
    }



def _cluster_health_entry(state: dict[str, object], cluster_name: str) -> dict[str, object]:
    for item in state.get("cluster_health", []):
        if str(item.get("cluster", "")) == cluster_name:
            return item
    return _default_health_entry(cluster_name)



def _cluster_health_rows(state: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cluster_name in SUPPORTED_CLUSTERS:
        health = _cluster_health_entry(state, cluster_name)
        local_auth = health.get("local_auth_status", {}) if isinstance(health.get("local_auth_status", {}), dict) else {}
        remote_status = health.get("remote_status", {}) if isinstance(health.get("remote_status", {}), dict) else {}
        rows.append(
            {
                "cluster": cluster_name,
                "status": str(health.get("overall_status", "unknown")),
                "checked_at": str(health.get("checked_at", "")),
                "local_auth": str(local_auth.get("status", "n/a")) if cluster_name == "supek" else "n/a",
                "remote_status": str(remote_status.get("status", "unknown")),
                "summary": str(health.get("summary", "")),
                "hint": str(health.get("hint", "")),
            }
        )
    return rows


def _latest_action_by_kind(actions: list[dict[str, object]], kind: str) -> dict[str, object] | None:
    for action in actions:
        if str(action.get("kind", "")) == kind:
            return action
    return None


def _preflight_expectations(kind: str) -> list[dict[str, str]]:
    if kind == "supek-submit-preflight":
        return [
            {"ok": "repo_ok", "missing": "repo_missing", "check": "Remote repo path", "missing_detail": "repo path is missing on SUPEK"},
            {"ok": "repo_git_ok", "missing": "repo_git_missing", "check": "Git checkout metadata", "missing_detail": "repo exists, but `.git` is missing"},
            {"ok": "conda_init_ok", "missing": "conda_init_missing", "check": "Conda init script", "missing_detail": "conda init path is missing"},
            {"ok": "scheduler_cmd_ok", "missing": "scheduler_cmd_missing", "check": "PBS submit command", "missing_detail": "`qsub` is not available in the remote shell"},
            {"ok": "scratch_root_ok", "missing": "scratch_root_missing", "check": "Scratch root", "missing_detail": "scratch root is missing on SUPEK"},
            {"ok": "run_state_staged", "missing": "run_state_missing", "check": "Staged run directory", "missing_detail": "run state is not staged on SUPEK yet"},
            {"ok": "log_root_ok", "missing": "log_root_missing", "check": "Log root", "missing_detail": "log root is missing on SUPEK"},
            {"ok": "python_import_ok", "missing": "python_import_missing", "check": "Activated env import", "missing_detail": "the configured environment could not import `active_learning_thesis`"},
        ]
    if kind == "bura-submit-readiness":
        return [
            {"ok": "campaign_dir_ok", "missing": "campaign_dir_missing", "check": "Remote campaign directory", "missing_detail": "remote campaign directory is missing"},
            {"ok": "preflight_script_ok", "missing": "preflight_script_missing", "check": "preflight_bura.sh", "missing_detail": "preflight script is missing"},
            {"ok": "submit_script_ok", "missing": "submit_script_missing", "check": "submit_chain.sh", "missing_detail": "submit script is missing"},
            {"ok": "preflight_syntax_ok", "missing": "preflight_syntax_missing", "check": "preflight script syntax", "missing_detail": "preflight script failed `bash -n`"},
            {"ok": "submit_syntax_ok", "missing": "submit_syntax_missing", "check": "submit script syntax", "missing_detail": "submit script failed `bash -n`"},
            {"ok": "package_dir_ok", "missing": "package_dir_missing", "check": "Package directory", "missing_detail": "package directory is missing for this peptide"},
            {"ok": "scheduler_cmd_ok", "missing": "scheduler_cmd_missing", "check": "Slurm commands", "missing_detail": "`sbatch` or `squeue` is not available in the remote shell"},
            {"ok": "dos2unix_ok", "missing": "dos2unix_missing", "check": "dos2unix helper", "missing_detail": "`dos2unix` is not available for script normalization"},
            {"ok": "module_load_ok", "missing": "module_load_missing", "check": "Module load command", "missing_detail": "module load command failed"},
        ]
    return []


def _preflight_status_rows(action: dict[str, object]) -> list[dict[str, str]]:
    kind = str(action.get("kind", ""))
    expectations = _preflight_expectations(kind)
    if not expectations:
        return []
    stdout_text = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=60).strip()
    stderr_text = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=40).strip()
    rows: list[dict[str, str]] = []
    for item in expectations:
        ok_token = str(item.get("ok", ""))
        missing_token = str(item.get("missing", ""))
        if ok_token and ok_token in stdout_text:
            status = "ok"
            detail = "check passed"
        elif missing_token and missing_token in stdout_text:
            status = "missing"
            detail = str(item.get("missing_detail", "")).strip() or "missing dependency detected"
        else:
            status = "unknown"
            detail = "not confirmed by the latest readiness output"
        rows.append({"check": str(item.get("check", "")), "status": status, "detail": detail})
    if stderr_text:
        rows.append({"check": "stderr excerpt", "status": "attention", "detail": stderr_text.splitlines()[-1]})
    return rows


def _render_latest_preflight_summary(st, *, title: str, actions: list[dict[str, object]], kind: str) -> None:
    action = _latest_action_by_kind(actions, kind)
    if action is None:
        return
    st.markdown(f"#### {title}")
    _render_metric_cards(
        st,
        [
            ("Last check", str(action.get("status", "")) or "-"),
            ("Checked at", str(action.get("finished_at", "")) or str(action.get("started_at", "")) or "-"),
            ("Exit", str(action.get("exit_code", "")) if action.get("exit_code") is not None else "-"),
        ],
    )
    rows = _preflight_status_rows(action)
    if rows:
        st.dataframe(rows)
    parsed_issue = _remote_console_issue_summary(
        action,
        read_log_excerpt(str(action.get("stdout_log", "")), max_lines=40),
        read_log_excerpt(str(action.get("stderr_log", "")), max_lines=40),
    )
    if parsed_issue is not None and str(parsed_issue.get("next_step", "")).strip():
        st.caption(f"Readiness guidance: {parsed_issue.get('next_step', '')}")


def _notification_severity_rank(severity: str) -> int:
    return {"error": 0, "warning": 1, "info": 2}.get(severity, 3)


STALE_WORK_THRESHOLDS_HOURS = {
    "approval-pending": 24,
    "review-ready": 48,
    "ingest-ready": 24,
    "md-slate-paused": 24,
    "md-slate-blocked": 24,
    "md-slate-failed": 24,
    "artifact-verification": 24,
    "feedback-ready": 24,
}


def _parse_dashboard_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _notification_age_hours(created_at: str, *, now: datetime) -> int | None:
    parsed = _parse_dashboard_timestamp(created_at)
    if parsed is None:
        return None
    return max(int((now - parsed).total_seconds() // 3600), 0)


def _format_stale_age(hours: int) -> str:
    if hours >= 48:
        days = hours // 24
        return f"{days} days"
    if hours >= 24:
        return "1 day"
    return f"{hours} hours"


def _stale_threshold_hours(notification: dict[str, str]) -> int | None:
    notification_id = str(notification.get("id", "")).strip()
    for prefix, threshold in STALE_WORK_THRESHOLDS_HOURS.items():
        if notification_id.startswith(prefix + ":"):
            return threshold
    return None


def _build_stale_work_notifications(
    raw_notifications: list[dict[str, str]],
    *,
    delivered: dict[str, str],
    acknowledged_ids: set[str],
    now_dt: datetime,
) -> list[dict[str, str]]:
    reminders: list[dict[str, str]] = []
    for item in raw_notifications:
        notification_id = str(item.get("id", "")).strip()
        if not notification_id or notification_id in acknowledged_ids:
            continue
        threshold = _stale_threshold_hours(item)
        if threshold is None:
            continue
        created_at = str(delivered.get(notification_id, "")).strip()
        if not created_at:
            continue
        age_hours = _notification_age_hours(created_at, now=now_dt)
        if age_hours is None or age_hours < threshold:
            continue
        target = str(item.get("target", "")).strip() or "Open handoff"
        severity = "error" if age_hours >= threshold * 3 else "warning"
        reminders.append(
            {
                "id": f"stale-work:{notification_id}",
                "severity": severity,
                "area": "Stale work",
                "target": target,
                "alert": f"{target} has been waiting for {_format_stale_age(age_hours)}. {str(item.get('alert', '')).strip()}",
                "next_move": str(item.get("next_move", "")).strip() or "Open the relevant dashboard page and either finish, recover, or acknowledge this handoff.",
                "open_view": str(item.get("open_view", "")).strip() or "Today",
                "run_dir": str(item.get("run_dir", "")),
                "sequence": str(item.get("sequence", "")),
            }
        )
    return reminders


def _build_dashboard_notifications(
    run_root: Path,
    *,
    actions: list[dict[str, object]],
    runs: list[dict[str, object]],
    peptide_ladders: list[dict[str, object]],
    md_slates: list[dict[str, object]],
    md_slate_exceptions: list[dict[str, object]],
    remote_reconciliation: list[dict[str, object]],
    artifact_verification: list[dict[str, object]],
    peptide_inventory: dict[str, object],
    overview: dict[str, object],
    cluster_health_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    store = load_dashboard_notifications(run_root)
    delivered = dict(store.get("delivered", {})) if isinstance(store.get("delivered", {}), dict) else {}
    acknowledged_ids = set(str(item) for item in store.get("acknowledged_ids", []) if str(item).strip())
    previous_cluster_status = dict(store.get("cluster_status", {})) if isinstance(store.get("cluster_status", {}), dict) else {}
    raw_notifications: list[dict[str, str]] = []
    run_dir_by_name = {
        str(run.get("run_display_name", run.get("run_name", ""))): str(run.get("run_dir", ""))
        for run in runs
    }

    for action in actions:
        if str(action.get("status", "")) not in APPROVAL_PENDING_STATUSES:
            continue
        action_id = str(action.get("id", "")).strip()
        if not action_id:
            continue
        related_run = str(action.get("related_run", "")).strip()
        raw_notifications.append(
            {
                "id": f"approval-pending:{action_id}",
                "severity": "info",
                "area": "Approval queue",
                "target": str(action.get("title", "")).strip() or action_id,
                "alert": "A dashboard action is waiting for approval before it can run.",
                "next_move": "Open Operations -> Approval queue and approve, pause, or cancel the draft.",
                "open_view": "Operations",
                "run_dir": related_run,
                "sequence": str(action.get("related_sequence", "")),
            }
        )

    for row in cluster_health_rows:
        cluster_name = str(row.get("cluster", "")).strip()
        status = str(row.get("status", "")).strip() or "unknown"
        if not cluster_name or status == "ok":
            continue
        previous_status = str(previous_cluster_status.get(cluster_name, "")).strip()
        alert_text = (
            f"{cluster_name.upper()} health turned unhealthy."
            if previous_status == "ok"
            else (str(row.get("summary", "")).strip() or f"{cluster_name.upper()} health needs attention.")
        )
        raw_notifications.append(
            {
                "id": f"cluster:{cluster_name}:{status}",
                "severity": "warning" if status != "error" else "error",
                "area": "Cluster health",
                "target": cluster_name.upper(),
                "alert": alert_text,
                "next_move": str(row.get("hint", "")).strip() or f"Re-check {cluster_name.upper()} health before relying on remote actions.",
                "open_view": "Operations",
                "run_dir": "",
                "sequence": "",
            }
        )

    for row in list(overview.get("waiting_on_review", [])):
        review_state = str(row.get("review_state", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        run_name = str(row.get("run", "")).strip()
        campaign = str(row.get("campaign", "")).strip()
        if review_state != "Needs review / label":
            continue
        raw_notifications.append(
            {
                "id": f"review-ready:{run_name}:{sequence}:{campaign}",
                "severity": "warning",
                "area": "Review ready",
                "target": sequence or campaign or run_name,
                "alert": f"{sequence} is ready for human review after full MD analysis.",
                "next_move": str(row.get("next_action", "")).strip() or "Open MD Validation and decide the `cgmd_label`.",
                "open_view": "MD Validation",
                "run_dir": run_dir_by_name.get(run_name, ""),
                "sequence": sequence,
            }
        )

    for row in list(peptide_inventory.get("ready_for_ingest", [])) if isinstance(peptide_inventory.get("ready_for_ingest", []), list) else []:
        sequence = str(row.get("sequence", "")).strip()
        run_name = str(row.get("run", "")).strip()
        campaign = str(row.get("campaign", "")).strip()
        raw_notifications.append(
            {
                "id": f"ingest-ready:{run_name}:{sequence}:{campaign}",
                "severity": "info",
                "area": "Ingest ready",
                "target": sequence or campaign or run_name,
                "alert": f"{sequence} already has a reviewed label and can move back into the model.",
                "next_move": str(row.get("next_action", "")).strip() or "Create the ingest CSV, then run Ingest returned labels.",
                "open_view": "MD Validation",
                "run_dir": run_dir_by_name.get(run_name, ""),
                "sequence": sequence,
            }
        )

    for slate in md_slates:
        slate_status = str(slate.get("effective_status", "")).strip()
        execution_mode = str(slate.get("execution_mode", "live")).strip() or "live"
        is_rehearsal = execution_mode == "rehearsal"
        slate_target = f"{_path_name(str(slate.get('run_dir', '')))} [{str(slate.get('slate_id', ''))}]"
        if slate_status == "paused":
            raw_notifications.append(
                {
                    "id": f"md-slate-paused:{str(slate.get('slate_id', ''))}",
                    "severity": "info",
                    "area": "MD rehearsal" if is_rehearsal else "MD slate",
                    "target": slate_target,
                    "alert": (
                        "The MD rehearsal is paused and will not schedule new local simulation steps until resumed."
                        if is_rehearsal
                        else "The MD slate is paused and will not schedule new child actions until resumed."
                    ),
                    "next_move": "Open MD Validation and resume the slate when you want the ladder to continue.",
                    "open_view": "MD Validation",
                    "run_dir": str(slate.get("run_dir", "")),
                    "sequence": "",
                }
            )
        if slate_status == "completed_with_failures":
            raw_notifications.append(
                {
                    "id": f"md-slate-failed:{str(slate.get('slate_id', ''))}",
                    "severity": "warning",
                    "area": "MD rehearsal" if is_rehearsal else "MD slate",
                    "target": slate_target,
                    "alert": (
                        "The MD rehearsal completed with one or more simulated blocked peptides."
                        if is_rehearsal
                        else "The MD slate completed with one or more blocked peptides."
                    ),
                    "next_move": (
                        "Open MD Validation -> Slate monitor to inspect the simulated failure path."
                        if is_rehearsal
                        else "Open MD Validation and retry or skip the blocked peptides."
                    ),
                    "open_view": "MD Validation",
                    "run_dir": str(slate.get("run_dir", "")),
                    "sequence": "",
                }
            )
        for peptide in list(slate.get("peptides", [])):
            if not isinstance(peptide, dict):
                continue
            sequence = str(peptide.get("sequence", "")).strip()
            if str(peptide.get("status", "")) == "blocked":
                raw_notifications.append(
                    {
                        "id": f"md-slate-blocked:{str(slate.get('slate_id', ''))}:{sequence}",
                        "severity": "warning",
                        "area": "MD rehearsal" if is_rehearsal else "MD slate",
                        "target": sequence or slate_target,
                        "alert": f"{sequence} is blocked inside the MD rehearsal." if is_rehearsal else f"{sequence} is blocked inside the MD slate.",
                        "next_move": str(peptide.get("failure_reason", "")).strip() or "Open MD Validation to inspect the blocked stage and decide whether to retry or skip it.",
                        "open_view": "MD Validation",
                        "run_dir": str(slate.get("run_dir", "")),
                        "sequence": sequence,
                    }
                )
            if bool(peptide.get("review_ready", False)):
                raw_notifications.append(
                    {
                        "id": f"md-slate-review-ready:{execution_mode}:{str(slate.get('slate_id', ''))}:{sequence}",
                        "severity": "info",
                        "area": "MD rehearsal" if is_rehearsal else "Review ready",
                        "target": sequence or slate_target,
                        "alert": (
                            f"{sequence} completed the rehearsal ladder only; no real MD result was produced."
                            if is_rehearsal
                            else f"{sequence} completed the full MD ladder and is ready for manual review."
                        ),
                        "next_move": (
                            "Open MD Validation -> Slate monitor to inspect the dry-run trace, then launch real MD only if the rehearsal looks healthy."
                            if is_rehearsal
                            else "Open MD Validation -> Review & ingest and assign the human `cgmd_label`."
                        ),
                        "open_view": "MD Validation",
                        "run_dir": str(slate.get("run_dir", "")),
                        "sequence": sequence,
                    }
                )

    for row in md_slate_exceptions:
        exception_type = str(row.get("exception_type", "")).strip()
        if exception_type in {"", "blocked"}:
            continue
        severity = str(row.get("severity", "")).strip() or "warning"
        raw_notifications.append(
            {
                "id": f"md-slate-exception:{str(row.get('slate_id', ''))}:{str(row.get('sequence', ''))}:{exception_type}",
                "severity": severity,
                "area": "MD recovery",
                "target": str(row.get("sequence", "")) or str(row.get("run", "")),
                "alert": str(row.get("summary", "")).strip() or "An MD slate peptide needs recovery attention.",
                "next_move": str(row.get("next_move", "")).strip() or "Open MD Validation and use the Recovery center.",
                "open_view": "MD Validation",
                "run_dir": str(row.get("run_dir", "")),
                "sequence": str(row.get("sequence", "")),
            }
        )

    for row in remote_reconciliation:
        severity = str(row.get("severity", "")).strip()
        if severity not in {"warning", "error"}:
            continue
        issue_type = str(row.get("issue_type", "")).strip()
        target = str(row.get("sequence", "")) if str(row.get("sequence", "")) not in {"", "-"} else str(row.get("run", ""))
        raw_notifications.append(
            {
                "id": (
                    "remote-reconciliation:"
                    f"{str(row.get('cluster', ''))}:"
                    f"{issue_type}:"
                    f"{str(row.get('remote_job_id', ''))}:"
                    f"{_canonical_path(str(row.get('run_dir', '')))}:"
                    f"{str(row.get('sequence', ''))}:"
                    f"{str(row.get('stage', ''))}"
                ),
                "severity": severity,
                "area": "Remote reconciliation",
                "target": target or str(row.get("remote_job_id", "")),
                "alert": str(row.get("summary", "")).strip() or "A tracked remote job disagrees with the latest queue snapshot.",
                "next_move": str(row.get("next_move", "")).strip() or "Open Operations -> Remote jobs and reconcile the job state.",
                "open_view": str(row.get("open_view", "")).strip() or "Operations",
                "run_dir": str(row.get("run_dir", "")),
                "sequence": "" if str(row.get("sequence", "")) == "-" else str(row.get("sequence", "")),
            }
        )

    for row in artifact_verification:
        if str(row.get("verification_state", "")) != "Attention needed":
            continue
        open_view = "MD Validation" if str(row.get("scope", "")) == "campaign" else "Operations"
        raw_notifications.append(
            {
                "id": f"artifact-verification:{_canonical_path(str(row.get('run_dir', '')))}:{str(row.get('sequence', ''))}:{str(row.get('campaign', ''))}:{str(row.get('sync_state', ''))}",
                "severity": str(row.get("severity", "")) or "warning",
                "area": "Artifact verification",
                "target": str(row.get("target", "")) or str(row.get("run", "")),
                "alert": str(row.get("summary", "")) or "Expected artifacts are missing or inconsistent.",
                "next_move": str(row.get("next_move", "")) or "Open the artifact verification workspace and fix the missing files.",
                "open_view": open_view,
                "run_dir": str(row.get("run_dir", "")),
                "sequence": str(row.get("sequence", "")) if str(row.get("sequence", "")) not in {"", "-"} else "",
            }
        )

    for run in runs:
        feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
        if not bool(feedback_queue.get("can_continue", False)):
            continue
        raw_notifications.append(
            {
                "id": f"feedback-ready:{_canonical_path(run.get('run_dir', ''))}:{str(feedback_queue.get('pending_round_id', ''))}",
                "severity": "info",
                "area": "Model feedback",
                "target": str(run.get("run_display_name", run.get("run_name", ""))),
                "alert": str(feedback_queue.get("summary", "")) or "The pending proposed batch is ready to continue back into active learning.",
                "next_move": "Open Model Workflow and run Continue AL from reviewed peptides.",
                "open_view": "Model Workflow",
                "run_dir": str(run.get("run_dir", "")),
                "sequence": "",
            }
        )

    finished_run_status = {
        "outputs_staged": "SUPEK finished the remote job and staged outputs for safe download.",
        "outputs_returned": "SUPEK outputs were copied back and are ready for local inspection.",
        "finalized_local": "SUPEK outputs were finalized locally and the run is ready for the next thesis decision.",
    }
    for run in runs:
        remote_status = str(run.get("remote_sync_status", "")).strip()
        if remote_status not in finished_run_status:
            continue
        remote_job_id = str(run.get("remote_job_id", "")).strip()
        if not remote_job_id:
            continue
        raw_notifications.append(
            {
                "id": f"supek-finished:{_canonical_path(run.get('run_dir', ''))}:{remote_status}",
                "severity": "info",
                "area": "Remote job finished",
                "target": str(run.get("run_display_name", run.get("run_name", ""))),
                "alert": finished_run_status[remote_status],
                "next_move": "Open Model Workflow to fetch logs, pull artifacts, or finalize the next run decision.",
                "open_view": "Model Workflow",
                "run_dir": str(run.get("run_dir", "")),
                "sequence": "",
            }
        )

    finished_ladder_status = {
        "outputs_staged": "BURA finished a remote campaign and staged outputs for safe download.",
        "outputs_returned": "BURA outputs were copied back and are ready for local review or finalize.",
        "finalized_local": "BURA outputs were finalized locally and the peptide may be ready for review or ingest.",
    }
    for ladder in peptide_ladders:
        remote_status = str(ladder.get("sync_status", "")).strip()
        if remote_status not in finished_ladder_status:
            continue
        remote_job_id = str(ladder.get("remote_job_id", "")).strip()
        if not remote_job_id:
            continue
        raw_notifications.append(
            {
                "id": f"bura-finished:{_canonical_path(ladder.get('run_dir', ''))}:{str(ladder.get('sequence', ''))}:{remote_status}",
                "severity": "info",
                "area": "Remote job finished",
                "target": str(ladder.get("sequence", "")),
                "alert": finished_ladder_status[remote_status],
                "next_move": "Open MD Validation to review the returned outputs or continue the ladder.",
                "open_view": "MD Validation",
                "run_dir": str(ladder.get("run_dir", "")),
                "sequence": str(ladder.get("sequence", "")),
            }
        )

    now_dt = datetime.now()
    raw_notifications.extend(
        _build_stale_work_notifications(
            raw_notifications,
            delivered=delivered,
            acknowledged_ids=acknowledged_ids,
            now_dt=now_dt,
        )
    )
    now = now_dt.isoformat(timespec="seconds")
    current_ids = {str(item.get("id", "")) for item in raw_notifications if str(item.get("id", "")).strip()}
    updated_delivered = {notification_id: str(delivered.get(notification_id, now)) for notification_id in current_ids}
    updated_acknowledged = sorted(notification_id for notification_id in acknowledged_ids if notification_id in current_ids)
    updated_cluster_status = {
        str(row.get("cluster", "")).strip(): str(row.get("status", "")).strip()
        for row in cluster_health_rows
        if str(row.get("cluster", "")).strip()
    }
    if (
        updated_delivered != delivered
        or updated_acknowledged != sorted(acknowledged_ids)
        or updated_cluster_status != previous_cluster_status
    ):
        save_dashboard_notifications(
            run_root,
            {
                "delivered": updated_delivered,
                "acknowledged_ids": updated_acknowledged,
                "cluster_status": updated_cluster_status,
            },
        )

    rows: list[dict[str, str]] = []
    for item in raw_notifications:
        notification_id = str(item.get("id", "")).strip()
        created_at = updated_delivered.get(notification_id, now)
        acknowledged = notification_id in set(updated_acknowledged)
        rows.append(
            {
                "id": notification_id,
                "state": "Acknowledged" if acknowledged else ("New" if notification_id not in delivered else "Open"),
                "severity": str(item.get("severity", "")),
                "area": str(item.get("area", "")),
                "target": str(item.get("target", "")),
                "alert": str(item.get("alert", "")),
                "next_move": str(item.get("next_move", "")),
                "open_view": str(item.get("open_view", "")),
                "created_at": created_at,
                "run_dir": str(item.get("run_dir", "")),
                "sequence": str(item.get("sequence", "")),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            1 if str(row.get("state", "")) == "Acknowledged" else 0,
            _notification_severity_rank(str(row.get("severity", ""))),
            str(row.get("created_at", "")),
            str(row.get("target", "")),
        ),
        reverse=False,
    )


def _notification_rows_for_display(
    state: dict[str, object],
    *,
    include_acknowledged: bool = False,
    limit: int | None = None,
) -> list[dict[str, str]]:
    rows = list(state.get("notifications", []))
    if not include_acknowledged:
        rows = [row for row in rows if str(row.get("state", "")) != "Acknowledged"]
    display_rows = [
        {
            "state": str(row.get("state", "")),
            "severity": str(row.get("severity", "")),
            "area": str(row.get("area", "")),
            "target": str(row.get("target", "")),
            "alert": str(row.get("alert", "")),
            "next_move": str(row.get("next_move", "")),
            "open_view": str(row.get("open_view", "")),
            "created_at": str(row.get("created_at", "")),
        }
        for row in rows
    ]
    return display_rows[:limit] if limit is not None else display_rows


def _readiness_alert_rows(state: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _cluster_health_rows(state):
        if str(row.get("status", "")) == "ok":
            continue
        cluster_name = str(row.get("cluster", "")).upper()
        rows.append(
            {
                "severity": "warning",
                "area": "Cluster",
                "target": cluster_name,
                "alert": str(row.get("summary", "")) or f"{cluster_name} needs attention.",
                "next_move": str(row.get("hint", "")) or f"Re-check {cluster_name} health before relying on remote actions.",
            }
        )

    overview = state.get("overview", {}) if isinstance(state.get("overview", {}), dict) else {}
    for row in list(overview.get("waiting_on_review", []))[:6]:
        rows.append(
            {
                "severity": "warning",
                "area": "Review",
                "target": str(row.get("sequence", "")),
                "alert": f"{row.get('sequence', '')} is still waiting on a human MD review label.",
                "next_move": str(row.get("next_action", "")) or "Open MD Validation and finish the review handoff.",
            }
        )

    for run in list(state.get("runs", [])):
        feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
        run_label = str(run.get("run_display_name", _path_name(run.get("run_dir", ""))))
        if bool(feedback_queue.get("can_continue", False)):
            rows.append(
                {
                    "severity": "info",
                    "area": "Model feedback",
                    "target": run_label,
                    "alert": "The full pending proposed batch is reviewed and ready to continue back into active learning.",
                    "next_move": "Open Model Workflow and run Continue AL from reviewed peptides.",
                }
            )
        elif str(feedback_queue.get("status", "")) == "blocked" and str(feedback_queue.get("pending_round_id", "")).strip():
            rows.append(
                {
                    "severity": "warning",
                    "area": "Model feedback",
                    "target": run_label,
                    "alert": str(feedback_queue.get("summary", "")) or "The pending proposed batch still needs more MD feedback work.",
                    "next_move": "Open Model Workflow to inspect the feedback queue, then finish the missing review/promotion/full-analysis steps.",
                }
            )
        if list(run.get("available_ingest_csvs", [])):
            rows.append(
                {
                    "severity": "info",
                    "area": "Model feedback",
                    "target": run_label,
                    "alert": "Reviewed MD feedback is staged and ready to ingest into the model.",
                    "next_move": "Open Model Workflow and run Ingest returned labels.",
                }
            )

    actions = list(state.get("actions", []))
    for kind, target, next_move in (
        ("supek-submit-preflight", "SUPEK submit path", "Fix the missing SUPEK dependency, then re-run the submit preflight before drafting another workflow."),
        ("bura-submit-readiness", "BURA staged campaign", "Fix the missing staged campaign dependency, then re-run the BURA readiness check before normalize/preflight/submit."),
        ("bura-preflight", "BURA preflight", "Inspect the preflight stderr and fix the campaign before you submit the chain."),
    ):
        action = _latest_action_by_kind(actions, kind)
        if action is None or str(action.get("status", "")) != "failed":
            continue
        rows.append(
            {
                "severity": "warning",
                "area": "Remote readiness",
                "target": target,
                "alert": str(action.get("title", "")) or "A readiness check failed.",
                "next_move": next_move,
            }
        )

    for row in list(state.get("md_slate_exceptions", []))[:6]:
        rows.append(
            {
                "severity": str(row.get("severity", "")) or "warning",
                "area": "MD recovery",
                "target": str(row.get("sequence", "")) or str(row.get("run", "")),
                "alert": str(row.get("summary", "")) or "An MD slate peptide needs recovery attention.",
                "next_move": str(row.get("next_move", "")) or "Open MD Validation -> Recovery center.",
            }
        )

    priority = {"warning": 0, "error": 0, "info": 1}
    return sorted(rows, key=lambda row: (priority.get(str(row.get("severity", "")), 2), str(row.get("area", "")), str(row.get("target", ""))))[:12]


def _render_cluster_health_notice(st, state: dict[str, object], cluster_name: str) -> None:
    health = _cluster_health_entry(state, cluster_name)
    status = str(health.get("overall_status", "unknown"))
    if status == "ok":
        return
    summary = str(health.get("summary", "")).strip() or f"{cluster_name.upper()} health needs attention."
    hint = str(health.get("hint", "")).strip()
    checked_at = str(health.get("checked_at", "")).strip()
    message = summary
    if hint:
        message += f" {hint}"
    if checked_at:
        message += f" Last checked: {checked_at}."
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.info(message)



def _render_cluster_health_panel(st, state: dict[str, object]) -> None:
    st.subheader("Cluster health")
    st.caption("Health checks are read-only and manual. They use Windows OpenSSH (`ssh-agent`, `ssh-add`, and your `~/.ssh/config` aliases) and reuse the latest cached result until you run them again.")

    action_cols = st.columns(3)
    if action_cols[0].button("Check SUPEK health", key="check_supek_health"):
        profile = get_cluster_profile(state.get("profiles", {}), "supek")
        if profile is None:
            st.warning(_cluster_profile_warning(state, "supek"))
        else:
            result = check_cluster_health(Path(str(state["run_root"])), "supek", profile)
            flash_level = "success" if str(result.get("overall_status", "unknown")) == "ok" else "warning"
            _stash_dashboard_flash(st, flash_level, f"Checked SUPEK health: {result.get('summary', 'done')}")
            _trigger_dashboard_rerun(st)
    if action_cols[1].button("Check BURA health", key="check_bura_health"):
        profile = get_cluster_profile(state.get("profiles", {}), "bura")
        if profile is None:
            st.warning(_cluster_profile_warning(state, "bura"))
        else:
            result = check_cluster_health(Path(str(state["run_root"])), "bura", profile)
            flash_level = "success" if str(result.get("overall_status", "unknown")) == "ok" else "warning"
            _stash_dashboard_flash(st, flash_level, f"Checked BURA health: {result.get('summary', 'done')}")
            _trigger_dashboard_rerun(st)
    if action_cols[2].button("Check all clusters", key="check_all_cluster_health"):
        results = check_all_cluster_health(Path(str(state["run_root"])), state.get("profiles", {}))
        if results:
            non_ok = [item for item in results if str(item.get("overall_status", "unknown")) != "ok"]
            if non_ok:
                _stash_dashboard_flash(st, "warning", f"Checked cluster health. {len(non_ok)} cluster(s) still need attention.")
            else:
                _stash_dashboard_flash(st, "success", "Checked cluster health. All configured clusters look healthy.")
            _trigger_dashboard_rerun(st)
        else:
            st.warning("No enabled cluster profiles are ready for health checks yet.")

    rows = _cluster_health_rows(state)
    cards = st.columns(len(rows)) if rows else []
    for column, row in zip(cards, rows):
        with column:
            st.markdown(f"#### {row['cluster'].upper()}")
            st.metric("Status", row["status"])
            st.caption(f"Last checked: {row['checked_at'] or '-'}")
            if row["local_auth"] != "n/a":
                st.caption(f"Local auth: {row['local_auth']}")
            st.caption(f"Remote probe: {row['remote_status']}")
            st.write(row["summary"] or "Health not checked yet.")
            if row["hint"]:
                st.caption(row["hint"])

    details_rows = []
    for cluster_name in SUPPORTED_CLUSTERS:
        health = _cluster_health_entry(state, cluster_name)
        details_rows.append(
            {
                "cluster": cluster_name,
                "status": str(health.get("overall_status", "unknown")),
                "checked_at": str(health.get("checked_at", "")),
                "summary": str(health.get("summary", "")),
                "hint": str(health.get("hint", "")),
            }
        )
    st.dataframe(details_rows)
    for cluster_name in SUPPORTED_CLUSTERS:
        health = _cluster_health_entry(state, cluster_name)
        details = [str(item) for item in health.get("details", []) if str(item).strip()]
        if not details:
            continue
        st.markdown(f"#### {cluster_name.upper()} health details")
        for line in details:
            st.write(f"- {line}")



def build_peptide_ladders(run_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in run_summaries:
        for campaign in run["md_campaigns"]:
            grouped.setdefault(str(campaign["sequence"]), []).append(campaign)

    ladders: list[dict[str, object]] = []
    for sequence, items in sorted(grouped.items()):
        profile_best: dict[str, dict[str, object]] = {}
        for item in items:
            profile = str(item["md_profile"])
            current = profile_best.get(profile)
            if current is None or _campaign_sort_key(item) > _campaign_sort_key(current):
                profile_best[profile] = item

        full_ready = bool(
            profile_best.get("full")
            and profile_best["full"]["job_root_status"] == profile_best["full"]["expected_terminal_status"]
        )

        next_profile = ""
        if not full_ready:
            for profile in ("line_smoke", "production_smoke", "full"):
                current = profile_best.get(profile)
                if current is None or current["job_root_status"] != current["expected_terminal_status"]:
                    next_profile = profile
                    break

        current = None
        if next_profile:
            current = profile_best.get(next_profile)
        elif profile_best:
            current = profile_best.get("full") or max(profile_best.values(), key=_campaign_sort_key)

        reference_campaign = (
            current
            or profile_best.get("full")
            or profile_best.get("production_smoke")
            or profile_best.get("line_smoke")
        )

        badges: list[str] = []
        if any(bool(item.get("legacy")) for item in items):
            badges.append("legacy")
        if full_ready:
            badges.append("ready_for_review")
        elif profile_best.get("full") and profile_best["full"].get("job_root_status") == "sasa_complete":
            badges.append("not ingest-ready")
        if current and str(current.get("sync_status", "")) in {"submitted", "running"}:
            badges.append("remote active")

        ladder = {
            "sequence": sequence,
            "campaigns": sorted(
                [
                    {
                        "campaign": item["campaign"],
                        "campaign_dir": item["campaign_dir"],
                        "md_profile": item["md_profile"],
                        "md_profile_label": _friendly_md_profile(str(item["md_profile"])),
                        "job_root_status": item["job_root_status"],
                        "job_root_status_label": _friendly_md_status(str(item["job_root_status"])),
                        "legacy": item["legacy"],
                        "cgmd_label": item["cgmd_label"],
                        "sync_status": item.get("sync_status", "not_synced"),
                        "remote_job_id": item.get("remote_job_id", ""),
                    }
                    for item in items
                ],
                key=lambda item: (_md_profile_sort_key(str(item["md_profile"])), item["campaign"]),
            ),
            "campaign_options": sorted(
                items,
                key=_campaign_sort_key,
                reverse=True,
            ),
            "line_smoke": profile_best.get("line_smoke"),
            "production_smoke": profile_best.get("production_smoke"),
            "full": profile_best.get("full"),
            "current": current,
            "next_profile": next_profile,
            "next_profile_label": _friendly_md_profile(next_profile) if next_profile else "",
            "ready_for_review": full_ready,
            "run_dir": str(reference_campaign["run_dir"]) if reference_campaign else "",
            "run_display_name": str(reference_campaign.get("run_display_name", "")) if reference_campaign else "",
            "cluster": str(reference_campaign["cluster"]) if reference_campaign else "",
            "source_batch_csv": str(reference_campaign["source_batch_csv"]) if reference_campaign else "",
            "source_batch_kind": str(reference_campaign.get("source_batch_kind", "")) if reference_campaign else "",
            "reuse_pdb_from": str(reference_campaign["campaign_dir"]) if reference_campaign else "",
            "exclude_nodes": str(reference_campaign["exclude_nodes"]) if reference_campaign else "",
            "sync_status": str(reference_campaign.get("sync_status", "not_synced")) if reference_campaign else "not_synced",
            "remote_job_id": str(reference_campaign.get("remote_job_id", "")) if reference_campaign else "",
            "badges": badges,
        }
        ladder["source_batch_kind_label"] = _source_batch_kind_label(str(ladder.get("source_batch_kind", "")))
        ladder["ingest_supported"] = _source_batch_ingest_supported(str(ladder.get("source_batch_kind", "")))
        ladder["ingest_blocker"] = _source_batch_ingest_blocker(str(ladder.get("source_batch_kind", "")))
        ladder["promotion_target_batch_csv"] = (
            _promotion_target_batch_csv(str(ladder.get("run_dir", "")), sequence)
            if str(ladder.get("source_batch_kind", "")) == "dashboard_generated"
            else ""
        )
        ladder["promotion_target_round_id"] = _promotion_round_id(
            str(ladder.get("promotion_target_batch_csv", "")),
            sequence,
        )
        ladder["promotion_available"] = bool(str(ladder.get("promotion_target_batch_csv", "")))
        ladder["next_step"] = _next_action_for_ladder(ladder)
        ladder["next_action"] = str(ladder["next_step"].get("summary", ""))
        ladder["prepare_next_command"] = (
            _prepare_stage_command(ladder=ladder, next_profile=next_profile) if next_profile else ""
        )
        if full_ready and ladder["full"]:
            ladder["make_ingest_command"] = _make_ingest_command(
                str(ladder["full"]["campaign_dir"]),
                str(ladder["full"]["review_path"]),
            )
        else:
            ladder["make_ingest_command"] = ""
        ladder["next_bura_commands"] = str(reference_campaign["next_commands"]) if reference_campaign else ""
        ladder["stage_meanings"] = [
            {
                "md_profile": profile,
                "label": _friendly_md_profile(profile),
                "description": str(MD_PROFILE_INFO.get(profile, {}).get("description", "")),
                "produces": str(MD_PROFILE_INFO.get(profile, {}).get("produces", "")),
            }
            for profile in ("line_smoke", "production_smoke", "full")
        ]
        ladders.append(ladder)
    return ladders


def build_peptide_inventory(
    run_root: Path,
    run_summaries: list[dict[str, object]],
    peptide_ladders: list[dict[str, object]],
    decisions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    suggested_rows: list[dict[str, str]] = []
    candidate_selection_rows: list[dict[str, str]] = []
    sent_for_md_rows: list[dict[str, str]] = []
    md_in_progress_rows: list[dict[str, str]] = []
    needs_review_rows: list[dict[str, str]] = []
    reporting_ready_rows: list[dict[str, str]] = []
    ready_for_ingest_rows: list[dict[str, str]] = []
    already_ingested_rows: list[dict[str, str]] = []
    review_pipeline_rows: list[dict[str, str]] = []
    ledger_entries: dict[tuple[str, str], dict[str, str]] = {}
    candidate_entries: dict[tuple[str, str], dict[str, object]] = {}
    campaign_lookup: dict[tuple[str, str], dict[str, object]] = {}
    latest_candidate_decisions: dict[tuple[str, str], dict[str, object]] = {}

    def _ledger_entry(*, run_key: str, run_label: str, sequence: str) -> dict[str, str]:
        key = (run_key, sequence)
        if key not in ledger_entries:
            ledger_entries[key] = {
                "sequence": sequence,
                "run": run_label,
                "suggested_via": "",
                "proposal_round": "",
                "md_stage": "-",
                "md_status": "-",
                "remote_sync": "-",
                "review_label": "",
                "lifecycle_state": "Suggested by model",
                "next_action": "",
            }
        return ledger_entries[key]

    def _candidate_entry(
        *,
        run_key: str,
        run_label: str,
        run_dir: str,
        run_slug: str,
        sequence: str,
    ) -> dict[str, object]:
        key = (run_key, sequence)
        if key not in candidate_entries:
            candidate_entries[key] = {
                "sequence": sequence,
                "run": run_label,
                "run_dir": run_dir,
                "run_key": run_key,
                "run_slug": run_slug,
                "proposal_round": "",
                "_sources": set(),
                "_strategies": set(),
                "_discovery_stages": set(),
            }
        return candidate_entries[key]

    if isinstance(decisions, list):
        for entry in decisions:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("scope", "")).strip() != "candidate":
                continue
            run_key = _canonical_path(str(entry.get("run_dir", "")))
            sequence = str(entry.get("sequence", "")).strip()
            if not run_key or not sequence:
                continue
            latest_candidate_decisions.setdefault((run_key, sequence), entry)

    ingested_sequences: set[str] = set()
    for run in run_summaries:
        run_key = str(run.get("run_dir_key", run.get("run_dir", "")))
        run_label = str(run.get("run_display_name", run.get("run_name", "")))
        for row in run.get("import_rows", []):
            sequence = str(row.get("sequence", "")).strip()
            if not sequence:
                continue
            ingested_sequences.add(sequence)
            entry = _ledger_entry(run_key=run_key, run_label=run_label, sequence=sequence)
            entry["review_label"] = str(row.get("label", entry.get("review_label", "")))
            entry["lifecycle_state"] = "Already ingested"
            entry["next_action"] = "Use the updated run state for the next propose/discovery/final-evaluation step."
            already_ingested_rows.append(
                {
                    "sequence": sequence,
                    "run": run_label,
                    "round_id": str(row.get("round_id", "")),
                    "import_csv": _path_name(str(row.get("path", ""))),
                }
            )

    for run in run_summaries:
        run_key = str(run.get("run_dir_key", run.get("run_dir", "")))
        run_label = str(run.get("run_display_name", run.get("run_name", "")))
        run_dir = str(run.get("run_dir", ""))
        run_slug = str(run.get("run_slug", _path_name(run_dir)))
        for campaign in run.get("md_campaigns", []):
            sequence = str(campaign.get("sequence", "")).strip()
            if not sequence:
                continue
            current = campaign_lookup.get((run_key, sequence))
            if current is None or _campaign_sort_key(campaign) > _campaign_sort_key(current):
                campaign_lookup[(run_key, sequence)] = campaign
        latest_batch = run.get("latest_batch") or {}
        for row in latest_batch.get("rows", []):
            sequence = str(row.get("sequence", "")).strip()
            if not sequence:
                continue
            entry = _ledger_entry(run_key=run_key, run_label=run_label, sequence=sequence)
            entry["suggested_via"] = "Proposed next batch"
            entry["proposal_round"] = str(row.get("round_id", ""))
            candidate = _candidate_entry(
                run_key=run_key,
                run_label=run_label,
                run_dir=run_dir,
                run_slug=run_slug,
                sequence=sequence,
            )
            candidate["proposal_round"] = str(row.get("round_id", ""))
            candidate["_sources"].add("Proposed next batch")
            strategy = str(row.get("acquisition_strategy", "")).strip()
            if strategy:
                candidate["_strategies"].add(strategy)
            suggested_rows.append(
                {
                    "sequence": sequence,
                    "run": run_label,
                    "source": "Proposed next batch",
                    "round_id": str(row.get("round_id", "")),
                    "strategy": str(row.get("acquisition_strategy", "")),
                }
            )
        for row in run.get("discovery_sequences", []):
            sequence = str(row.get("sequence", "")).strip()
            if not sequence:
                continue
            entry = _ledger_entry(run_key=run_key, run_label=run_label, sequence=sequence)
            suggested_via = [item.strip() for item in entry.get("suggested_via", "").split(" + ") if item.strip()]
            if "Discovery shortlist" not in suggested_via:
                suggested_via.append("Discovery shortlist")
            entry["suggested_via"] = " + ".join(suggested_via)
            candidate = _candidate_entry(
                run_key=run_key,
                run_label=run_label,
                run_dir=run_dir,
                run_slug=run_slug,
                sequence=sequence,
            )
            candidate["_sources"].add("Discovery shortlist")
            strategy = str(row.get("strategy", "")).strip()
            if strategy:
                candidate["_strategies"].add(strategy)
            discovery_stage = str(row.get("surrogate_stage", "")).strip()
            if discovery_stage:
                candidate["_discovery_stages"].add(discovery_stage)
            suggested_rows.append(
                {
                    "sequence": sequence,
                    "run": run_label,
                    "source": "Discovery shortlist",
                    "round_id": "",
                    "strategy": str(row.get("strategy", "")),
                }
                )

    for (run_key, sequence), candidate in candidate_entries.items():
        campaign = campaign_lookup.get((run_key, sequence))
        decision = latest_candidate_decisions.get((run_key, sequence))
        source_labels = sorted(str(item) for item in candidate.get("_sources", set()) if str(item).strip())
        strategy_labels = sorted(str(item) for item in candidate.get("_strategies", set()) if str(item).strip())
        discovery_stages = sorted(str(item) for item in candidate.get("_discovery_stages", set()) if str(item).strip())
        decision_type = str(decision.get("decision_type", "")).strip() if isinstance(decision, dict) else ""
        source_batch_csv = find_md_source_batch_csv(run_root, Path(str(candidate.get("run_dir", ""))), sequence)
        source_batch_kind = _source_batch_kind(run_root, source_batch_csv)
        launch_ready = bool(campaign) or bool(source_batch_csv)
        launch_blocker = (
            ""
            if launch_ready
            else "No source batch CSV currently contains this peptide. Export or create a batch CSV row before launching it into MD."
        )
        if campaign:
            candidate_status = "Already in MD"
            next_action = "Open MD Validation and continue the guided ladder for this peptide."
        elif decision_type == "select_candidate_for_md":
            candidate_status = "Selected for MD"
            next_action = (
                "Open MD Validation and prepare the first safe ladder step for this peptide."
                if launch_ready
                else "Export or create a source batch CSV for this peptide before launching the MD slate."
            )
        elif decision_type == "defer_candidate":
            candidate_status = "Deferred"
            next_action = "Leave this peptide out of the next MD batch until the thesis priorities change."
        elif decision_type == "reject_candidate":
            candidate_status = "Rejected"
            next_action = "Keep the peptide out of the validation set and preserve the rationale in the decision log."
        else:
            candidate_status = "Undecided"
            next_action = "Record whether this peptide should go to MD now, later, or not at all."
        priority_score, priority_band, priority_reason = _candidate_priority_fields(
            candidate_status=candidate_status,
            source_labels=source_labels,
            strategy_labels=strategy_labels,
        )
        candidate_selection_rows.append(
            {
                "sequence": str(candidate.get("sequence", "")),
                "run": str(candidate.get("run", "")),
                "run_dir": str(candidate.get("run_dir", "")),
                "run_slug": str(candidate.get("run_slug", "")),
                "source": " + ".join(source_labels) or "-",
                "proposal_round": str(candidate.get("proposal_round", "")) or "-",
                "strategy": " + ".join(strategy_labels) or "-",
                "discovery_stage": " + ".join(discovery_stages) or "-",
                "candidate_status": candidate_status,
                "md_stage": _friendly_md_profile(str(campaign.get("md_profile", ""))) if campaign else "-",
                "md_status": _friendly_md_status(str(campaign.get("job_root_status", ""))) if campaign else "-",
                "remote_sync": _friendly_remote_sync(str(campaign.get("sync_status", ""))) if campaign else "-",
                "campaign": str(campaign.get("campaign", "")) if campaign else "-",
                "source_batch_csv": source_batch_csv or "-",
                "source_batch_kind": _source_batch_kind_label(source_batch_kind) if source_batch_csv else "-",
                "launch_ready": "yes" if launch_ready else "no",
                "launch_blocker": launch_blocker or "-",
                "last_decision": _decision_type_label(decision_type) if decision_type else "-",
                "decision_title": str(decision.get("title", "")) if isinstance(decision, dict) else "",
                "decision_rationale": str(decision.get("rationale", "")) if isinstance(decision, dict) else "",
                "priority_score": str(priority_score),
                "priority_band": priority_band,
                "priority_reason": priority_reason,
                "next_action": next_action,
            }
        )

    for ladder in peptide_ladders:
        sequence = str(ladder.get("sequence", ""))
        run_key = _canonical_path(str(ladder.get("run_dir", "")))
        run_label = str(ladder.get("run_display_name", "") or _path_name(ladder.get("run_dir", "")))
        entry = _ledger_entry(run_key=run_key, run_label=run_label, sequence=sequence)
        current = ladder.get("current")
        full_item = ladder.get("full")
        next_step_title = str((ladder.get("next_step") or {}).get("title", ""))
        entry["next_action"] = next_step_title or entry.get("next_action", "")
        base_row = {
            "sequence": sequence,
            "run": run_label,
            "next_step": str((ladder.get("next_step") or {}).get("title", "")),
            "sync_status": str(ladder.get("sync_status", "")),
        }
        if current:
            entry["md_stage"] = _friendly_md_profile(str(current.get("md_profile", "")))
            entry["md_status"] = _friendly_md_status(str(current.get("job_root_status", "")))
            entry["remote_sync"] = _friendly_remote_sync(str(current.get("sync_status", "")))
            if entry.get("lifecycle_state") != "Already ingested":
                entry["lifecycle_state"] = "Sent for MD"
            sent_for_md_rows.append(
                {
                    **base_row,
                    "current_stage": _friendly_md_profile(str(current.get("md_profile", ""))),
                    "campaign": str(current.get("campaign", "")),
                }
            )
            if str(current.get("sync_status", "")) in {"staged_remote", "submitted", "running", "outputs_staged", "outputs_returned"}:
                if entry.get("lifecycle_state") != "Already ingested":
                    entry["lifecycle_state"] = "MD in progress"
                md_in_progress_rows.append(
                    {
                        **base_row,
                        "current_stage": _friendly_md_profile(str(current.get("md_profile", ""))),
                        "campaign": str(current.get("campaign", "")),
                        "state": str(current.get("sync_status", "")),
                    }
                )
        if ladder.get("ready_for_review") and full_item:
            ingest_csv_path = _ingest_csv_path(str(full_item.get("campaign_dir", "")))
            ingest_exists = ingest_csv_path.exists()
            full_source_batch_kind = str(full_item.get("source_batch_kind", ""))
            ingest_supported = _source_batch_ingest_supported(full_source_batch_kind)
            promotion_available = bool(ladder.get("promotion_available"))
            promotion_round_id = str(ladder.get("promotion_target_round_id", ""))
            promoted_at = str(full_item.get("promoted_to_real_batch_at", "")).strip()
            promoted_round_id = str(full_item.get("promoted_round_id", "")).strip()
            promoted_state_label = (
                f"Promoted into real batch (round {promoted_round_id})"
                if promoted_at and promoted_round_id
                else ("Promoted into real batch" if promoted_at else "")
            )
            label_is_final = str(full_item.get("cgmd_label", "")).strip() in {"0", "1"}
            review_status = review_evidence_status(full_item)
            review_evidence_row = _review_evidence_summary_row(full_item)
            evidence_ready = bool(review_status.get("ingest_ready", False))
            runtime_review_row = {
                "md_runtime": str(full_item.get("md_runtime_wall_hms", "")) or "-",
                "md_ns_per_day": str(full_item.get("md_runtime_ns_per_day", "")) or "-",
                "ap_contact_exact_paper_200ns": str(full_item.get("ap_contact_same_paper_formula_200ns", "")) or "-",
                "paper_path_ap_contact_200ns": str(full_item.get("paper_path_ap_contact_200ns", "")) or "-",
            }

            if label_is_final and evidence_ready and ingest_supported:
                entry["review_label"] = str(full_item.get("cgmd_label", ""))
                if entry.get("lifecycle_state") != "Already ingested":
                    entry["lifecycle_state"] = "Ready for ingest"
                entry["next_action"] = (
                    "Run Ingest returned labels in Model Workflow"
                    if ingest_exists
                    else "Create cgmd_ingest.csv"
                )
                ready_for_ingest_rows.append(
                    {
                        **base_row,
                        "campaign": str(full_item.get("campaign", "")),
                        "label": str(full_item.get("cgmd_label", "")),
                        "review_csv": _path_name(str(full_item.get("review_path", ""))) or "-",
                        "ingest_csv": _path_name(ingest_csv_path) if ingest_exists else "-",
                        "promotion_state": promoted_state_label or "-",
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
                review_pipeline_rows.append(
                    {
                        "sequence": sequence,
                        "run": run_label,
                        "campaign": str(full_item.get("campaign", "")),
                        "review_state": "Ready for ingest",
                        "current_label": str(full_item.get("cgmd_label", "")),
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "ingest_csv": _path_name(ingest_csv_path) if ingest_exists else "-",
                        "ingest_support": "AL-ingestable",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": promoted_state_label or "-",
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
            elif label_is_final and not evidence_ready:
                entry["review_label"] = str(full_item.get("cgmd_label", ""))
                if entry.get("lifecycle_state") != "Already ingested":
                    entry["lifecycle_state"] = "Needs review / label"
                entry["next_action"] = "Complete the evidence-backed review fields before creating model feedback."
                needs_review_rows.append(
                    {
                        **base_row,
                        "campaign": str(full_item.get("campaign", "")),
                        "status": _friendly_md_status(str(full_item.get("job_root_status", ""))),
                        "review_csv": _path_name(str(full_item.get("review_path", ""))) or "-",
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": promoted_state_label or "-",
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
                review_pipeline_rows.append(
                    {
                        "sequence": sequence,
                        "run": run_label,
                        "campaign": str(full_item.get("campaign", "")),
                        "review_state": "Needs review evidence",
                        "current_label": str(full_item.get("cgmd_label", "")),
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "ingest_csv": "-",
                        "ingest_support": "Reporting-only batch" if not ingest_supported else "AL-ingestable",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": promoted_state_label or "-",
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
            elif label_is_final:
                entry["review_label"] = str(full_item.get("cgmd_label", ""))
                if entry.get("lifecycle_state") != "Already ingested":
                    entry["lifecycle_state"] = "Reviewed for reporting"
                entry["next_action"] = (
                    "Promote this reviewed peptide into the real proposed batch, then create cgmd_ingest.csv."
                    if promotion_available
                    else "Use this result for thesis reporting, or wait until the peptide appears in a real proposed batch before ingesting."
                )
                reporting_ready_rows.append(
                    {
                        **base_row,
                        "campaign": str(full_item.get("campaign", "")),
                        "label": str(full_item.get("cgmd_label", "")),
                        "review_csv": _path_name(str(full_item.get("review_path", ""))) or "-",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": (
                            f"Can promote now (round {promotion_round_id})"
                            if promotion_available and promotion_round_id
                            else ("Can promote now" if promotion_available else "Waiting for real proposed batch")
                        ),
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
                review_pipeline_rows.append(
                    {
                        "sequence": sequence,
                        "run": run_label,
                        "campaign": str(full_item.get("campaign", "")),
                        "review_state": "Reviewed for reporting",
                        "current_label": str(full_item.get("cgmd_label", "")),
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "ingest_csv": "-",
                        "ingest_support": "Reporting-only batch",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": (
                            f"Can promote now (round {promotion_round_id})"
                            if promotion_available and promotion_round_id
                            else ("Can promote now" if promotion_available else "Waiting for real proposed batch")
                        ),
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
            else:
                if entry.get("lifecycle_state") != "Already ingested":
                    entry["lifecycle_state"] = "Needs review / label"
                entry["next_action"] = (
                    "Assign cgmd_label for reporting"
                    if not ingest_supported
                    else "Assign cgmd_label in md_review.csv"
                )
                needs_review_rows.append(
                    {
                        **base_row,
                        "campaign": str(full_item.get("campaign", "")),
                        "status": _friendly_md_status(str(full_item.get("job_root_status", ""))),
                        "review_csv": _path_name(str(full_item.get("review_path", ""))) or "-",
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": (
                            f"Can promote now (round {promotion_round_id})"
                            if (not ingest_supported) and promotion_available and promotion_round_id
                            else ("Can promote now" if (not ingest_supported) and promotion_available else "-")
                        ),
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )
                review_pipeline_rows.append(
                    {
                        "sequence": sequence,
                        "run": run_label,
                        "campaign": str(full_item.get("campaign", "")),
                        "review_state": "Needs review / label",
                        "current_label": "-",
                        "review_notes": str(full_item.get("review_notes", "")) or "-",
                        "ingest_csv": "-",
                        "ingest_support": "Reporting-only batch" if not ingest_supported else "AL-ingestable",
                        "source_batch_kind": _source_batch_kind_label(full_source_batch_kind),
                        "promotion_state": (
                            f"Can promote now (round {promotion_round_id})"
                            if (not ingest_supported) and promotion_available and promotion_round_id
                            else ("Can promote now" if (not ingest_supported) and promotion_available else "-")
                        ),
                        **runtime_review_row,
                        **review_evidence_row,
                        "next_action": entry["next_action"],
                    }
                )

    def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str, str, str]] = set()
        output: list[dict[str, str]] = []
        for row in rows:
            key = (
                str(row.get("sequence", "")),
                str(row.get("run", "")),
                str(row.get("source", row.get("campaign", row.get("current_stage", "")))),
                str(row.get("round_id", row.get("state", ""))),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(row)
        return output

    suggested_rows = _dedupe(suggested_rows)
    candidate_selection_rows = sorted(
        candidate_selection_rows,
        key=lambda row: (
            ["Undecided", "Selected for MD", "Already in MD", "Deferred", "Rejected"].index(str(row.get("candidate_status", "Undecided")))
            if str(row.get("candidate_status", "Undecided")) in ["Undecided", "Selected for MD", "Already in MD", "Deferred", "Rejected"]
            else 99,
            -int(str(row.get("priority_score", "0")) or 0),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )
    sent_for_md_rows = _dedupe(sent_for_md_rows)
    md_in_progress_rows = _dedupe(md_in_progress_rows)
    needs_review_rows = _dedupe(needs_review_rows)
    reporting_ready_rows = _dedupe(reporting_ready_rows)
    ready_for_ingest_rows = _dedupe(ready_for_ingest_rows)
    already_ingested_rows = _dedupe(already_ingested_rows)
    review_pipeline_rows = _dedupe(review_pipeline_rows)
    ledger_rows = sorted(
        ledger_entries.values(),
        key=lambda row: (
            PEPTIDE_BUCKET_ORDER.index(str(row.get("lifecycle_state", "Suggested by model")))
            if str(row.get("lifecycle_state", "Suggested by model")) in PEPTIDE_BUCKET_ORDER
            else len(PEPTIDE_BUCKET_ORDER),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )

    return {
        "suggested_by_model": suggested_rows,
        "candidate_selection": candidate_selection_rows,
        "review_pipeline": review_pipeline_rows,
        "sent_for_md": sent_for_md_rows,
        "md_in_progress": md_in_progress_rows,
        "needs_review": needs_review_rows,
        "reviewed_for_reporting": reporting_ready_rows,
        "ready_for_ingest": ready_for_ingest_rows,
        "already_ingested": already_ingested_rows,
        "ledger": ledger_rows,
        "counts": {
            "Suggested by model": len(suggested_rows),
            "Sent for MD": len(sent_for_md_rows),
            "MD in progress": len(md_in_progress_rows),
            "Needs review / label": len(needs_review_rows),
            "Reviewed for reporting": len(reporting_ready_rows),
            "Ready for ingest": len(ready_for_ingest_rows),
            "Already ingested": len(already_ingested_rows),
        },
    }



def _summarize_action(action: dict[str, object]) -> dict[str, str]:
    return {
        "id": str(action.get("id", "")),
        "title": str(action.get("title", "")),
        "scope": str(action.get("scope", "local")),
        "cluster": str(action.get("cluster", "")),
        "status": str(action.get("status", "")),
        "related_run": _path_name(str(action.get("related_run", ""))),
        "related_campaign": _path_name(str(action.get("related_campaign", ""))),
        "related_sequence": str(action.get("related_sequence", "")),
        "started_at": str(action.get("started_at", "")),
        "finished_at": str(action.get("finished_at", "")),
        "exit_code": "" if action.get("exit_code") is None else str(action.get("exit_code")),
        "remote_job_id": str(action.get("remote_job_id", "")),
        "sync_status": str(action.get("sync_status", "")),
    }


def _transfer_status_label(status: str) -> str:
    mapping = {
        "not_synced": "Local only",
        "staged_remote": "Uploaded / staged remotely",
        "submitted": "Remote job submitted",
        "running": "Remote job running",
        "outputs_staged": "Downloaded into safe staging",
        "outputs_returned": "Copied back into live campaign",
        "finalized_local": "Finalized locally",
    }
    return mapping.get(status, status or "-")


def _transfer_manifest_rows(sync_records: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in sync_records:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        cluster = str(record.get("cluster", "")).strip() or "-"
        sequence = str(record.get("related_sequence", "")).strip()
        related_campaign = str(record.get("related_campaign", "")).strip()
        related_run = str(record.get("related_run", "")).strip()
        target_key = str(record.get("target_key", "")).strip()
        local_stage_path = str(metadata.get("local_stage_path", "")).strip()
        remote_path = str(record.get("remote_path", "")).strip()
        status = str(record.get("status", "")).strip()

        if sequence:
            target_kind = "Peptide package"
            target = sequence
        elif related_campaign:
            target_kind = "MD campaign"
            target = _path_name(related_campaign) or _path_name(target_key) or "-"
        else:
            target_kind = "Model run"
            target = _path_name(related_run) or _path_name(target_key) or "-"

        source_path = related_campaign or related_run or target_key or "-"
        if status in {"outputs_staged", "outputs_returned"} and remote_path:
            transfer_direction = "Remote -> local"
        elif status in {"staged_remote", "submitted", "running"} and remote_path:
            transfer_direction = "Local -> remote"
        else:
            transfer_direction = "Local state"

        if status == "staged_remote":
            next_step = "Run the remote preflight or submit sequence next."
        elif status in {"submitted", "running"}:
            next_step = "Monitor the queue or fetch the latest remote logs."
        elif status == "outputs_staged":
            next_step = "Inspect the staged files, then finalize or merge them locally."
        elif status == "outputs_returned":
            next_step = "Finalize the copied-back outputs locally."
        elif status == "finalized_local":
            next_step = "Use the refreshed local state for review, ingest, or the next ladder step."
        else:
            next_step = "This target has not been transferred yet."

        rows.append(
            {
                "cluster": cluster,
                "target_kind": target_kind,
                "target": target,
                "transfer_state": _transfer_status_label(status),
                "direction": transfer_direction,
                "source_path": source_path,
                "remote_path": remote_path or "-",
                "staging_path": local_stage_path or "-",
                "remote_job_id": str(record.get("remote_job_id", "")).strip() or "-",
                "updated_at": str(record.get("updated_at", "")).strip(),
                "next_action": next_step,
            }
        )
    return rows


def _artifact_rows_for_run(state: dict[str, object], run_dir: str) -> list[dict[str, object]]:
    run_key = _canonical_path(run_dir)
    return [
        row
        for row in list(state.get("artifact_verification", []))
        if _canonical_path(str(row.get("run_dir", ""))) == run_key
    ]


def _artifact_rows_for_ladder(state: dict[str, object], ladder: dict[str, object]) -> list[dict[str, object]]:
    run_key = _canonical_path(str(ladder.get("run_dir", "")))
    sequence = str(ladder.get("sequence", ""))
    return [
        row
        for row in list(state.get("artifact_verification", []))
        if _canonical_path(str(row.get("run_dir", ""))) == run_key
        and str(row.get("sequence", "")) in {"", sequence, "-"}
    ]


def _render_artifact_verification_workspace(
    st,
    rows: list[dict[str, object]],
    *,
    title: str,
    caption: str,
    key_prefix: str,
    render_export_pack=None,
) -> None:
    st.subheader(title)
    st.caption(caption)
    if not rows:
        st.success("No artifact integrity issues are visible in this scope right now.")
        return
    summary = build_artifact_verification_summary(rows)
    _render_metric_cards(
        st,
        [
            ("Checked targets", summary.get("total", 0)),
            ("Verified", summary.get("verified", 0)),
            ("Waiting on outputs", summary.get("waiting", 0)),
            ("Attention needed", summary.get("attention", 0)),
            ("Errors", summary.get("errors", 0)),
            ("Warnings", summary.get("warnings", 0)),
        ],
    )
    st.dataframe(rows)
    options = [
        f"{row.get('run', '')} :: {row.get('target', '')}"
        for row in rows
    ]
    selected_label = st.selectbox(
        "Inspect artifact verification row",
        options,
        index=0,
        key=f"{key_prefix}_artifact_select",
    )
    selected_row = rows[options.index(selected_label)]
    _render_recommended_card(
        st,
        eyebrow="Artifact verification",
        title=str(selected_row.get("summary", "")) or "Inspect the selected artifact verification row",
        summary=str(selected_row.get("next_move", "")),
        why=(
            f"Scope: {selected_row.get('target_kind', '-')} | "
            f"Sync state: {selected_row.get('sync_state', '-')} | "
            f"Verification: {selected_row.get('verification_state', '-')}"
        ),
        do_now=(
            f"Completeness: {selected_row.get('stage_completeness', '-')} | "
            f"Missing groups: {selected_row.get('missing_groups', '-')}"
        ),
        next_after=f"Safest retry: {selected_row.get('safest_retry_action', '-')}",
    )
    group_states = selected_row.get("group_states", [])
    if isinstance(group_states, list) and group_states:
        st.markdown("#### Artifact groups for selected row")
        st.dataframe(group_states)
    st.caption(
        f"Expected artifacts: {selected_row.get('expected_artifacts', '-')} | "
        f"Found artifacts: {selected_row.get('found_artifacts', '-')}"
    )
    if render_export_pack is not None:
        render_export_pack(
            st,
            title="Artifact verification export",
            description="Use this as the copy-ready audit of expected versus found artifacts for the visible scope.",
            rows=rows,
            key_prefix=f"{key_prefix}_artifact_export",
        )


def _actions_for_run(actions: list[dict[str, object]], run_dir: str) -> list[dict[str, object]]:
    run_key = _canonical_path(run_dir)
    return [
        action
        for action in actions
        if _canonical_path(str(action.get("related_run", ""))) == run_key
        or _path_is_within(str(action.get("related_campaign", "")), run_key)
    ]



def _actions_for_ladder(actions: list[dict[str, object]], ladder: dict[str, object]) -> list[dict[str, object]]:
    run_key = _canonical_path(ladder.get("run_dir", ""))
    sequence = str(ladder.get("sequence", ""))
    campaign_paths = {
        _canonical_path(item.get("campaign_dir", ""))
        for item in [ladder.get("line_smoke"), ladder.get("production_smoke"), ladder.get("full"), ladder.get("current")]
        if item
    }
    filtered: list[dict[str, object]] = []
    for action in actions:
        action_run = _canonical_path(str(action.get("related_run", "")))
        action_campaign = _canonical_path(str(action.get("related_campaign", "")))
        action_sequence = str(action.get("related_sequence", ""))
        if action_sequence == sequence and (action_run == run_key or action_campaign in campaign_paths):
            filtered.append(action)
            continue
        if action_campaign and action_campaign in campaign_paths:
            filtered.append(action)
    return filtered



def build_overview(
    run_summaries: list[dict[str, object]],
    peptide_ladders: list[dict[str, object]],
    actions: list[dict[str, object]],
    snapshots: list[dict[str, object]],
    sync_records: list[dict[str, object]],
    peptide_inventory: dict[str, object],
    md_slates: list[dict[str, object]],
    md_slate_exceptions: list[dict[str, object]],
    remote_reconciliation: list[dict[str, object]],
    bura_utilization: dict[str, object],
    artifact_verification: list[dict[str, object]],
) -> dict[str, object]:
    latest_proposals: list[dict[str, str]] = []
    today_queue: list[dict[str, str]] = []
    blocked_items: list[dict[str, str]] = []
    feedback_queue_rows: list[dict[str, str]] = []
    for run in run_summaries:
        latest_batch = run.get("latest_batch") or {}
        for row in latest_batch.get("rows", []):
            latest_proposals.append(
                {
                    "run": str(run.get("run_display_name", run.get("run_name", ""))),
                    "sequence": row.get("sequence", ""),
                    "round_id": row.get("round_id", ""),
                    "acquisition_strategy": row.get("acquisition_strategy", ""),
                    "acquisition_score": row.get("acquisition_score", ""),
                }
            )
        next_step = run.get("recommended_next_step", {}) if isinstance(run.get("recommended_next_step"), dict) else {}
        run_target = str(run.get("run_display_name", run.get("run_name", "")))
        feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
        if str(feedback_queue.get("pending_round_id", "")).strip():
            feedback_queue_rows.append(
                {
                    "run": run_target,
                    "pending_round": str(feedback_queue.get("pending_round_id", "")) or "-",
                    "proposed_peptides": str(len(list(feedback_queue.get("pending_sequences", [])))),
                    "ready_now": str(feedback_queue.get("ready_count", 0)),
                    "blocked": str(feedback_queue.get("blocked_count", 0)),
                    "state": (
                        "Ready to continue AL"
                        if bool(feedback_queue.get("can_continue", False))
                        else "Waiting on more MD feedback"
                    ),
                    "next_move": (
                        "Run Continue AL from reviewed peptides"
                        if bool(feedback_queue.get("can_continue", False))
                        else str(feedback_queue.get("summary", "")) or "Finish the missing review/promotion/full-analysis work."
                    ),
                }
            )
        if bool(feedback_queue.get("can_continue", False)):
            today_queue.append(
                _queue_row(
                    category="Ingest",
                    target=run_target,
                    action_now="Continue AL from reviewed peptides",
                    why=str(feedback_queue.get("summary", "")) or "The full pending proposed batch is reviewed and can re-enter the model now.",
                    open_view="Model Workflow",
                )
            )
        elif str(feedback_queue.get("status", "")) == "blocked" and str(feedback_queue.get("pending_round_id", "")).strip():
            today_queue.append(
                _queue_row(
                    category="Review",
                    target=run_target,
                    action_now="Finish the MD feedback queue",
                    why=str(feedback_queue.get("summary", "")) or "The pending proposed batch still needs review, promotion, or completed full-analysis outputs.",
                    open_view="Model Workflow",
                )
            )
        elif list(run.get("available_ingest_csvs", [])):
            today_queue.append(
                _queue_row(
                    category="Ingest",
                    target=run_target,
                    action_now="Run Ingest returned labels",
                    why="A reviewed `cgmd_ingest.csv` already exists for this run, so the active-learning loop can be closed now.",
                    open_view="Model Workflow",
                )
            )
        elif str(run.get("remote_sync_status", "")) in {"submitted", "running"} and str(run.get("remote_job_id", "")):
            today_queue.append(
                _queue_row(
                    category="Remote monitoring",
                    target=run_target,
                    action_now="Poll the SUPEK queue",
                    why="A remote SUPEK job is already active for this run, so monitoring is safer than launching new work.",
                    open_view="Model Workflow",
                )
            )
        elif str(next_step.get("title", "")) == "Review and export thesis results":
            today_queue.append(
                _queue_row(
                    category="Reporting",
                    target=run_target,
                    action_now="Review final metrics for thesis reporting",
                    why="This run already has a frozen holdout evaluation and is ready for results tables or plots.",
                    open_view="Results",
                )
            )
        else:
            today_queue.append(
                _queue_row(
                    category="Model workflow",
                    target=run_target,
                    action_now=str(next_step.get("title", "")),
                    why=str(next_step.get("summary", "")),
                    open_view="Model Workflow",
                )
            )

    candidate_rows = list(peptide_inventory.get("candidate_selection", [])) if isinstance(peptide_inventory.get("candidate_selection", []), list) else []
    selected_candidate_rows = [
        row for row in candidate_rows if str(row.get("candidate_status", "")) == "Selected for MD"
    ][:6]
    for row in selected_candidate_rows:
        today_queue.append(
            _queue_row(
                category="MD preparation",
                target=f"{row.get('sequence', '')} ({row.get('run', '')})",
                action_now="Advance this selected candidate into MD",
                why=f"Source: {row.get('source', '-')} | Strategy: {row.get('strategy', '-')} | Next: {row.get('next_action', '-')}",
                open_view="MD Validation",
            )
        )

    undecided_candidate_rows = [
        row for row in candidate_rows if str(row.get("candidate_status", "")) == "Undecided"
    ]
    highlighted_candidate_rows = [
        row for row in undecided_candidate_rows if _candidate_priority_rank(str(row.get("priority_band", ""))) >= 3
    ]
    if not highlighted_candidate_rows:
        highlighted_candidate_rows = undecided_candidate_rows
    highlighted_candidate_rows = highlighted_candidate_rows[:8]
    for row in highlighted_candidate_rows:
        today_queue.append(
            _queue_row(
                category="Candidate selection",
                target=f"{row.get('sequence', '')} ({row.get('run', '')})",
                action_now=f"Choose whether this {str(row.get('priority_band', '')).lower()} candidate should go to MD",
                why=f"Source: {row.get('source', '-')} | Strategy: {row.get('strategy', '-')} | Priority: {row.get('priority_reason', '-')}",
                open_view="Peptides",
            )
        )
    remaining_undecided = max(len(undecided_candidate_rows) - len(highlighted_candidate_rows), 0)
    if remaining_undecided:
        today_queue.append(
            _queue_row(
                category="Candidate selection",
                target="Remaining candidate pool",
                action_now=f"Review the remaining {remaining_undecided} undecided candidates",
                why="The highest-priority shortlist candidates are already surfaced individually; use Peptides -> Candidate selection to review the rest in batch.",
                open_view="Peptides",
            )
        )

    active_md_slates = [
        slate for slate in md_slates if str(slate.get("effective_status", "")) not in {"completed", "completed_with_failures", "cancelled"}
    ]
    visible_remote_reconciliation = [
        row
        for row in remote_reconciliation
        if str(row.get("severity", "")) in {"warning", "error"}
    ]
    for row in visible_remote_reconciliation[:6]:
        target = str(row.get("sequence", "")) if str(row.get("sequence", "")) not in {"", "-"} else str(row.get("run", ""))
        today_queue.append(
            _queue_row(
                category="Remote reconciliation",
                target=target or str(row.get("remote_job_id", "")),
                action_now=str(row.get("recommended_recovery", "")) or "Reconcile the remote job state",
                why=str(row.get("summary", "")) or "The dashboard-tracked job and latest queue snapshot disagree.",
                open_view=str(row.get("open_view", "")) or "Operations",
            )
        )
        blocked_items.append(
            {
                "scope": "Remote reconciliation",
                "target": target or str(row.get("remote_job_id", "")),
                "blocker": str(row.get("summary", "")),
                "what_to_do": str(row.get("next_move", "")),
            }
        )
    visible_md_slate_exceptions = [
        row
        for row in md_slate_exceptions
        if str(row.get("severity", "")) in {"warning", "error"}
    ]
    for row in visible_md_slate_exceptions[:6]:
        today_queue.append(
            _queue_row(
                category="Slate recovery",
                target=str(row.get("sequence", "")) or f"{row.get('run', '')} [{row.get('slate_id', '')}]",
                action_now=(
                    "Recover the stale MD peptide"
                    if bool(row.get("recover_available", False))
                    else (
                        "Rebind the peptide to the tracked job"
                        if bool(row.get("rebind_available", False))
                        else "Inspect the MD slate exception"
                    )
                ),
                why=str(row.get("summary", "")) or "This peptide needs recovery attention inside the active MD slate.",
                open_view="MD Validation",
            )
        )
        blocked_items.append(
            {
                "scope": "MD slate recovery",
                "target": str(row.get("sequence", "")) or str(row.get("run", "")),
                "blocker": str(row.get("summary", "")),
                "what_to_do": str(row.get("next_move", "")),
            }
        )
    for slate in active_md_slates[:4]:
        today_queue.append(
            _queue_row(
                category="MD slate",
                target=f"{_path_name(str(slate.get('run_dir', '')))} ({int(slate.get('peptide_count', 0))} peptides)",
                action_now=(
                    "Resume the paused MD slate"
                    if str(slate.get("effective_status", "")) == "paused"
                    else "Monitor the active MD slate"
                ),
                why=(
                    f"State: {slate.get('effective_status', '-')} | "
                    f"active={slate.get('active_count', 0)} | "
                    f"blocked={slate.get('blocked_count', 0)} | "
                    f"review-ready={slate.get('review_ready_count', 0)}"
                ),
                open_view="MD Validation",
            )
        )

    visible_artifact_attention = [
        row
        for row in artifact_verification
        if str(row.get("verification_state", "")) == "Attention needed"
    ]
    for row in visible_artifact_attention[:6]:
        open_view = "MD Validation" if str(row.get("scope", "")) == "campaign" else "Operations"
        today_queue.append(
            _queue_row(
                category="Artifact verification",
                target=str(row.get("target", "")) or str(row.get("run", "")),
                action_now="Fix the missing or mismatched artifacts",
                why=str(row.get("summary", "")) or "This target is missing expected files or has an output-integrity mismatch.",
                open_view=open_view,
            )
        )
        blocked_items.append(
            {
                "scope": "Artifact verification",
                "target": str(row.get("target", "")) or str(row.get("run", "")),
                "blocker": str(row.get("summary", "")),
                "what_to_do": str(row.get("next_move", "")),
            }
        )

    waiting_on_md: list[dict[str, str]] = []
    waiting_on_review: list[dict[str, str]] = []
    attention_items: list[dict[str, str]] = []
    for ladder in peptide_ladders:
        next_step = ladder.get("next_step", {}) if isinstance(ladder.get("next_step"), dict) else {}
        ladder_target = str(ladder.get("sequence", ""))
        if ladder["next_profile"]:
            waiting_on_md.append(
                {
                    "sequence": ladder["sequence"],
                    "next_stage": ladder.get("next_profile_label", ladder["next_profile"]),
                    "next_action": str(next_step.get("title", "")),
                    "sync_status": ladder.get("sync_status", ""),
                }
            )
        if ladder["ready_for_review"]:
            full_item = ladder.get("full")
            if full_item and str(full_item.get("cgmd_label", "")).strip() in {"0", "1"} and not bool(ladder.get("ingest_supported", True)):
                if bool(ladder.get("promotion_available", False)):
                    today_queue.append(
                        _queue_row(
                            category="AL promotion",
                            target=ladder_target,
                            action_now="Promote this reviewed result into the real AL batch",
                            why="A real proposed batch now contains this peptide, so the reporting-only MD result can be rebound and made ingest-ready.",
                            open_view="MD Validation",
                        )
                    )
            elif full_item and str(full_item.get("cgmd_label", "")).strip() in {"0", "1"}:
                today_queue.append(
                    _queue_row(
                        category="Ingest",
                        target=ladder_target,
                        action_now="Create the ingest CSV",
                        why="This peptide already has a reviewed full-analysis label and is one step away from model ingest.",
                        open_view="MD Validation",
                    )
                )
            else:
                today_queue.append(
                    _queue_row(
                        category="Review",
                        target=ladder_target,
                        action_now="Review the full-analysis result",
                        why="The full MD analysis is finished, but it still needs a human `cgmd_label` before ingest is possible.",
                        open_view="MD Validation",
                    )
                )
        elif str(ladder.get("sync_status", "")) in {"submitted", "running"} and str(ladder.get("remote_job_id", "")):
            today_queue.append(
                _queue_row(
                    category="Remote monitoring",
                    target=ladder_target,
                    action_now="Poll the BURA queue",
                    why="The peptide already has a tracked remote BURA job, so monitoring is the next safe step.",
                    open_view="MD Validation",
                )
            )
        elif ladder["next_profile"]:
            today_queue.append(
                _queue_row(
                    category="MD preparation",
                    target=ladder_target,
                    action_now=str(next_step.get("title", "")),
                    why=str(next_step.get("summary", "")),
                    open_view="MD Validation",
                )
            )
        current = ladder.get("current")
        if current and current.get("job_root_status") in {"package_prepared", "sasa_complete", "pdb_missing"}:
            blocked_items.append(
                {
                    "scope": "MD validation",
                    "target": str(ladder["sequence"]),
                    "blocker": _friendly_md_status(str(current.get("job_root_status", ""))),
                    "what_to_do": str(next_step.get("summary", "")),
                }
            )
            attention_items.append(
                {
                    "sequence": ladder["sequence"],
                    "campaign": current.get("campaign", ""),
                    "job_root_status": _friendly_md_status(str(current.get("job_root_status", ""))),
                    "sync_status": current.get("sync_status", ""),
                    "next_action": str(next_step.get("summary", "")),
                }
            )

    review_pipeline = list(peptide_inventory.get("review_pipeline", [])) if isinstance(peptide_inventory.get("review_pipeline", []), list) else []
    waiting_on_review = [
        {
            "sequence": str(row.get("sequence", "")),
            "run": str(row.get("run", "")),
            "campaign": str(row.get("campaign", "")),
            "review_state": str(row.get("review_state", "")),
            "next_action": str(row.get("next_action", "")),
        }
        for row in review_pipeline[:12]
    ]

    today_queue = sorted(
        today_queue,
        key=lambda row: _today_queue_priority(
            category=str(row.get("category", "")),
            target=str(row.get("target", "")),
        ),
    )[:18]

    active_local_actions = [
        _summarize_action(action)
        for action in actions
        if action.get("status") in ACTIVE_ACTION_STATUSES and action.get("scope") == "local"
    ]
    active_remote_actions = [
        _summarize_action(action)
        for action in actions
        if action.get("status") in ACTIVE_ACTION_STATUSES and action.get("scope") in {"supek", "bura"}
    ]
    approval_queue = [
        _summarize_action(action)
        for action in actions
        if action.get("status") in APPROVAL_PENDING_STATUSES
    ]
    recent_failures = [
        _summarize_action(action)
        for action in actions
        if action.get("status") in {"failed", "cancelled"}
    ][:10]

    return {
        "run_count": len(run_summaries),
        "peptide_count": len(peptide_ladders),
        "latest_proposed_peptides": latest_proposals[:15],
        "feedback_queue": feedback_queue_rows[:10],
        "md_slate_exceptions": visible_md_slate_exceptions[:12],
        "remote_reconciliation": visible_remote_reconciliation[:12],
        "remote_reconciliation_summary": build_remote_reconciliation_summary(remote_reconciliation),
        "waiting_on_md": waiting_on_md,
        "waiting_on_review": waiting_on_review,
        "attention_items": attention_items,
        "blocked_items": blocked_items,
        "today_queue": today_queue,
        "peptide_inventory": peptide_inventory,
        "active_local_actions": active_local_actions,
        "active_remote_actions": active_remote_actions,
        "approval_queue": approval_queue,
        "recent_failures": recent_failures,
        "cluster_summary": _frame_records(remote_job_summary_frame(snapshots)),
        "sync_records": sync_records,
        "md_slates": md_slates,
        "active_md_slates": active_md_slates,
        "bura_utilization": bura_utilization,
        "artifact_verification": artifact_verification,
        "artifact_verification_summary": build_artifact_verification_summary(artifact_verification),
    }



def collect_dashboard_state(run_root: Path) -> dict[str, object]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    run_summaries = discover_dashboard_run_summaries(run_root, summarize_run)
    records = load_dashboard_state_records(run_root, run_summaries)
    curation = records["curation"]
    actions = records["actions"]
    md_slates = records["md_slates"]
    md_slate_planners = records["md_slate_planners"]
    al_loop_simulations = records["al_loop_simulations"]
    md_slate_exceptions = records["md_slate_exceptions"]
    decisions = records["decisions"]
    progress_events = records["progress_events"]
    profiles = records["profiles"]
    snapshots = records["snapshots"]
    sync_records = records["sync_records"]
    cluster_health = records["cluster_health"]
    _attach_remote_state(run_summaries, sync_records)
    _decorate_runs_with_curation(run_summaries, curation)

    for run in run_summaries:
        run["dashboard_actions"] = _actions_for_run(actions, str(run["run_dir"]))

    peptide_ladders = build_peptide_ladders(run_summaries)
    for ladder in peptide_ladders:
        ladder["dashboard_actions"] = _actions_for_ladder(actions, ladder)

    peptide_inventory = build_peptide_inventory(run_root, run_summaries, peptide_ladders, decisions)
    bura_utilization = build_bura_utilization_summary(run_root, actions)
    artifact_verification = build_artifact_verification_rows(run_summaries)
    remote_reconciliation = build_remote_reconciliation_rows(
        run_summaries=run_summaries,
        md_slates=md_slates,
        sync_records=sync_records,
        snapshots=snapshots,
        actions=actions,
    )
    ignored_reconciliation_ids = {
        str(item).strip()
        for item in curation.get("ignored_reconciliation_ids", [])
        if str(item).strip()
    }
    if ignored_reconciliation_ids:
        remote_reconciliation = [
            row
            for row in remote_reconciliation
            if str(row.get("reconciliation_id", "")).strip() not in ignored_reconciliation_ids
        ]
    remote_watchdog = build_remote_watchdog_rows(
        run_summaries=run_summaries,
        md_slates=md_slates,
        sync_records=sync_records,
        snapshots=snapshots,
        cluster_health=cluster_health,
        remote_reconciliation=remote_reconciliation,
        artifact_verification=artifact_verification,
    )
    overview = build_overview(
        run_summaries,
        peptide_ladders,
        actions,
        snapshots,
        sync_records,
        peptide_inventory,
        md_slates,
        md_slate_exceptions,
        remote_reconciliation,
        bura_utilization,
        artifact_verification,
    )
    notifications = _build_dashboard_notifications(
        run_root,
        actions=actions,
        runs=run_summaries,
        peptide_ladders=peptide_ladders,
        md_slates=md_slates,
        md_slate_exceptions=md_slate_exceptions,
        remote_reconciliation=remote_reconciliation,
        artifact_verification=artifact_verification,
        peptide_inventory=peptide_inventory,
        overview=overview,
        cluster_health_rows=_cluster_health_rows({"cluster_health": cluster_health}),
    )

    state = {
        "run_root": str(run_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": run_summaries,
        "all_runs": deepcopy(run_summaries),
        "peptides": peptide_ladders,
        "peptide_inventory": peptide_inventory,
        "actions": actions,
        "md_slates": md_slates,
        "md_slate_planners": md_slate_planners,
        "al_loop_simulations": al_loop_simulations,
        "md_slate_exceptions": md_slate_exceptions,
        "md_slate_exception_summary": build_md_slate_exception_summary(md_slate_exceptions),
        "remote_reconciliation": remote_reconciliation,
        "remote_reconciliation_summary": build_remote_reconciliation_summary(remote_reconciliation),
        "remote_watchdog": remote_watchdog,
        "remote_watchdog_summary": build_remote_watchdog_summary(remote_watchdog),
        "artifact_verification": artifact_verification,
        "artifact_verification_summary": build_artifact_verification_summary(artifact_verification),
        "decisions": decisions,
        "progress_events": progress_events,
        "profiles": profiles,
        "profile_rows": profile_rows(profiles),
        "snapshots": snapshots,
        "sync_records": sync_records,
        "bura_utilization": bura_utilization,
        "cluster_health": cluster_health,
        "notifications": notifications,
        "curation": curation,
        "overview": overview,
    }
    return with_execution_readiness(state)



def apply_dashboard_filters(
    state: dict[str, object],
    *,
    workspace_scope: str = "Current Thesis Work",
    run_name: str = "All",
    sequence: str = "All",
    md_profile: str = "All",
    status: str = "All",
) -> dict[str, object]:
    run_root = Path(str(state.get("run_root", "")))
    filtered_runs = deepcopy(list(state.get("runs", [])))
    has_pins = bool(state.get("curation", {}).get("pinned_runs", [])) if isinstance(state.get("curation", {}), dict) else False
    filtered_runs = [run for run in filtered_runs if _run_scope_matches(run, workspace_scope, has_pins=has_pins)]
    if run_name != "All":
        filtered_runs = [run for run in filtered_runs if _run_matches_selector(run, run_name)]

    if md_profile != "All" or status != "All":
        trimmed_runs: list[dict[str, object]] = []
        for run in filtered_runs:
            campaigns = list(run.get("md_campaigns", []))
            if md_profile != "All":
                campaigns = [item for item in campaigns if str(item.get("md_profile", "")) == md_profile]
            if status != "All":
                campaigns = [item for item in campaigns if str(item.get("job_root_status", "")) == status or str(item.get("sync_status", "")) == status]
            run["md_campaigns"] = campaigns
            trimmed_runs.append(run)
        filtered_runs = trimmed_runs

    filtered_actions = list(state.get("actions", []))
    if run_name != "All" or workspace_scope != "All Runs":
        run_keys = {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        filtered_actions = [
            action
            for action in filtered_actions
            if _canonical_path(str(action.get("related_run", ""))) in run_keys
            or any(_path_is_within(str(action.get("related_campaign", "")), run_key) for run_key in run_keys)
        ]
    if sequence != "All":
        filtered_actions = [
            action for action in filtered_actions if str(action.get("related_sequence", "")) in {"", sequence} or sequence in str(action.get("title", ""))
        ]
    if status != "All":
        filtered_actions = [action for action in filtered_actions if str(action.get("status", "")) == status]

    filtered_decisions = list(state.get("decisions", []))
    if run_name != "All" or workspace_scope != "All Runs":
        run_keys = {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        filtered_decisions = [
            decision
            for decision in filtered_decisions
            if not _canonical_path(str(decision.get("run_dir", "")))
            or _canonical_path(str(decision.get("run_dir", ""))) in run_keys
            or any(_path_is_within(str(decision.get("campaign_dir", "")), run_key) for run_key in run_keys)
        ]
    if sequence != "All":
        filtered_decisions = [
            decision
            for decision in filtered_decisions
            if str(decision.get("sequence", "")) in {"", sequence}
        ]

    filtered_progress_events = list(state.get("progress_events", []))
    if run_name != "All" or workspace_scope != "All Runs":
        run_keys = {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        filtered_progress_events = [
            entry
            for entry in filtered_progress_events
            if not _canonical_path(str(entry.get("run_dir", "")))
            or _canonical_path(str(entry.get("run_dir", ""))) in run_keys
            or any(_path_is_within(str(entry.get("campaign_dir", "")), run_key) for run_key in run_keys)
        ]
    if sequence != "All":
        filtered_progress_events = [
            entry
            for entry in filtered_progress_events
            if str(entry.get("sequence", "")) in {"", sequence}
        ]

    for run in filtered_runs:
        run["dashboard_actions"] = _actions_for_run(filtered_actions, str(run["run_dir"]))

    peptide_ladders = build_peptide_ladders(filtered_runs)
    if sequence != "All":
        peptide_ladders = [ladder for ladder in peptide_ladders if str(ladder.get("sequence", "")) == sequence]
    if md_profile != "All":
        peptide_ladders = [
            ladder
            for ladder in peptide_ladders
            if any(str(campaign.get("md_profile", "")) == md_profile for campaign in ladder.get("campaigns", []))
        ]
    if status != "All":
        peptide_ladders = [
            ladder
            for ladder in peptide_ladders
            if any(
                str(campaign.get("job_root_status", "")) == status or str(campaign.get("sync_status", "")) == status
                for campaign in ladder.get("campaigns", [])
            )
        ]

    for ladder in peptide_ladders:
        ladder["dashboard_actions"] = _actions_for_ladder(filtered_actions, ladder)

    peptide_inventory = build_peptide_inventory(run_root, filtered_runs, peptide_ladders, filtered_decisions)
    filtered_md_slates = list(state.get("md_slates", []))
    filtered_md_slate_planners = list(state.get("md_slate_planners", []))
    filtered_al_loop_simulations = list(state.get("al_loop_simulations", []))
    if run_name != "All" or workspace_scope != "All Runs":
        run_keys = {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        filtered_md_slates = [
            slate
            for slate in filtered_md_slates
            if not _canonical_path(str(slate.get("run_dir", "")))
            or _canonical_path(str(slate.get("run_dir", ""))) in run_keys
        ]
        filtered_md_slate_planners = [
            planner
            for planner in filtered_md_slate_planners
            if not _canonical_path(str(planner.get("run_dir", "")))
            or _canonical_path(str(planner.get("run_dir", ""))) in run_keys
        ]
        filtered_al_loop_simulations = [
            simulation
            for simulation in filtered_al_loop_simulations
            if not _canonical_path(str(simulation.get("run_dir", "")))
            or _canonical_path(str(simulation.get("run_dir", ""))) in run_keys
        ]
    if sequence != "All":
        filtered_md_slates = [
            {
                **slate,
                "peptides": [
                    peptide
                    for peptide in list(slate.get("peptides", []))
                    if str(peptide.get("sequence", "")) == sequence
                ],
            }
            for slate in filtered_md_slates
            if any(str(peptide.get("sequence", "")) == sequence for peptide in list(slate.get("peptides", [])))
        ]
        filtered_md_slate_planners = [
            {
                **planner,
                "candidates": [
                    candidate
                    for candidate in list(planner.get("candidates", []))
                    if str(candidate.get("sequence", "")) == sequence
                ],
            }
            for planner in filtered_md_slate_planners
            if any(str(candidate.get("sequence", "")) == sequence for candidate in list(planner.get("candidates", [])))
        ]
        filtered_al_loop_simulations = [
            {
                **simulation,
                "candidate_snapshot": [
                    candidate
                    for candidate in list(simulation.get("candidate_snapshot", []))
                    if str(candidate.get("sequence", "")) == sequence
                ],
                "simulated_review_labels": [
                    label
                    for label in list(simulation.get("simulated_review_labels", []))
                    if str(label.get("sequence", "")) == sequence
                ],
            }
            for simulation in filtered_al_loop_simulations
            if any(str(candidate.get("sequence", "")) == sequence for candidate in list(simulation.get("candidate_snapshot", [])))
        ]
    filtered_bura_utilization = build_bura_utilization_summary(
        Path(str(state["run_root"])),
        filtered_actions,
        run_dir=str(filtered_runs[0]["run_dir"]) if len(filtered_runs) == 1 else "",
    )
    filtered_md_slate_exceptions = [
        row
        for row in list(state.get("md_slate_exceptions", []))
        if (
            (run_name == "All" and workspace_scope == "All Runs")
            or not _canonical_path(str(row.get("run_dir", "")))
            or _canonical_path(str(row.get("run_dir", ""))) in {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        )
        and (sequence == "All" or str(row.get("sequence", "")) == sequence)
    ]
    filtered_exception_counts_by_slate: dict[str, int] = {}
    for row in filtered_md_slate_exceptions:
        slate_id = str(row.get("slate_id", "")).strip()
        if not slate_id:
            continue
        filtered_exception_counts_by_slate[slate_id] = filtered_exception_counts_by_slate.get(slate_id, 0) + 1
    filtered_md_slates = [
        {
            **slate,
            "exception_count": filtered_exception_counts_by_slate.get(str(slate.get("slate_id", "")).strip(), 0),
        }
        for slate in filtered_md_slates
    ]
    filtered_artifact_verification = [
        row
        for row in list(state.get("artifact_verification", []))
        if (
            (run_name == "All" and workspace_scope == "All Runs")
            or not _canonical_path(str(row.get("run_dir", "")))
            or _canonical_path(str(row.get("run_dir", ""))) in {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        )
        and (sequence == "All" or str(row.get("sequence", "")) in {"", sequence, "-"})
        and (
            status == "All"
            or str(row.get("sync_state", "")) == status
            or str(row.get("verification_state", "")) == status
        )
        and (
            md_profile == "All"
            or str(row.get("md_profile", "")) in {"", md_profile, "-"}
        )
    ]
    filtered_remote_reconciliation = filter_remote_reconciliation_rows(
        list(state.get("remote_reconciliation", [])),
        run_dirs={str(run.get("run_dir", "")) for run in filtered_runs},
        sequence=sequence,
        md_profile=md_profile,
        status=status,
    )
    filtered_remote_watchdog = filter_remote_watchdog_rows(
        list(state.get("remote_watchdog", [])),
        run_dirs={str(run.get("run_dir", "")) for run in filtered_runs},
        sequence=sequence,
        md_profile=md_profile,
        status=status,
    )

    filtered_state = dict(state)
    filtered_state["runs"] = filtered_runs
    filtered_state["peptides"] = peptide_ladders
    filtered_state["peptide_inventory"] = peptide_inventory
    filtered_state["actions"] = filtered_actions
    filtered_state["md_slates"] = filtered_md_slates
    filtered_state["md_slate_planners"] = filtered_md_slate_planners
    filtered_state["al_loop_simulations"] = filtered_al_loop_simulations
    filtered_state["md_slate_exceptions"] = filtered_md_slate_exceptions
    filtered_state["md_slate_exception_summary"] = build_md_slate_exception_summary(filtered_md_slate_exceptions)
    filtered_state["remote_reconciliation"] = filtered_remote_reconciliation
    filtered_state["remote_reconciliation_summary"] = build_remote_reconciliation_summary(filtered_remote_reconciliation)
    filtered_state["remote_watchdog"] = filtered_remote_watchdog
    filtered_state["remote_watchdog_summary"] = build_remote_watchdog_summary(filtered_remote_watchdog)
    filtered_state["artifact_verification"] = filtered_artifact_verification
    filtered_state["artifact_verification_summary"] = build_artifact_verification_summary(filtered_artifact_verification)
    filtered_state["decisions"] = filtered_decisions
    filtered_state["progress_events"] = filtered_progress_events
    filtered_state["bura_utilization"] = filtered_bura_utilization
    filtered_overview = build_overview(
        filtered_runs,
        peptide_ladders,
        filtered_actions,
        list(state.get("snapshots", [])),
        list(state.get("sync_records", [])),
        peptide_inventory,
        filtered_md_slates,
        filtered_md_slate_exceptions,
        filtered_remote_reconciliation,
        filtered_bura_utilization,
        filtered_artifact_verification,
    )
    filtered_state["overview"] = filtered_overview
    filtered_notifications = list(state.get("notifications", []))
    if run_name != "All" or workspace_scope != "All Runs":
        run_keys = {_canonical_path(run.get("run_dir", "")) for run in filtered_runs}
        filtered_notifications = [
            item
            for item in filtered_notifications
            if not _canonical_path(str(item.get("run_dir", "")))
            or _canonical_path(str(item.get("run_dir", ""))) in run_keys
        ]
    if sequence != "All":
        filtered_notifications = [
            item
            for item in filtered_notifications
            if str(item.get("sequence", "")) in {"", sequence}
        ]
    filtered_state["notifications"] = filtered_notifications
    filtered_state["workspace_scope"] = workspace_scope
    return with_execution_readiness(filtered_state)



def _render_autorefresh(st, refresh_seconds: int) -> None:
    if refresh_seconds <= 0:
        return
    st.info(
        "Timed auto refresh is temporarily disabled in safe mode because browser-level reloads were causing dimmed and duplicated Streamlit views on Windows. Use 'Refresh now' when you want to reload the dashboard."
    )


def _dashboard_session_state(st) -> dict[str, object]:
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        session_state = {}
        try:
            setattr(st, "session_state", session_state)
        except Exception:
            pass
    return session_state



def _query_param_get(st, key: str, default: str) -> str:
    query_params = getattr(st, "query_params", None)
    if query_params is None:
        return default
    try:
        value = query_params.get(key, default)
    except Exception:
        return default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value not in {None, ""} else default



def _query_param_set(st, key: str, value: str) -> None:
    query_params = getattr(st, "query_params", None)
    if query_params is None:
        return
    target = str(value)
    if _query_param_get(st, key, "") == target:
        return
    try:
        query_params[key] = target
    except Exception:
        try:
            query_params.update({key: target})
        except Exception:
            return



def _queue_query_param_update(st, key: str, value: str) -> None:
    target = str(value)
    if _query_param_get(st, key, "") == target:
        return
    session_state = _dashboard_session_state(st)
    pending = session_state.get("_dashboard_query_updates")
    if not isinstance(pending, dict):
        pending = {}
    pending[str(key)] = target
    session_state["_dashboard_query_updates"] = pending



def _flush_query_param_updates(st) -> None:
    session_state = _dashboard_session_state(st)
    pending = session_state.pop("_dashboard_query_updates", None)
    if not isinstance(pending, dict) or not pending:
        return

    updates = {
        str(key): str(value)
        for key, value in pending.items()
        if _query_param_get(st, str(key), "") != str(value)
    }
    if not updates:
        return

    query_params = getattr(st, "query_params", None)
    if query_params is None:
        return
    try:
        query_params.update(updates)
        return
    except Exception:
        pass
    for key, value in updates.items():
        _query_param_set(st, key, value)



def _persisted_choice(
    st,
    widget,
    *,
    label: str,
    options: list[str],
    key: str,
    query_key: str,
    default: str,
    read_query: bool = True,
    write_query: bool = True,
) -> str:
    if not options:
        return ""
    session_state = _dashboard_session_state(st)
    stored = str(session_state.get(key, ""))
    if stored not in options:
        query_value = _query_param_get(st, query_key, "") if read_query else ""
        initial = default if default in options else options[0]
        if query_value in options:
            initial = query_value
        session_state[key] = initial
    selected = widget(label, options, key=key)
    if write_query:
        _queue_query_param_update(st, query_key, selected)
    return selected



def _persisted_refresh_seconds(st, default_seconds: int) -> int:
    options = [0, 10, 30, 60]
    labels = {0: "Off", 10: "10s", 30: "30s", 60: "60s"}
    reverse = {value: key for key, value in labels.items()}
    default_label = labels.get(default_seconds, "Off")
    selected_label = _persisted_choice(
        st,
        st.sidebar.selectbox,
        label="Auto refresh",
        options=list(labels.values()),
        key="dashboard_refresh_label",
        query_key="refresh",
        default=default_label,
        write_query=False,
    )
    return reverse.get(selected_label, 0)



def _trigger_dashboard_rerun(st) -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()



def _stash_dashboard_flash(st, level: str, message: str) -> None:
    session_state = _dashboard_session_state(st)
    session_state["_dashboard_flash"] = {"level": level, "message": message}



def _render_dashboard_flash(st) -> None:
    session_state = _dashboard_session_state(st)
    flash = session_state.pop("_dashboard_flash", None)
    if not isinstance(flash, dict):
        return
    message = str(flash.get("message", "")).strip()
    if not message:
        return
    level = str(flash.get("level", "info"))
    method = getattr(st, level, None)
    if callable(method):
        method(message)
    else:
        st.write(message)



def _report_action_result(st, action: dict[str, object]) -> None:
    if action.get("status") in APPROVAL_PENDING_STATUSES:
        message = f"Created draft action {action['id']}: {action['title']}"
    elif action.get("status") in FINAL_ACTION_STATUSES:
        message = f"Finished action {action['id']}: {action['title']}"
    else:
        message = f"Queued action {action['id']}: {action['title']}"
    _stash_dashboard_flash(st, "success", message)
    _sync_navigation_query_params(st)
    _trigger_dashboard_rerun(st)


def _sync_navigation_query_params(st) -> None:
    session_state = _dashboard_session_state(st)
    navigation_keys = {
        "dashboard_view": "view",
        "dashboard_workspace_scope": "workspace",
        "dashboard_ui_mode": "ui_mode",
        "dashboard_workflow_mode": "workflow_mode",
        "dashboard_approval_mode": "approval_mode",
        "dashboard_refresh_mode": "refresh_mode",
        "dashboard_run_detail_name": "run_detail",
        "dashboard_model_section": "model_section",
        "dashboard_md_section": "md_section",
        "dashboard_operations_section": "operations_section",
        "dashboard_peptides_section": "peptides_section",
        "dashboard_results_section": "results_section",
        "dashboard_peptide_sequence": "peptide",
    }
    updates = {
        query_key: str(session_state.get(session_key, ""))
        for session_key, query_key in navigation_keys.items()
        if str(session_state.get(session_key, "")).strip()
    }
    if not updates:
        return
    query_params = getattr(st, "query_params", None)
    if query_params is None:
        return
    try:
        query_params.update(updates)
        return
    except Exception:
        pass
    for key, value in updates.items():
        _query_param_set(st, key, value)


def _readiness_badge_markup(verdict: str) -> str:
    palette = {
        "Ready": ("#0f5132", "#d1e7dd", "#badbcc"),
        "Ready with caution": ("#664d03", "#fff3cd", "#ffecb5"),
        "Blocked": ("#842029", "#f8d7da", "#f5c2c7"),
    }
    text_color, background, border = palette.get(verdict, ("#495057", "#e9ecef", "#ced4da"))
    label = verdict or "Unknown"
    return (
        "<div style=\"margin:0.1rem 0 0.45rem 0;\">"
        f"<span style=\"display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;"
        f"background:{background};border:1px solid {border};color:{text_color};font-size:0.8rem;font-weight:700;\">"
        f"Execution readiness: {label}"
        "</span></div>"
    )


def _render_compact_info_badge(st, *, label: str, lines: list[str], tone: str = "blue") -> None:
    clean_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not clean_lines:
        return
    palette = {
        "blue": ("#e0f2fe", "#075985", "#bae6fd"),
        "slate": ("#f1f5f9", "#334155", "#cbd5e1"),
    }
    background, color, border = palette.get(tone, palette["blue"])
    tooltip = html.escape("\n".join(clean_lines), quote=True).replace("\n", "&#10;")
    safe_label = html.escape(label, quote=False)
    st.markdown(
        (
            "<span "
            f"title=\"{tooltip}\" "
            f"style=\"display:inline-flex;align-items:center;justify-content:center;width:1.1rem;height:1.1rem;"
            f"border-radius:999px;background:{background};color:{color};font-size:0.78rem;font-weight:800;"
            f"border:1px solid {border};cursor:help;margin-right:0.35rem;\">i</span>"
            f"<span style=\"font-size:0.82rem;color:#475569;\">{safe_label}</span>"
        ),
        unsafe_allow_html=True,
    )


def _render_action_contract_summary(st, contract_id: str, readiness: dict[str, object] | None = None) -> None:
    render_action_contract_compact(st, contract_id, readiness=readiness)


def _action_form_container(st, *, key: str):
    form = getattr(st, "form", None)
    if callable(form):
        try:
            return form(key=key, clear_on_submit=False, border=False)
        except TypeError:
            return form(key=key, clear_on_submit=False)
    return st.container()


def _action_form_submit_button(st, label: str, *, key: str, disabled: bool = False) -> bool:
    form_submit_button = getattr(st, "form_submit_button", None)
    if callable(form_submit_button):
        return bool(form_submit_button(label, disabled=disabled))
    return bool(st.button(label, key=key, disabled=disabled))


def _latest_dashboard_action_for_kind(state: dict[str, object], action_kind: str) -> dict[str, object] | None:
    actions = state.get("actions", [])
    if not isinstance(actions, list):
        return None
    for action in actions:
        if isinstance(action, dict) and str(action.get("kind", "")) == str(action_kind):
            return action
    return None


def _render_action_owner_mirror(
    st,
    *,
    state: dict[str, object],
    label: str,
    action_kind: str,
    key_prefix: str,
) -> None:
    hint = canonical_navigation_hint(action_kind)
    target_view = hint.get("view", "")
    target_section = hint.get("section", "")
    st.info(
        f"{label} is available from {target_view} -> {target_section}. "
        "Guided mode keeps one executable home per thesis action so you do not have to wonder which duplicate button is correct."
    )
    latest = _latest_dashboard_action_for_kind(state, action_kind)
    if latest:
        st.caption(
            "Latest matching action: "
            f"{latest.get('kind', action_kind)} | {latest.get('status', 'unknown')} | "
            f"{latest.get('started_at', latest.get('created_at', ''))}"
        )
    summary = hint.get("summary", "")
    if summary:
        st.caption(summary)
    if target_view and target_section and st.button(
        f"Open {target_view} -> {target_section}",
        key=f"open_owner_{key_prefix}_{action_kind}",
    ):
        session_state = _dashboard_session_state(st)
        session_state["dashboard_view"] = target_view
        section_key = view_section_session_key(target_view)
        if section_key:
            session_state[section_key] = target_section
        _queue_query_param_update(st, "view", target_view)
        section_query_key = view_section_query_key(target_view)
        if section_query_key:
            _queue_query_param_update(st, section_query_key, target_section)
        _trigger_dashboard_rerun(st)
    st.divider()


def _guided_action_is_mirror(
    *,
    state: dict[str, object] | None,
    action_kind: str,
    view: str,
    section: str,
) -> bool:
    if not state or not action_kind:
        return False
    if not _dashboard_guided_mode(state):
        return False
    if not view or not section:
        return False
    return not is_canonical_context(action_kind, view, section)



def _render_launch_action(
    st,
    *,
    label: str,
    command: str,
    key_prefix: str,
    on_submit,
    after_submit=None,
    button_text: str = "Run now",
    what: str = "",
    when: str = "",
    produces: str = "",
    next_step: str = "",
    contract_id: str = "",
    readiness: dict[str, object] | None = None,
    action_kind: str = "",
    state: dict[str, object] | None = None,
    view: str = "",
    section: str = "",
) -> None:
    owner_kind = action_kind or contract_id
    if _guided_action_is_mirror(state=state, action_kind=owner_kind, view=view, section=section):
        _render_action_owner_mirror(st, state=state or {}, label=label, action_kind=owner_kind, key_prefix=key_prefix)
        return
    with st.container():
        st.write(label)
        _render_action_guidance(st, what=what, when=when, produces=produces, next_step=next_step)
        if contract_id:
            _render_action_contract_summary(st, contract_id, readiness=readiness)
        button_disabled = bool(readiness and readiness.get("disable_button", False))
        if button_disabled:
            st.info("This action is currently blocked. Clear the blocker above and the button will re-enable.")
        st.code(command, language="bash")
        with _action_form_container(st, key=f"run_form_{key_prefix}"):
            confirmed = st.checkbox(f"Confirm {label.lower()}", key=f"confirm_{key_prefix}")
            submitted = _action_form_submit_button(
                st,
                button_text,
                key=f"run_{key_prefix}",
                disabled=button_disabled,
            )
        if submitted:
            if not confirmed:
                st.warning("Confirm the command first, then run it.")
                return
            try:
                action = on_submit()
            except Exception as exc:
                st.error(str(exc))
                return
            if after_submit is not None:
                try:
                    after_submit(action)
                except Exception as exc:
                    st.warning(f"Progress memory could not be updated: {exc}")
            _report_action_result(st, action)
        st.divider()



def _render_draft_action(
    st,
    *,
    label: str,
    command: str,
    key_prefix: str,
    on_submit,
    after_submit=None,
    what: str = "",
    when: str = "",
    produces: str = "",
    next_step: str = "",
    contract_id: str = "",
    readiness: dict[str, object] | None = None,
    action_kind: str = "",
    state: dict[str, object] | None = None,
    view: str = "",
    section: str = "",
) -> None:
    owner_kind = action_kind or contract_id
    if _guided_action_is_mirror(state=state, action_kind=owner_kind, view=view, section=section):
        _render_action_owner_mirror(st, state=state or {}, label=label, action_kind=owner_kind, key_prefix=key_prefix)
        return
    with st.container():
        st.write(label)
        _render_action_guidance(st, what=what, when=when, produces=produces, next_step=next_step)
        if contract_id:
            _render_action_contract_summary(st, contract_id, readiness=readiness)
        button_disabled = bool(readiness and readiness.get("disable_button", False))
        if button_disabled:
            st.info("This action is currently blocked. Clear the blocker above and the draft button will re-enable.")
        st.code(command, language="bash")
        with _action_form_container(st, key=f"draft_form_{key_prefix}"):
            confirmed = st.checkbox(f"Confirm {label.lower()}", key=f"draft_confirm_{key_prefix}")
            submitted = _action_form_submit_button(
                st,
                "Create draft",
                key=f"draft_{key_prefix}",
                disabled=button_disabled,
            )
        if submitted:
            if not confirmed:
                st.warning("Confirm the command first, then create the draft action.")
                return
            try:
                action = on_submit()
            except Exception as exc:
                st.error(str(exc))
                return
            if after_submit is not None:
                try:
                    after_submit(action)
                except Exception as exc:
                    st.warning(f"Progress memory could not be updated: {exc}")
            _report_action_result(st, action)
        st.divider()


def _latest_action_for_kinds(actions: list[dict[str, object]], kinds: set[str]) -> dict[str, object] | None:
    for action in actions:
        if str(action.get("kind", "")) in kinds:
            return action
    return None


def _remote_console_hint(action: dict[str, object]) -> str:
    status = str(action.get("status", ""))
    kind = str(action.get("kind", ""))
    if status == "failed":
        if kind in {"supek-submit-workflow", "bura-submit-chain"}:
            return "The latest remote submission failed. Check stderr first, then rerun the readiness check before submitting again."
        if kind in {"supek-poll-qstat", "bura-poll-squeue"}:
            return "The latest queue poll failed. Re-check cluster health and the tracked remote job id."
        if kind in {"supek-submit-preflight", "bura-submit-readiness"}:
            return "The readiness check found a missing path, script, or environment dependency. Fix that before the next submit."
    if status in ACTIVE_ACTION_STATUSES:
        return "This remote action is still active. Refresh or fetch logs again once you want a newer snapshot."
    if kind in {"supek-fetch-logs", "bura-inspect-logs"}:
        return "These excerpts are the fastest way to sanity-check a live or recently finished remote job."
    if kind in {"supek-poll-qstat", "bura-poll-squeue"}:
        return "Use this queue snapshot to decide whether to keep waiting, fetch logs, cancel, or pull results back."
    return "Use this snapshot as the latest remote-control breadcrumb for the current context."


def _remote_job_id_for_console(action: dict[str, object]) -> str:
    direct_job_id = str(action.get("remote_job_id", "")).strip()
    if direct_job_id:
        return direct_job_id
    metadata = action.get("metadata")
    if isinstance(metadata, dict):
        metadata_job_id = str(metadata.get("remote_job_id", "")).strip()
        if metadata_job_id:
            return metadata_job_id
        remote_job_ids = metadata.get("remote_job_ids")
        if isinstance(remote_job_ids, list) and remote_job_ids:
            return str(remote_job_ids[0]).strip()
    return ""


def _remote_console_job_summary(action: dict[str, object], stdout_text: str) -> dict[str, str] | None:
    kind = str(action.get("kind", ""))
    job_id = _remote_job_id_for_console(action)
    if kind == "supek-poll-qstat":
        jobs = parse_qstat_output(stdout_text)
        if not jobs:
            return None
        matched = next((job for job in jobs if job_id and str(job.get("job_id", "")) == job_id), None)
        if matched is None and len(jobs) == 1:
            matched = jobs[0]
        if matched is None:
            if job_id:
                return {
                    "level": "info",
                    "title": "Tracked SUPEK job is no longer in qstat",
                    "summary": f"{job_id} is not visible in the latest queue snapshot.",
                    "next_step": "Fetch the latest logs or pull artifacts back to confirm whether the job finished or failed before leaving the queue.",
                }
            return None
        state = str(matched.get("state", "")).upper()
        if state == "H":
            return {
                "level": "warning",
                "title": "Tracked SUPEK job is held",
                "summary": f"{matched.get('job_id', 'The tracked job')} is held by PBS instead of progressing.",
                "next_step": "Inspect the latest wrapper/stdout/stderr first. If it stays held, run qstat -f on SUPEK for the exact hold reason, then cancel and resubmit once the launch issue is fixed.",
            }
        if state == "Q":
            return {
                "level": "info",
                "title": "Tracked SUPEK job is queued",
                "summary": f"{matched.get('job_id', 'The tracked job')} is waiting in the PBS queue.",
                "next_step": "Keep polling or fetch logs again if you expect the wrapper to have started writing output.",
            }
        if state == "R":
            return {
                "level": "success",
                "title": "Tracked SUPEK job is running",
                "summary": f"{matched.get('job_id', 'The tracked job')} is actively running on SUPEK.",
                "next_step": "Keep polling the queue or fetch the latest remote logs when you want a fresher execution snapshot.",
            }
        return {
            "level": "info",
            "title": "Tracked SUPEK job has a scheduler state to inspect",
            "summary": f"{matched.get('job_id', 'The tracked job')} is currently reported as {state or 'unknown'}.",
            "next_step": "Use the latest queue snapshot together with remote logs to decide whether to wait, cancel, or pull results back.",
        }

    if kind == "bura-poll-squeue":
        jobs = parse_squeue_output(stdout_text)
        if not jobs:
            return None
        matched = next((job for job in jobs if job_id and str(job.get("job_id", "")) == job_id), None)
        if matched is None and len(jobs) == 1:
            matched = jobs[0]
        if matched is None:
            if job_id:
                return {
                    "level": "info",
                    "title": "Tracked BURA job is no longer in squeue",
                    "summary": f"{job_id} is not visible in the latest Slurm snapshot.",
                    "next_step": "Fetch the latest logs or pull the package back to confirm whether the chain completed or failed before leaving the queue.",
                }
            return None
        state = str(matched.get("state", "")).upper()
        reason = str(matched.get("reason", "")).strip()
        if state == "PD":
            lowered_reason = reason.lower()
            if any(token in lowered_reason for token in ("dependencynever", "held", "dependency")):
                return {
                    "level": "warning",
                    "title": "Tracked BURA job is pending on a blocking dependency",
                    "summary": f"{matched.get('job_id', 'The tracked job')} is still pending in Slurm because `{reason or 'a dependency is not satisfied'}`.",
                    "next_step": "Check the earlier job in the chain or the latest BURA stderr before resubmitting. This usually means the dependency job failed or never materialized.",
                }
            return {
                "level": "info",
                "title": "Tracked BURA job is queued",
                "summary": f"{matched.get('job_id', 'The tracked job')} is pending in Slurm{f' with reason `{reason}`' if reason else ''}.",
                "next_step": "Keep polling if this is normal queue wait, or fetch the latest logs if you expected execution to start already.",
            }
        if state == "R":
            return {
                "level": "success",
                "title": "Tracked BURA job is running",
                "summary": f"{matched.get('job_id', 'The tracked job')} is actively running on BURA.",
                "next_step": "Keep polling or inspect the latest logs when you want a fresher snapshot of the active chain.",
            }
        return {
            "level": "info",
            "title": "Tracked BURA job has a scheduler state to inspect",
            "summary": f"{matched.get('job_id', 'The tracked job')} is currently reported as {state or 'unknown'}{f' with reason `{reason}`' if reason else ''}.",
            "next_step": "Use this together with the latest logs to decide whether to keep waiting, cancel the chain, or finalize outputs.",
        }
    return None


def _remote_console_issue_summary(action: dict[str, object], stdout_text: str, stderr_text: str) -> dict[str, str] | None:
    combined_text = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()
    lowered = combined_text.lower()
    if not lowered:
        return _remote_console_job_summary(action, stdout_text)

    missing_preflight_tokens = {
        "repo_missing": "remote repo path",
        "repo_git_missing": "git checkout metadata",
        "conda_init_missing": "conda init script",
        "scheduler_cmd_missing": "scheduler command",
        "scratch_root_missing": "scratch root",
        "run_state_missing": "staged run directory",
        "log_root_missing": "log root",
        "python_import_missing": "activated Python environment import",
        "campaign_dir_missing": "remote campaign directory",
        "preflight_script_missing": "preflight script",
        "submit_script_missing": "submit script",
        "preflight_syntax_missing": "preflight script syntax",
        "submit_syntax_missing": "submit script syntax",
        "package_dir_missing": "package directory",
        "dos2unix_missing": "dos2unix helper",
        "module_load_missing": "module load command",
    }
    missing_hits = [label for token, label in missing_preflight_tokens.items() if token in lowered]
    if missing_hits:
        return {
            "level": "warning",
            "title": "Readiness check found missing submit dependencies",
            "summary": "The latest readiness/preflight check could not confirm: " + ", ".join(missing_hits[:4]) + (", ..." if len(missing_hits) > 4 else "") + ".",
            "next_step": "Fix the missing dependency first, then rerun the readiness/preflight check before you submit the next remote step.",
        }

    if "job held, too many failed attempts to run" in lowered:
        return {
            "level": "warning",
            "title": "Scheduler hold detected",
            "summary": "PBS retried the SUPEK job several times and then placed it on hold after repeated launch failures.",
            "next_step": "Inspect the latest wrapper/stdout/stderr for the real launch problem, then cancel and resubmit once that issue is fixed.",
        }
    if "could not read username for 'https://github.com'" in lowered:
        return {
            "level": "error",
            "title": "Git remote still needs interactive HTTPS auth",
            "summary": "The remote repository is still using an HTTPS GitHub origin, so the non-interactive dashboard action cannot authenticate.",
            "next_step": "Switch the remote repo to an SSH GitHub origin and re-run the sync once the cluster key is working.",
        }
    if "permission denied (publickey)" in lowered:
        if "github.com" in lowered:
            return {
                "level": "error",
                "title": "GitHub SSH authentication failed",
                "summary": "The cluster could reach GitHub, but GitHub rejected the SSH key for this repo action.",
                "next_step": "Check the deploy key or cluster SSH config, then retry the repo or log action once GitHub SSH works non-interactively.",
            }
        return {
            "level": "error",
            "title": "SSH authentication failed",
            "summary": "The remote action could not authenticate with the cluster over SSH.",
            "next_step": "Re-check cluster health, confirm the right key is loaded in ssh-agent, and retry once non-interactive SSH works again.",
        }
    if "the agent has no identities" in lowered or "could not open a connection to your authentication agent" in lowered:
        return {
            "level": "warning",
            "title": "OpenSSH agent is not ready",
            "summary": "The remote action could not find a usable identity in the local OpenSSH agent.",
            "next_step": "Start ssh-agent, run ssh-add with the intended private key, then re-check cluster health before retrying the remote action.",
        }
    if "path canonicalization failed" in lowered or ("realpath" in lowered and "no such file" in lowered):
        return {
            "level": "error",
            "title": "Remote path resolution failed",
            "summary": "The remote transfer or command hit a path that does not exist or cannot be canonicalized on the cluster side.",
            "next_step": "Re-run the readiness/preflight check and confirm the remote scratch, campaign, or log path exists before trying the next remote step.",
        }
    if "no such file or directory" in lowered:
        return {
            "level": "error",
            "title": "A required file or directory is missing",
            "summary": "The latest remote action refers to a file, folder, or wrapper path that was not present when the command ran.",
            "next_step": "Use the readiness check or latest wrapper/log paths to find the missing dependency, then retry once the path exists.",
        }
    if "broken pipe" in lowered or "connection reset" in lowered:
        return {
            "level": "warning",
            "title": "Transfer or SSH session was interrupted",
            "summary": "The connection dropped while the remote transfer or command was still in progress.",
            "next_step": "Retry once cluster connectivity looks healthy again. If it repeats, fetch logs or use a smaller staged payload before resubmitting.",
        }
    if "connection timed out" in lowered or "no route to host" in lowered or "could not resolve hostname" in lowered or "connection refused" in lowered:
        cluster = str(action.get("cluster", "")).strip().lower()
        if cluster == "bura":
            return {
                "level": "warning",
                "title": "BURA looks unreachable",
                "summary": "The remote action could not reach BURA non-interactively over SSH.",
                "next_step": "If FortiClient VPN is required, connect it first, then re-check BURA health before retrying the action.",
            }
        return {
            "level": "warning",
            "title": "Cluster connectivity problem detected",
            "summary": "The remote action could not complete because the SSH connection itself was not reachable or timed out.",
            "next_step": "Re-check cluster health, confirm the SSH alias resolves correctly, and retry once connectivity stabilizes.",
        }
    if "traceback" in lowered:
        return {
            "level": "warning",
            "title": "Remote command crashed",
            "summary": "The latest remote action emitted a Python traceback instead of cleanly finishing.",
            "next_step": "Inspect the stderr excerpt first, then adjust the run or environment before rerunning that step.",
        }

    return _remote_console_job_summary(action, stdout_text)


def _remote_console_reference_rows(action: dict[str, object]) -> list[dict[str, str]]:
    metadata = action.get("metadata", {}) if isinstance(action.get("metadata", {}), dict) else {}
    rows: list[dict[str, str]] = []
    for label, value in (
        ("Remote wrapper", metadata.get("remote_wrapper", "")),
        ("Remote stdout", metadata.get("remote_stdout", "")),
        ("Remote stderr", metadata.get("remote_stderr", "")),
        ("Remote campaign/run path", metadata.get("remote_path", "")),
        ("Tracked remote job", _remote_job_id_for_console(action)),
        ("Local stdout log", action.get("stdout_log", "")),
        ("Local stderr log", action.get("stderr_log", "")),
    ):
        text = str(value).strip()
        if not text:
            continue
        rows.append({"reference": label, "path_or_id": text})
    return rows


def _remote_console_history_rows(actions: list[dict[str, object]], kinds: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action in actions:
        if str(action.get("kind", "")) not in kinds:
            continue
        stdout_text = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=40)
        stderr_text = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=40)
        parsed_issue = _remote_console_issue_summary(action, stdout_text, stderr_text)
        if parsed_issue is None:
            continue
        level = str(parsed_issue.get("level", "info"))
        status = str(action.get("status", ""))
        title = str(parsed_issue.get("title", "")).strip()
        if status != "failed" and level not in {"warning", "error"}:
            continue
        rows.append(
            {
                "captured_at": str(action.get("finished_at", "")) or str(action.get("started_at", "")) or "-",
                "action": str(action.get("title", "")) or "-",
                "status": status or "-",
                "issue": title or "Parsed issue",
                "summary": str(parsed_issue.get("summary", "")).strip(),
                "next_move": str(parsed_issue.get("next_step", "")).strip(),
            }
        )
    return rows[:6]


def _matched_remote_job(action: dict[str, object], stdout_text: str) -> dict[str, str] | None:
    kind = str(action.get("kind", ""))
    job_id = _remote_job_id_for_console(action)
    if kind == "supek-poll-qstat":
        jobs = parse_qstat_output(stdout_text)
        matched = next((job for job in jobs if job_id and str(job.get("job_id", "")) == job_id), None)
        if matched is None and len(jobs) == 1:
            matched = jobs[0]
        return matched
    if kind == "bura-poll-squeue":
        jobs = parse_squeue_output(stdout_text)
        matched = next((job for job in jobs if job_id and str(job.get("job_id", "")) == job_id), None)
        if matched is None and len(jobs) == 1:
            matched = jobs[0]
        return matched
    return None


def _remote_console_scheduler_history_rows(actions: list[dict[str, object]], kinds: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action in actions:
        kind = str(action.get("kind", ""))
        if kind not in kinds or kind not in {"supek-poll-qstat", "bura-poll-squeue"}:
            continue
        stdout_text = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=60)
        matched = _matched_remote_job(action, stdout_text)
        if matched is None:
            continue
        state = str(matched.get("state", "")).upper()
        reason = str(matched.get("reason", "")).strip()
        queue_name = str(matched.get("queue", "")).strip() or str(matched.get("partition", "")).strip()
        rows.append(
            {
                "captured_at": str(action.get("finished_at", "")) or str(action.get("started_at", "")) or "-",
                "job_id": str(matched.get("job_id", "")) or _remote_job_id_for_console(action) or "-",
                "scheduler_state": state or "-",
                "queue_or_partition": queue_name or "-",
                "reason": reason or "-",
                "raw": str(matched.get("raw", "")).strip() or "-",
            }
        )
    return rows[:6]


def _render_remote_console(st, *, title: str, actions: list[dict[str, object]], kinds: set[str], key_prefix: str) -> None:
    st.markdown(f"#### {title}")
    action = _latest_action_for_kinds(actions, kinds)
    if action is None:
        st.info("No remote console snapshot is recorded yet for this context. Run a readiness check, queue poll, or log fetch first.")
        return
    _render_metric_cards(
        st,
        [
            ("Latest action", str(action.get("title", "")) or "-"),
            ("Status", str(action.get("status", "")) or "-"),
            ("Started", str(action.get("started_at", "")) or "-"),
            ("Exit", str(action.get("exit_code", "")) if action.get("exit_code") is not None else "-"),
        ],
    )
    st.caption(_remote_console_hint(action))
    st.code(str(action.get("display_command", "")), language="bash")
    stdout_parse_text = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=40)
    stderr_parse_text = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=40)
    parsed_issue = _remote_console_issue_summary(action, stdout_parse_text, stderr_parse_text)
    if parsed_issue is not None:
        level = str(parsed_issue.get("level", "info"))
        message = f"{parsed_issue.get('title', 'Parsed remote state')}: {parsed_issue.get('summary', '')}".strip()
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        elif level == "success":
            st.success(message)
        else:
            st.info(message)
        next_step = str(parsed_issue.get("next_step", "")).strip()
        if next_step:
            st.caption(f"Next move: {next_step}")
    reference_rows = _remote_console_reference_rows(action)
    if reference_rows:
        st.write("Remote file references")
        st.dataframe(reference_rows)
    stdout_excerpt = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=18)
    stderr_excerpt = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=18)
    if stdout_excerpt:
        st.write("Latest stdout excerpt")
        st.code(stdout_excerpt, language="text")
    if stderr_excerpt:
        st.write("Latest stderr excerpt")
        st.code(stderr_excerpt, language="text")
    if not stdout_excerpt and not stderr_excerpt:
        st.info("The latest remote action has no captured output yet.")
    failure_history = _remote_console_history_rows(actions, kinds)
    if failure_history:
        st.write("Recent parsed failures / warnings")
        st.dataframe(failure_history)
    scheduler_history = _remote_console_scheduler_history_rows(actions, kinds)
    if scheduler_history:
        st.write("Scheduler / hold history")
        st.dataframe(scheduler_history)
    if str(action.get("output_path", "")).strip() and st.button("Open latest action output", key=f"open_console_output_{key_prefix}_{action['id']}"):
        try:
            open_local_path(str(action.get("output_path", "")))
        except Exception as exc:
            st.error(str(exc))
    st.divider()


def _render_action_history(st, *, actions: list[dict[str, object]], run_root: str, key_prefix: str) -> None:
    if not actions:
        st.info("No dashboard actions recorded for this context yet.")
        return

    for action in actions[:12]:
        with st.container():
            header_cols = st.columns([5, 2, 2])
            header_cols[0].markdown(f"**{action['title']}**")
            header_cols[1].markdown(f"`{action.get('scope', 'local')}`")
            header_cols[2].markdown(f"`{action.get('status', '')}`")
            st.code(str(action.get("display_command", "")), language="bash")

            detail_bits = [
                f"Started: `{action.get('started_at', '') or '-'}`",
                f"Finished: `{action.get('finished_at', '') or '-'}`",
                f"Exit: `{action.get('exit_code', '') if action.get('exit_code') is not None else '-'}`",
            ]
            if action.get("remote_job_id"):
                detail_bits.append(f"Remote job id: `{action['remote_job_id']}`")
            if action.get("sync_status"):
                detail_bits.append(f"Sync: `{action['sync_status']}`")
            st.caption("  |  ".join(detail_bits))

            if action.get("operator_note"):
                st.write(f"Note: {action['operator_note']}")

            has_live_process = action.get("worker_pid") is not None or action.get("command_pid") is not None
            if action.get("status") == "failed":
                excerpt = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=12)
                if excerpt:
                    st.error(excerpt)
            elif action.get("status") in ACTIVE_ACTION_STATUSES:
                st.info(f"Action is currently {action['status']}.")
            elif action.get("status") in APPROVAL_PENDING_STATUSES:
                st.warning("Awaiting approval before execution.")
            elif action.get("status") == "paused":
                st.warning("Action is paused.")
            elif action.get("status") in {"cancelled", "manual_override"} and has_live_process:
                st.warning(f"Action is terminating with status {action['status']}. Wait for cleanup to finish before rerunning.")
            else:
                st.success(f"Action finished with status {action['status']}.")

            action_cols = st.columns(6)
            if action_cols[0].button("View logs", key=f"view_logs_{key_prefix}_{action['id']}"):
                stdout_excerpt = read_log_excerpt(str(action.get("stdout_log", "")), max_lines=20)
                stderr_excerpt = read_log_excerpt(str(action.get("stderr_log", "")), max_lines=20)
                if stdout_excerpt:
                    st.write("stdout excerpt")
                    st.code(stdout_excerpt, language="text")
                if stderr_excerpt:
                    st.write("stderr excerpt")
                    st.code(stderr_excerpt, language="text")
                if not stdout_excerpt and not stderr_excerpt:
                    st.info("No log output captured yet.")

            if action_cols[1].button("Open stdout", key=f"open_stdout_{key_prefix}_{action['id']}"):
                try:
                    open_local_path(str(action.get("stdout_log", "")))
                except Exception as exc:
                    st.error(str(exc))
            if action_cols[2].button("Open stderr", key=f"open_stderr_{key_prefix}_{action['id']}"):
                try:
                    open_local_path(str(action.get("stderr_log", "")))
                except Exception as exc:
                    st.error(str(exc))

            output_path = str(action.get("output_path", ""))
            if output_path and action_cols[3].button("Open output", key=f"open_output_{key_prefix}_{action['id']}"):
                try:
                    open_local_path(output_path)
                except Exception as exc:
                    st.error(str(exc))

            if action.get("status") in APPROVAL_PENDING_STATUSES:
                if action_cols[4].button("Approve", key=f"approve_{key_prefix}_{action['id']}"):
                    try:
                        approved = approve_dashboard_action(Path(run_root), str(action["id"]))
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _report_action_result(st, approved)
                if action_cols[5].button("Pause", key=f"pause_{key_prefix}_{action['id']}"):
                    try:
                        pause_dashboard_action(Path(run_root), str(action["id"]))
                        _stash_dashboard_flash(st, "success", "Action paused.")
                        _trigger_dashboard_rerun(st)
                    except Exception as exc:
                        st.error(str(exc))
            elif action.get("status") in PAUSABLE_ACTION_STATUSES:
                if action_cols[4].button("Pause", key=f"pause_{key_prefix}_{action['id']}"):
                    try:
                        pause_dashboard_action(Path(run_root), str(action["id"]))
                        _stash_dashboard_flash(st, "success", "Action paused.")
                        _trigger_dashboard_rerun(st)
                    except Exception as exc:
                        st.error(str(exc))
            elif action.get("status") == "paused":
                if action_cols[4].button("Resume", key=f"resume_{key_prefix}_{action['id']}"):
                    try:
                        resumed = resume_dashboard_action(Path(run_root), str(action["id"]))
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _report_action_result(st, resumed)

            footer_cols = st.columns(4)
            if action.get("status") not in {"manual_override", "cancelled"}:
                if footer_cols[0].button("Handled manually", key=f"manual_{key_prefix}_{action['id']}"):
                    try:
                        mark_manual_override(Path(run_root), str(action["id"]))
                        _stash_dashboard_flash(st, "success", "Marked as manual override.")
                        _trigger_dashboard_rerun(st)
                    except Exception as exc:
                        st.error(str(exc))

            if action.get("status") in {"succeeded", "failed", "cancelled", "manual_override"} and not has_live_process:
                if footer_cols[1].button("Rerun", key=f"rerun_{key_prefix}_{action['id']}"):
                    try:
                        rerun_action = rerun_dashboard_action(Path(run_root), str(action["id"]))
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _report_action_result(st, rerun_action)
            elif action.get("status") in ACTIVE_ACTION_STATUSES and footer_cols[1].button("Cancel", key=f"cancel_{key_prefix}_{action['id']}"):
                try:
                    cancel_dashboard_action(Path(run_root), str(action["id"]))
                    _stash_dashboard_flash(st, "success", "Cancellation requested.")
                    _trigger_dashboard_rerun(st)
                except Exception as exc:
                    st.error(str(exc))
            st.divider()


def _default_campaign_name(ladder: dict[str, object]) -> str:
    run_name = Path(str(ladder.get("run_dir", "run"))).name or "run"
    sequence = str(ladder.get("sequence", "SEQ"))
    next_profile = str(ladder.get("next_profile", "stage"))
    return f"{run_name}_{next_profile}_{sequence[:12]}".lower()



def _frame_empty(frame) -> bool:
    if frame is None:
        return True
    if hasattr(frame, "empty"):
        return bool(frame.empty)
    return not bool(frame)


def _frame_records(frame) -> list[dict[str, object]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        try:
            return list(frame.to_dict("records"))
        except Exception:
            pass
    if isinstance(frame, list):
        return [dict(row) for row in frame]
    return []


def _with_constant(frame, **kwargs):
    return [dict(row, **kwargs) for row in _frame_records(frame)]


def _wide_chart(rows: list[dict[str, object]], *, index_key: str, series_key: str, value_key: str) -> dict[str, dict[object, object]]:
    chart: dict[str, dict[object, object]] = {}
    for row in rows:
        series = str(row.get(series_key, ""))
        index = row.get(index_key, "")
        value = row.get(value_key)
        if not series or value in {None, ""}:
            continue
        chart.setdefault(series, {})[index] = value
    return chart


def _multi_metric_chart(rows: list[dict[str, object]], *, index_key: str, value_keys: list[str], label_key: str) -> dict[str, dict[object, object]]:
    chart: dict[str, dict[object, object]] = {}
    for row in rows:
        label = str(row.get(label_key, ""))
        index = row.get(index_key, "")
        for value_key in value_keys:
            value = row.get(value_key)
            if value in {None, ""}:
                continue
            key = f"{label}:{value_key}" if label else value_key
            chart.setdefault(key, {})[index] = value
    return chart


def _concat_frames(frames):
    usable = [frame for frame in frames if not _frame_empty(frame)]
    if not usable:
        return []
    if pd is not None and all(hasattr(frame, "to_dict") for frame in usable):
        return pd.concat(usable, ignore_index=True)
    rows: list[dict[str, object]] = []
    for frame in usable:
        rows.extend(_frame_records(frame))
    return rows


def _render_overview(st, state: dict[str, object]) -> None:
    overview = state["overview"]
    peptide_inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    counts = peptide_inventory.get("counts", {}) if isinstance(peptide_inventory.get("counts", {}), dict) else {}
    readiness_alerts = _readiness_alert_rows(state)
    notifications = _notification_rows_for_display(state, include_acknowledged=False, limit=8)
    st.header("Today")
    st.caption("Start here when you want the dashboard to answer: what should I do next, what is blocked, and which peptides are ready to move.")
    if str(state.get("workflow_mode", DEFAULT_DASHBOARD_WORKFLOW_MODE)) == "Guided thesis mode":
        _render_metric_cards(
            st,
            [
                ("Visible runs", overview["run_count"]),
                ("Suggested peptides", counts.get("Suggested by model", 0)),
                ("Ready for ingest", counts.get("Ready for ingest", 0)),
                ("Needs review", counts.get("Needs review / label", 0)),
                ("Unread notifications", len(notifications)),
                ("Readiness alerts", len(readiness_alerts)),
                ("Awaiting approval", len(overview["approval_queue"])),
            ],
        )
        act_now, keep_moving = _today_queue_sections(overview)
        st.subheader("Act now")
        if act_now:
            st.caption("These are the highest-value items right now. Other actions remain available on their task pages.")
            st.dataframe(act_now[:8])
        elif readiness_alerts:
            st.caption("No queue item is urgent, but readiness alerts need attention before some actions are safe.")
            st.dataframe(readiness_alerts[:8])
        else:
            st.info("No urgent review, ingest, monitoring, or readiness items are currently visible.")

        if overview["approval_queue"]:
            st.subheader("Awaiting approval")
            st.dataframe(overview["approval_queue"][:6])
        if overview["blocked_items"]:
            st.subheader("Blocking issues")
            st.dataframe(overview["blocked_items"][:8])

        with st.expander("Full thesis checklist details", expanded=False):
            st.subheader("Guided thesis checklist")
            thesis_phase_rows = build_thesis_phase_rows(state)
            thesis_phase_summary = build_thesis_phase_summary(thesis_phase_rows)
            _render_metric_cards(
                st,
                [
                    ("Checklist verdict", thesis_phase_summary.get("verdict", "")),
                    ("Overall progress", f"{thesis_phase_summary.get('progress', 0)}%"),
                    ("Phases done", f"{thesis_phase_summary.get('done', 0)}/{thesis_phase_summary.get('total', 0)}"),
                    ("Blocked phases", thesis_phase_summary.get("blocked", 0)),
                    ("Next phase", thesis_phase_summary.get("next_phase", "")),
                ],
            )
            st.dataframe(thesis_phase_display_rows(thesis_phase_rows))
        with st.expander("Beginner workflow map", expanded=False):
            st.caption("Reference map for the normal thesis route. It is informational, not a strict wizard.")
            st.dataframe(BEGINNER_WORKFLOW_GUIDE)
        with st.expander("GUI coverage", expanded=False):
            st.caption("Dashboard coverage reference for thesis workflow steps.")
            st.dataframe(GUI_COVERAGE_GUIDE)
        with st.expander("Advanced Today details", expanded=False):
            if keep_moving:
                st.subheader("Keep moving")
                st.dataframe(keep_moving)
            if overview["latest_proposed_peptides"]:
                st.subheader("Suggested by the model right now")
                st.dataframe(overview["latest_proposed_peptides"])
            feedback_queue = list(overview.get("feedback_queue", [])) if isinstance(overview.get("feedback_queue", []), list) else []
            if feedback_queue:
                st.subheader("Feedback queue")
                st.dataframe(feedback_queue)
            if readiness_alerts:
                st.subheader("Readiness alerts")
                st.dataframe(readiness_alerts)
            if notifications:
                st.subheader("Notifications")
                st.dataframe(notifications)
            if overview["active_remote_actions"]:
                st.subheader("Remote work in progress")
                st.dataframe(overview["active_remote_actions"])
            if overview["recent_failures"]:
                st.subheader("Recent failures")
                st.dataframe(overview["recent_failures"])
        return
    _render_recommended_card(st, **_primary_dashboard_recommendation(state))
    if str(state.get("workspace_scope", "")) == "Current Thesis Work" and not state.get("curation", {}).get("pinned_runs", []):
        st.info("No runs are pinned yet, so Current Thesis Work is using a safe default view that hides likely smoke / bugcheck / tuning runs. You can pin the real thesis runs in Operations.")
    _render_metric_cards(
        st,
        [
            ("Visible runs", overview["run_count"]),
            ("Suggested peptides", counts.get("Suggested by model", 0)),
            ("Ready for ingest", counts.get("Ready for ingest", 0)),
            ("Needs review", counts.get("Needs review / label", 0)),
            ("Unread notifications", len(notifications)),
            ("Readiness alerts", len(readiness_alerts)),
            ("Awaiting approval", len(overview["approval_queue"])),
        ],
    )
    thesis_phase_rows = build_thesis_phase_rows(state)
    thesis_phase_summary = build_thesis_phase_summary(thesis_phase_rows)
    st.subheader("Guided thesis checklist")
    st.caption("This live roadmap keeps the beginner path visible: Setup -> Run -> Study -> MD -> Ingest -> Freeze -> Export.")
    _render_metric_cards(
        st,
        [
            ("Checklist verdict", thesis_phase_summary.get("verdict", "")),
            ("Overall progress", f"{thesis_phase_summary.get('progress', 0)}%"),
            ("Phases done", f"{thesis_phase_summary.get('done', 0)}/{thesis_phase_summary.get('total', 0)}"),
            ("Blocked phases", thesis_phase_summary.get("blocked", 0)),
            ("Next phase", thesis_phase_summary.get("next_phase", "")),
        ],
    )
    _render_recommended_card(
        st,
        eyebrow="Thesis phase navigator",
        title=f"Next phase: {thesis_phase_summary.get('next_phase', '')}",
        summary=str(thesis_phase_summary.get("safe_next_move", "")),
        why="This is inferred from the current dashboard state and thesis artifacts already on disk.",
        do_now=str(thesis_phase_summary.get("next_click", "")),
        next_after="Open Operations -> Thesis checklist for the full phase-by-phase audit and copy-friendly roadmap.",
    )
    st.dataframe(thesis_phase_display_rows(thesis_phase_rows))
    st.subheader("Beginner workflow map")
    st.caption("If you are not sure where to click next, use this as the thesis route map. The CLI can still do these jobs, but the normal path should be dashboard-first.")
    st.dataframe(BEGINNER_WORKFLOW_GUIDE)
    st.markdown("#### GUI coverage")
    st.caption("This shows which parts of the thesis workflow are already safe to operate from the dashboard and which pieces remain better as advanced scripted runs.")
    st.dataframe(GUI_COVERAGE_GUIDE)
    act_now, keep_moving = _today_queue_sections(overview)

    left_col, right_col = st.columns([1.35, 1])
    with left_col:
        st.subheader("Act now")
        if act_now:
            st.caption("These are the highest-value operational steps right now: review, ingest, live monitoring, or the next MD preparation step.")
            st.dataframe(act_now)
        else:
            st.info("No urgent review / ingest / monitoring items are currently visible.")

        st.subheader("Keep moving")
        if keep_moving:
            st.caption("These are the next broader model or reporting steps once the urgent queue is clear.")
            st.dataframe(keep_moving)
        else:
            st.info("No broader follow-up steps were inferred yet for the current view.")

        if overview["latest_proposed_peptides"]:
            st.subheader("Suggested by the model right now")
            st.dataframe(overview["latest_proposed_peptides"])
        else:
            st.info("No current proposed peptide batch is visible for this workspace view.")

        feedback_queue = list(overview.get("feedback_queue", [])) if isinstance(overview.get("feedback_queue", []), list) else []
        if feedback_queue:
            st.subheader("Feedback queue")
            st.caption("This is the run-level handoff from reviewed MD evidence back into the active-learning loop.")
            st.dataframe(feedback_queue)

        md_slate_exceptions = list(overview.get("md_slate_exceptions", [])) if isinstance(overview.get("md_slate_exceptions", []), list) else []
        if md_slate_exceptions:
            st.subheader("Slate recovery")
            st.caption("These peptides are the ones that need recovery attention inside the active MD slate flow.")
            st.dataframe(md_slate_exceptions)

        if overview["waiting_on_review"]:
            st.subheader("Review / ingest handoff")
            st.caption("These peptides are closest to closing the MD feedback loop. Either assign the human label or turn that reviewed label into model feedback.")
            st.dataframe(overview["waiting_on_review"])

    with right_col:
        active_md_slates = list(overview.get("active_md_slates", [])) if isinstance(overview.get("active_md_slates", []), list) else []
        bura_utilization = overview.get("bura_utilization", {}) if isinstance(overview.get("bura_utilization", {}), dict) else {}
        snapshot_summary = bura_utilization.get("snapshot_summary", {}) if isinstance(bura_utilization, dict) else {}
        tracked_counts = bura_utilization.get("tracked_external_counts", {}) if isinstance(bura_utilization, dict) else {}
        if active_md_slates or tracked_counts:
            st.subheader("MD slate status")
            _render_metric_cards(
                st,
                [
                    ("Active slates", len(active_md_slates)),
                    ("line_smoke in use", tracked_counts.get("line_smoke", 0)),
                    ("production_smoke in use", tracked_counts.get("production_smoke", 0)),
                    ("full in use", tracked_counts.get("full", 0)),
                    ("BURA running", snapshot_summary.get("running", 0)),
                    ("BURA pending", snapshot_summary.get("pending", 0)),
                ],
            )
            if active_md_slates:
                st.dataframe(
                    [
                        {
                            "run": _path_name(str(row.get("run_dir", ""))),
                            "status": row.get("effective_status", ""),
                            "peptides": row.get("peptide_count", 0),
                            "active": row.get("active_count", 0),
                            "blocked": row.get("blocked_count", 0),
                            "exceptions": row.get("exception_count", 0),
                            "review_ready": row.get("review_ready_count", 0),
                        }
                        for row in active_md_slates[:6]
                    ]
                )
        if notifications:
            st.subheader("Notifications")
            st.caption("These are remembered event-style reminders. Acknowledge them in Operations once you have handled them or consciously parked them.")
            st.dataframe(notifications)
        execution_readiness = state.get("execution_readiness", {}) if isinstance(state.get("execution_readiness", {}), dict) else {}
        readiness_top_rows = list(execution_readiness.get("top_rows", [])) if isinstance(execution_readiness.get("top_rows", []), list) else []
        readiness_counts = execution_readiness.get("counts", {}) if isinstance(execution_readiness.get("counts", {}), dict) else {}
        if readiness_top_rows:
            st.subheader("Execution readiness")
            st.caption("These verdicts answer whether the main real actions are safe right now, and what needs fixing when they are not.")
            _render_metric_cards(
                st,
                [
                    ("Blocked", readiness_counts.get("blocked", 0)),
                    ("Caution", readiness_counts.get("caution", 0)),
                    ("Ready", readiness_counts.get("ready", 0)),
                ],
            )
            st.dataframe(
                [
                    {
                        "readiness": row.get("verdict", ""),
                        "action": row.get("label", ""),
                        "target": row.get("target", ""),
                        "summary": row.get("summary", ""),
                        "fix_now": row.get("fix_now", ""),
                    }
                    for row in readiness_top_rows
                ]
            )
        if readiness_alerts:
            st.subheader("Readiness alerts")
            st.caption("These are the most actionable readiness blockers or handoff signals right now.")
            st.dataframe(readiness_alerts)
        if overview.get("blocked_items"):
            st.subheader("Blocked / waiting items")
            st.dataframe(overview["blocked_items"])
        if overview["approval_queue"]:
            st.subheader("Approval queue")
            st.dataframe(overview["approval_queue"])
        if overview["active_remote_actions"]:
            st.subheader("Remote work in progress")
            st.dataframe(overview["active_remote_actions"])
        if overview["recent_failures"]:
            st.subheader("Recent failures")
            st.dataframe(overview["recent_failures"])

    cluster_rows = [row for row in _cluster_health_rows(state) if row.get("status") != "ok"]
    if cluster_rows:
        st.subheader("Cluster blockers")
        st.dataframe(cluster_rows)


def _render_results_view(st, state: dict[str, object]) -> None:
    _render_results_view_impl(st, state, ns=sys.modules[__name__])


def _render_peptides_view(st, state: dict[str, object]) -> None:
    _render_peptides_view_impl(st, state, ns=sys.modules[__name__])


def _render_run_detail(st, state: dict[str, object]) -> None:
    _render_model_workflow_view_impl(st, state, ns=sys.modules[__name__])


def _render_stage_progress(st, ladder: dict[str, object]) -> None:
    stage_specs = [
        ("line_smoke", _friendly_md_profile("line_smoke")),
        ("production_smoke", _friendly_md_profile("production_smoke")),
        ("full", _friendly_md_profile("full")),
    ]
    columns = st.columns(len(stage_specs))
    for column, (profile_key, label) in zip(columns, stage_specs):
        item = ladder.get(profile_key)
        status = str(item.get("job_root_status", "not_started")) if item else "not_started"
        sync_status = str(item.get("sync_status", "not_synced")) if item else "not_synced"
        campaign_name = str(item.get("campaign", "")) if item else ""
        profile_info = MD_PROFILE_INFO.get(profile_key, {})
        with column:
            column.markdown(f"#### {label}")
            column.caption(str(profile_info.get("description", "")))
            column.metric("Stage status", _friendly_md_status(status))
            column.caption(f"Campaign: {campaign_name or '-'}")
            column.caption(f"Remote sync: {sync_status}")
            produces = str(profile_info.get("produces", ""))
            if produces:
                column.caption(f"Produces: {produces}")


def _render_make_ingest_action(
    st,
    ladder: dict[str, object],
    *,
    run_root: Path,
    key_prefix: str,
    after_submit=None,
    state: dict[str, object] | None = None,
) -> None:
    if not ladder["make_ingest_command"] or not ladder.get("full"):
        return
    if not bool(ladder.get("ingest_supported", True)):
        blocker = str(ladder.get("ingest_blocker", "")) or "This peptide is currently reporting-only and should not be turned into `cgmd_ingest.csv` yet."
        if bool(ladder.get("promotion_available", False)):
            blocker += " Use the promotion bridge first so the campaign points to the real proposed batch."
        st.warning(blocker)
        return
    full_item = ladder["full"]
    ingest_csv_path = _ingest_csv_path(str(full_item["campaign_dir"]))
    if ingest_csv_path.exists():
        st.info("`cgmd_ingest.csv` already exists for this campaign. The next step is to switch to Model Workflow and run Ingest returned labels for the parent run.")
        return
    review_status = review_evidence_status(full_item)
    if not bool(review_status.get("ingest_ready", False)):
        st.warning(
            "Complete the evidence-backed review before creating `cgmd_ingest.csv`: "
            + _review_evidence_missing_text(review_status)
        )
        return
    readiness = build_button_readiness(state or {"run_root": str(run_root)}, "make-md-ingest-csv", ladder=ladder)
    _render_launch_action(
        st,
        label=f"Create ingest CSV for {ladder['sequence']}",
        command=ladder["make_ingest_command"],
        key_prefix=key_prefix,
        button_text="Run locally",
        what="Convert a reviewed full-analysis MD result into the `cgmd_ingest.csv` format expected by `ingest-round`.",
        when="Use this after the full-analysis run has a human-approved `cgmd_label`.",
        produces="A `cgmd_ingest.csv` file that the model workflow can ingest.",
        next_step="The run can then execute 'Ingest returned labels' in Model Workflow.",
        contract_id="make-md-ingest-csv",
        readiness=readiness,
        after_submit=after_submit,
        on_submit=lambda full_item=full_item: submit_make_md_ingest_action(
            run_root=run_root,
            campaign_dir=Path(str(full_item["campaign_dir"])),
            review_csv=Path(str(full_item["review_path"])),
            sequence=str(ladder["sequence"]),
            related_run=str(ladder["run_dir"]),
        ),
    )


def _run_local_block_reason(run: dict[str, object]) -> str:
    if _recommended_run_workflow_command(run):
        return ""
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
    if str(feedback_queue.get("status", "")) == "blocked" and str(feedback_queue.get("pending_round_id", "")).strip():
        return str(feedback_queue.get("summary", "")) or "the pending proposed batch still needs review, promotion, or full-analysis completion before ingest can run"
    if str(run.get("ml_status", "")) == "discovery-complete" and not run.get("final_metrics"):
        return "discovery already finished, so the next move depends on whether you want to freeze a final evaluation or validate more peptides first"
    if list(run.get("available_ingest_csvs", [])) and not list(run.get("import_rows", [])):
        return "reviewed MD feedback is available, but ingest has not been run yet"
    return "the next local step still depends on a human choice between review, discovery, or reporting"


def _run_remote_block_reason(state: dict[str, object], run: dict[str, object]) -> tuple[str, str]:
    health = _cluster_health_entry(state, "supek")
    if str(health.get("overall_status", "unknown")) != "ok":
        summary = str(health.get("summary", "")).strip()
        hint = str(health.get("hint", "")).strip()
        return (" ".join(part for part in [summary, hint] if part), "warning")
    if get_cluster_profile(state.get("profiles", {}), "supek") is None:
        return (_cluster_profile_warning(state, "supek"), "warning")
    remote_status = str(run.get("remote_sync_status", "not_synced"))
    if remote_status == "submitted" and not str(run.get("remote_job_id", "")).strip():
        return ("the run looks remotely submitted, but there is no tracked SUPEK job id attached yet", "warning")
    if remote_status in {"outputs_staged", "outputs_returned"}:
        return ("artifacts are already back on the local side, so the next safe move is local inspection, finalization, or pull-back review rather than another submit", "info")
    if remote_status == "staged_remote" and not _recommended_run_workflow_command(run):
        return ("the run is staged on SUPEK, but the next remote workflow still depends on a human choice between review, discovery, or final reporting", "warning")
    return ("", "info")


def _ladder_block_reason(state: dict[str, object], ladder: dict[str, object]) -> tuple[str, str]:
    macro_key = _recommended_ladder_macro_key(ladder)
    health = _cluster_health_entry(state, "bura")
    if str(health.get("overall_status", "unknown")) != "ok" and macro_key in {"upload-bura", "normalize-bura", "poll-bura"}:
        summary = str(health.get("summary", "")).strip()
        hint = str(health.get("hint", "")).strip()
        return (" ".join(part for part in [summary, hint] if part), "warning")
    if macro_key:
        return ("", "info")
    if bool(ladder.get("ready_for_review")):
        return ("the peptide has reached the human-review handoff, so the next step is a label or ingest decision rather than another ladder macro", "info")
    if str(ladder.get("sync_status", "")) in {"submitted", "running"} and not str((ladder.get("current") or {}).get("remote_job_id", "")).strip():
        return ("the ladder is marked as remotely active, but there is no tracked BURA job id attached yet", "warning")
    return ("no single safe ladder macro is available because this peptide currently needs review, inspection, or a manual branch choice", "warning")


def _render_run_workflow_macros(st, state: dict[str, object], run: dict[str, object]) -> None:
    st.markdown("#### Guided workflow runner")
    st.caption("Use this runner when you want the cockpit to show the whole run-level path, highlight the current checkpoint, and give you one safe action to advance that path.")
    run_root = Path(str(state["run_root"]))
    run_dir_text = str(run["run_dir"])
    recommended_command = _recommended_run_workflow_command(run)
    local_rows = _run_plan_rows(run)
    local_current = next((row for row in local_rows if str(row.get("status", "")) == "current"), {})
    _render_plan_checkpoint_table(
        st,
        title="Local thesis loop checkpoints",
        rows=local_rows,
        summary="This is the local research path from baseline benchmarking to a report-ready frozen result.",
    )
    _render_runner_memory_panel(
        st,
        current_checkpoint=str(local_current.get("checkpoint", "")),
        events=_progress_events_for_context(
            state,
            scope="run",
            plan_kind="run-local",
            run_dir=run_dir_text,
        ),
        blocker=_run_local_block_reason(run),
        blocker_level="warning",
    )
    if recommended_command:
        info = RUN_ACTION_INFO[recommended_command]
        if recommended_command == "continue-feedback":
            feedback_command = (
                "python -m active_learning_thesis dashboard-continue-feedback "
                f"--run-dir {_quote_path(run['run_dir'])}"
            )
            continue_feedback_readiness = build_button_readiness(state, "continue-al-feedback", run=run)
            _render_launch_action(
                st,
                label=f"Advance local plan: {info['label']}",
                command=feedback_command,
                key_prefix=f"macro_{run['run_slug']}_continue_feedback",
                button_text="Advance plan",
                what=info["what"],
                when="Use this when the full pending proposed batch is already reviewed and you want the cockpit to complete the ingest/retrain handoff for you.",
                produces=info["produces"],
                next_step=info["next"],
                contract_id="continue-al-feedback",
                readiness=continue_feedback_readiness,
                after_submit=lambda action, run_dir=run_dir_text, checkpoint=info["label"], label=info["label"]: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-local",
                    checkpoint=checkpoint,
                    action_label=label,
                    action=action,
                    run_dir=run_dir,
                    note="Advanced from the guided local thesis loop.",
                ),
                on_submit=lambda run_dir=Path(str(run["run_dir"])): submit_continue_feedback_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    propose_next_batch=False,
                ),
            )
        elif recommended_command == "ingest-round" and list(run.get("available_ingest_csvs", [])):
            selected_ingest_csv = str(list(run.get("available_ingest_csvs", []))[0])
            ingest_command = (
                "python -m active_learning_thesis ingest-round "
                f"--run-dir {_quote_path(run['run_dir'])} "
                f"--import-csv {_quote_path(selected_ingest_csv)}"
            )
            ingest_readiness = build_button_readiness(state, "ingest-round", run=run)
            _render_launch_action(
                st,
                label=f"Advance local plan: {info['label']}",
                command=ingest_command,
                key_prefix=f"macro_{run['run_slug']}_ingest",
                button_text="Advance plan",
                what="Close the loop by ingesting the first reviewed `cgmd_ingest.csv` currently available for this run.",
                when="Use this when the run already has reviewed MD feedback staged and you want the model to absorb it now.",
                produces=info["produces"],
                next_step=info["next"],
                contract_id="ingest-round",
                readiness=ingest_readiness,
                after_submit=lambda action, run_dir=run_dir_text, label=info["label"]: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-local",
                    checkpoint="Ingest returned labels",
                    action_label=label,
                    action=action,
                    run_dir=run_dir,
                    note="Advanced from the guided local thesis loop.",
                ),
                on_submit=lambda run_dir=Path(str(run["run_dir"])), import_csv=Path(selected_ingest_csv): submit_ingest_round_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    import_csv=import_csv,
                ),
            )
        else:
            command = f"python -m active_learning_thesis {recommended_command} --run-dir {_quote_path(run['run_dir'])}"
            recommended_readiness = build_button_readiness(state, recommended_command, run=run)
            _render_launch_action(
                st,
                label=f"Advance local plan: {info['label']}",
                command=command,
                key_prefix=f"macro_{run['run_slug']}_{recommended_command}",
                button_text="Advance plan",
                what=f"Dispatch the current best next local model step for this run: {info['what']}",
                when="Use this when you want the dashboard to advance the run according to its current thesis state.",
                produces=info["produces"],
                next_step=info["next"],
                contract_id=recommended_command,
                readiness=recommended_readiness,
                after_submit=lambda action, run_dir=run_dir_text, checkpoint=info["label"], label=info["label"]: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-local",
                    checkpoint=checkpoint,
                    action_label=label,
                    action=action,
                    run_dir=run_dir,
                    note="Advanced from the guided local thesis loop.",
                ),
                on_submit=lambda command_name=recommended_command, run_dir=Path(str(run["run_dir"])): submit_run_workflow_action(
                    run_root=run_root,
                    command_name=command_name,
                    run_dir=run_dir,
                ),
            )
    else:
        st.info("No single local model macro is recommended for this run right now. The next step likely depends on peptide validation, review, or interpretation.")

    discovery_ready = str(run.get("ml_status", "")) in {"replay-complete", "batch-proposed"} or bool(run.get("import_rows"))
    if discovery_ready and str(run.get("ml_status", "")) != "discovery-complete":
        info = RUN_ACTION_INFO["run-discovery"]
        command = f"python -m active_learning_thesis run-discovery --run-dir {_quote_path(run['run_dir'])}"
        discovery_readiness = build_button_readiness(state, "run-discovery", run=run)
        _render_launch_action(
            st,
            label="Optional branch: Explore discovery now",
            command=command,
            key_prefix=f"macro_{run['run_slug']}_discovery",
            button_text="Run branch",
            what="Launch discovery mode as a guided side path when the run is mature enough to explore novel candidates.",
            when="Use this when you want exploratory thesis evidence in parallel with the main active-learning loop.",
            produces=info["produces"],
            next_step=info["next"],
            contract_id="run-discovery",
            readiness=discovery_readiness,
            after_submit=lambda action, run_dir=run_dir_text: _record_runner_progress(
                run_root,
                scope="run",
                plan_kind="run-local",
                checkpoint="Run discovery",
                action_label="Optional branch: Explore discovery now",
                action=action,
                run_dir=run_dir,
                note="User chose the discovery side path from the guided runner.",
            ),
            on_submit=lambda run_dir=Path(str(run["run_dir"])): submit_run_workflow_action(
                run_root=run_root,
                command_name="run-discovery",
                run_dir=run_dir,
            ),
        )

    supek_profile = get_cluster_profile(state.get("profiles", {}), "supek")
    remote_sync_status = str(run.get("remote_sync_status", "not_synced"))
    if supek_profile is not None:
        remote_rows = _remote_run_plan_rows(run)
        remote_current = next((row for row in remote_rows if str(row.get("status", "")) == "current"), {})
        _render_plan_checkpoint_table(
            st,
            title="Remote SUPEK checkpoints",
            rows=remote_rows,
            summary="This is the remote execution path for the same run: stage, submit, monitor, then pull artifacts back.",
        )
        remote_blocker, remote_blocker_level = _run_remote_block_reason(state, run)
        _render_runner_memory_panel(
            st,
            current_checkpoint=str(remote_current.get("checkpoint", "")),
            events=_progress_events_for_context(
                state,
                scope="run",
                plan_kind="run-remote",
                run_dir=run_dir_text,
            ),
            blocker=remote_blocker,
            blocker_level=remote_blocker_level,
        )
        recommended_remote_command = _recommended_run_workflow_command(run)
        if remote_sync_status == "not_synced":
            supek_stage_readiness = build_button_readiness(state, "supek-sync-run", run=run)
            _render_draft_action(
                st,
                label="Advance remote plan: Stage this run on SUPEK",
                command=f"scp -r {_quote_path(run['run_dir'])} {supek_profile['username']}@{supek_profile['host']}:<scratch>",
                key_prefix=f"macro_supek_stage_{run['run_slug']}",
                what="Create the exact remote draft needed to get this run staged on SUPEK before any remote workflow submission.",
                when="Use this when you want the next run action to happen remotely and the run has not been uploaded yet.",
                produces="A draft upload action ready for approval.",
                next_step="Once approved and completed, the remote workflow macro can submit the recommended SUPEK job.",
                contract_id="supek-sync-run",
                readiness=supek_stage_readiness,
                after_submit=lambda action, run_dir=run_dir_text: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-remote",
                    checkpoint="Stage run on SUPEK",
                    action_label="Advance remote plan: Stage this run on SUPEK",
                    action=action,
                    run_dir=run_dir,
                    note="Created the SUPEK staging draft from the guided runner.",
                ),
                on_submit=lambda run_dir=Path(str(run["run_dir"])): draft_supek_sync_run_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    profile=supek_profile,
                ),
            )
        elif remote_sync_status == "staged_remote" and recommended_remote_command:
            info = RUN_ACTION_INFO[recommended_remote_command]
            supek_submit_readiness = build_button_readiness(state, "supek-submit-workflow", run=run)
            _render_draft_action(
                st,
                label=f"Advance remote plan: Submit {info['label']} on SUPEK",
                command=f"Submit {recommended_remote_command} via qsub on SUPEK",
                key_prefix=f"macro_supek_submit_{run['run_slug']}_{recommended_remote_command}",
                what="Create the exact next remote workflow draft for the currently recommended run step.",
                when="Use this when the run is already staged on SUPEK and you want the next thesis step to happen remotely.",
                produces="An approval-gated remote SUPEK workflow draft with tracked job metadata.",
                next_step="After approval, polling and log fetch become the active monitoring controls.",
                contract_id="supek-submit-workflow",
                readiness=supek_submit_readiness,
                after_submit=lambda action, run_dir=run_dir_text, label=info["label"]: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-remote",
                    checkpoint="Submit next SUPEK workflow",
                    action_label=f"Advance remote plan: Submit {label} on SUPEK",
                    action=action,
                    run_dir=run_dir,
                    note="Created the next SUPEK workflow draft from the guided runner.",
                ),
                on_submit=lambda command_name=recommended_remote_command, run_dir=Path(str(run["run_dir"])): draft_supek_submit_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    profile=supek_profile,
                    command_name=command_name,
                ),
            )
        elif remote_sync_status in {"submitted", "running"} and str(run.get("remote_job_id", "")):
            supek_poll_readiness = build_button_readiness(state, "supek-poll-qstat", run=run)
            _render_launch_action(
                st,
                label="Advance remote plan: Monitor current SUPEK job",
                command=f"ssh {supek_profile['username']}@{supek_profile['host']} qstat -u {supek_profile['username']}",
                key_prefix=f"macro_supek_monitor_{run['run_slug']}",
                button_text="Advance plan",
                what="Poll the active SUPEK job for this run using the tracked job id already attached to the run state.",
                when="Use this while the remote workflow is still queued or running.",
                produces="An updated remote queue snapshot and refreshed sync state.",
                next_step="If the job is finished, fetch logs or pull artifacts back next.",
                contract_id="supek-poll-qstat",
                readiness=supek_poll_readiness,
                after_submit=lambda action, run_dir=run_dir_text: _record_runner_progress(
                    run_root,
                    scope="run",
                    plan_kind="run-remote",
                    checkpoint="Monitor remote queue",
                    action_label="Advance remote plan: Monitor current SUPEK job",
                    action=action,
                    run_dir=run_dir,
                    note="Polled the tracked SUPEK job from the guided runner.",
                ),
                on_submit=lambda run_dir=Path(str(run["run_dir"])): queue_supek_poll_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    profile=supek_profile,
                    remote_job_id=str(run.get("remote_job_id", "")),
                ),
            )


def _render_review_workspace(st, ladder: dict[str, object], *, run_root: Path, state: dict[str, object]) -> None:
    st.subheader("Review & model feedback")
    full_item = ladder.get("full")
    next_step = ladder.get("next_step", {}) if isinstance(ladder.get("next_step"), dict) else {}
    if not full_item:
        st.info("This peptide has not reached the full-analysis stage yet, so there is nothing to review or feed back into the model.")
        if ladder.get("next_profile_label"):
            st.caption(f"Current ladder focus: {ladder['next_profile_label']}.")
        if next_step:
            st.write(f"Next ladder step: {next_step.get('title', 'Review ladder state')}")
            st.caption(str(next_step.get("summary", "")))
        return

    label_value = str(full_item.get("cgmd_label", "")).strip()
    review_notes = str(full_item.get("review_notes", "")).strip()
    review_status = review_evidence_status(full_item)
    evidence_ready = bool(review_status.get("ingest_ready", False))
    full_status = _friendly_md_status(str(full_item.get("job_root_status", "")))
    review_ready = bool(ladder.get("ready_for_review"))
    source_batch_kind = str(full_item.get("source_batch_kind", ladder.get("source_batch_kind", "")))
    source_batch_label = _source_batch_kind_label(source_batch_kind)
    ingest_supported = _source_batch_ingest_supported(source_batch_kind)
    ingest_blocker = _source_batch_ingest_blocker(source_batch_kind)
    ingest_csv_path = _ingest_csv_path(str(full_item.get("campaign_dir", "")))
    ingest_exists = ingest_csv_path.exists()
    promoted_at = str(full_item.get("promoted_to_real_batch_at", "")).strip()
    promoted_round_id = str(full_item.get("promoted_round_id", "")).strip()
    promoted_from_batch = str(full_item.get("promoted_from_source_batch_csv", "")).strip()
    if review_ready and label_value in {"0", "1"} and not evidence_ready:
        review_state = "Needs review evidence"
        next_feedback_step = "Complete the label rubric, confidence, evidence summary, and notes before creating model feedback."
    elif review_ready and label_value in {"0", "1"} and ingest_supported:
        review_state = "Ready for ingest"
        next_feedback_step = (
            "Run Ingest returned labels in Model Workflow"
            if ingest_exists
            else "Create cgmd_ingest.csv"
        )
    elif review_ready and label_value in {"0", "1"}:
        review_state = "Reviewed for reporting"
        next_feedback_step = "Use the returned MD evidence in thesis reporting, or wait until the peptide is part of a real proposed batch before ingesting."
    elif review_ready:
        review_state = "Needs review / label"
        next_feedback_step = (
            "Assign cgmd_label in md_review.csv"
            if ingest_supported
            else "Assign cgmd_label for reporting support"
        )
    else:
        review_state = "Not review-ready yet"
        next_feedback_step = str(next_step.get("title", "Complete the full analysis rerun"))

    _render_metric_cards(
        st,
        [
            ("Review state", review_state),
            ("Full-analysis status", full_status),
            ("Source batch", source_batch_label),
            ("Current label", label_value or "Not assigned"),
            ("Review evidence", str(review_status.get("state", ""))),
            ("Confidence", str(review_status.get("confidence", "")) or "-"),
            ("Simulation runtime", str(full_item.get("md_runtime_wall_hms", "")) or "-"),
            ("MD speed", (str(full_item.get("md_runtime_ns_per_day", "")).strip() + " ns/day") if str(full_item.get("md_runtime_ns_per_day", "")).strip() else "-"),
            ("Largest cluster", str(full_item.get("cluster_largest_fraction_200ns", "")) or "-"),
            ("Ingest CSV", "Reporting-only batch" if not ingest_supported else ("Created locally" if ingest_exists else "Not created")),
        ],
    )

    if review_ready and label_value in {"0", "1"} and not evidence_ready:
        st.warning(
            "A binary label is saved, but the review is not evidence-backed yet. Missing or blocked: "
            + _review_evidence_missing_text(review_status)
        )
    elif review_ready and label_value in {"0", "1"} and ingest_supported:
        if ingest_exists:
            st.success("This peptide already has a human MD label and a `cgmd_ingest.csv`. The next step is to open Model Workflow and run Ingest returned labels for the parent run.")
        else:
            st.success("This peptide already has a human MD label. The next step is to create `cgmd_ingest.csv`, then ingest it from Model Workflow.")
    elif review_ready and label_value in {"0", "1"}:
        st.info(ingest_blocker or "This peptide is reviewed for reporting, but it is not yet valid for `ingest-round`.")
    elif review_ready:
        if ingest_supported:
            st.warning("The full-analysis outputs are ready, but the peptide still needs a human `cgmd_label` in `md_review.csv` before it can be fed back into the model.")
        else:
            st.warning("The full-analysis outputs are ready, but this dashboard-local MD batch still needs a human `cgmd_label` for reporting support.")
    else:
        st.info("This peptide is still in the MD ladder. Review and ingest stay blocked until the full-analysis stage reaches `analysis_complete`.")

    if promoted_at:
        st.markdown("#### Promotion provenance")
        st.success(
            "This campaign was already promoted from a reporting-only MD batch into the real AL loop."
            + (f" The promoted round is {promoted_round_id}." if promoted_round_id else "")
        )
        st.dataframe(
            [
                {
                    "sequence": ladder.get("sequence", ""),
                    "promoted_at": promoted_at,
                    "original_source_batch": promoted_from_batch or "-",
                    "current_source_batch": str(full_item.get("source_batch_csv", "")) or "-",
                    "promoted_round": promoted_round_id or "-",
                }
            ]
        )

    if review_ready and not ingest_supported:
        st.markdown("#### Promotion bridge")
        if bool(ladder.get("promotion_available", False)):
            round_hint = str(ladder.get("promotion_target_round_id", "")).strip()
            st.success(
                "A real proposed batch now contains this peptide, so you can promote the current reporting-only MD campaign into the real AL loop."
                + (f" The detected target round is {round_hint}." if round_hint else "")
            )
            st.dataframe(
                [
                    {
                        "sequence": ladder.get("sequence", ""),
                        "real_batch_csv": str(ladder.get("promotion_target_batch_csv", "")) or "-",
                        "target_round": round_hint or "-",
                        "current_source_batch": str(full_item.get("source_batch_csv", "")) or "-",
                    }
                ]
            )
            _render_launch_action(
                st,
                label=f"Promote {ladder['sequence']} into the real AL batch",
                command=f"rebind reporting-only MD campaign to {_path_name(str(ladder.get('promotion_target_batch_csv', '')))}",
                key_prefix=f"promote_reporting_{ladder['sequence']}",
                button_text="Promote locally",
                what="Rewrite the current campaign metadata from the dashboard-local MD batch to the real proposed batch row for this peptide.",
                when="Use this only after a real proposed batch contains the same peptide and you want the existing MD result to become genuinely ingestable.",
                produces="Updated campaign manifest/review round metadata, a real source-batch reference, and removal of any stale ingest CSV built against the old dashboard batch.",
                next_step="After promotion, create cgmd_ingest.csv and run Ingest returned labels from Model Workflow.",
                contract_id="promote-reporting-md-campaign",
                readiness=build_button_readiness(state, "promote-reporting-md-campaign", ladder=ladder),
                on_submit=lambda full_item=full_item: submit_promote_reporting_md_campaign_action(
                    run_root=run_root,
                    campaign_dir=Path(str(full_item["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    related_run=str(ladder["run_dir"]),
                ),
            )
        else:
            st.info(
                "No real proposed batch currently contains this peptide yet. The current MD result remains valid for reporting, and the promotion step will unlock automatically once the peptide appears in a real batch."
            )

    st.markdown("#### Evidence packet")
    st.caption("Use this compact packet as the thesis handoff: the binary label should point back to the AP/SASA outputs, your rubric call, and a confidence level.")
    st.dataframe(
        [
            {
                "evidence_item": "Full-analysis status",
                "value": full_status,
                "how_to_use": "Only final labels from analysis_complete full runs should enter model feedback.",
            },
            {
                "evidence_item": "AP trajectory ratios",
                "value": ", ".join(
                    f"{target}ns={_format_decimal_text(full_item.get(f'ap_{target}ns', '')) or '-'}"
                    for target in (5, 12, 25, 50, 100, 200)
                ),
                "how_to_use": "Legacy diagnostic: SASA at production 0 ns divided by SASA at each time point.",
            },
            {
                "evidence_item": "Paper-style AP_SASA final-10-ns average",
                "value": (
                    f"mean={_format_decimal_text(full_item.get('paper_ap_sasa_last10ns_mean', '')) or '-'}, "
                    f"sd={_format_decimal_text(full_item.get('paper_ap_sasa_last10ns_sd', '')) or '-'}, "
                    f"n={str(full_item.get('paper_ap_sasa_last10ns_n_frames', '')).strip() or '-'}, "
                    f"initial={str(full_item.get('paper_ap_sasa_initial_source', '')).strip() or '-'}, "
                    f"status={str(full_item.get('paper_ap_sasa_status', '')).strip() or '-'}"
                ),
                "how_to_use": "Njirjak/Thapa-style AP_SASA evidence: initial non-contact SASA divided by final-10-ns mean SASA; approximate if only production 0 ns is available.",
            },
            {
                "evidence_item": "AP contact fractions",
                "value": ", ".join(
                    f"{target}ns={_format_decimal_text(full_item.get(f'ap_contact_{target}ns', '')) or '-'}"
                    for target in (5, 12, 25, 50, 100, 200)
                ),
                "how_to_use": "Use AP_contact as diagnostic contact-fraction evidence. The retained Phase 3 label uses paper_path_APcontact_last10ns >= 0.5.",
            },
            {
                "evidence_item": "AP contact, exact paper formula",
                "value": ", ".join(
                    f"{target}ns={_format_decimal_text(full_item.get(f'ap_contact_same_paper_formula_{target}ns', '')) or '-'}"
                    for target in (5, 12, 25, 50, 100, 200)
                ),
                "how_to_use": "Uses the paper's piecewise distance weight: 1 below 4 A, exponential decay from 4-12 A, and 0 above 12 A.",
            },
            {
                "evidence_item": "AP contact, paper path score",
                "value": ", ".join(
                    f"{target}ns={_format_decimal_text(full_item.get(f'paper_path_ap_contact_{target}ns', '')) or '-'}"
                    for target in (5, 12, 25, 50, 100, 200)
                ),
                "how_to_use": "Uses the paper distance formula plus a path visiting every peptide copy once; for large systems the path maximization uses deterministic beam search.",
            },
            {
                "evidence_item": "AP contact, paper path last-10-ns average",
                "value": (
                    f"mean={_format_decimal_text(full_item.get('paper_path_ap_contact_last10ns_mean', '')) or '-'}, "
                    f"sd={_format_decimal_text(full_item.get('paper_path_ap_contact_last10ns_sd', '')) or '-'}, "
                    f"n={str(full_item.get('paper_path_ap_contact_last10ns_n_frames', '')).strip() or '-'}, "
                    f"status={str(full_item.get('paper_path_ap_contact_last10ns_status', '')).strip() or '-'}"
                ),
                "how_to_use": "Closest AP_contact evidence to the reference definition when final-10-ns frames have been extracted; otherwise the status points to the BURA extraction script.",
            },
            {
                "evidence_item": "200 ns cluster structure",
                "value": (
                    f"largest={str(full_item.get('cluster_largest_fraction_200ns', '')).strip() or '-'}, "
                    f"clusters={str(full_item.get('cluster_count_200ns', '')).strip() or '-'}, "
                    f"singletons={str(full_item.get('cluster_singleton_fraction_200ns', '')).strip() or '-'}, "
                    f"mean contacts={str(full_item.get('cluster_mean_contacts_200ns', '')).strip() or '-'}"
                ),
                "how_to_use": "Distinguishes one coherent aggregate from many weak contacts; largest cluster close to 1 with few singletons supports self-assembly.",
            },
            {
                "evidence_item": "Simulation runtime",
                "value": (
                    f"{str(full_item.get('md_runtime_wall_hms', '')).strip() or '-'}"
                    + (
                        f" ({str(full_item.get('md_runtime_ns_per_day', '')).strip()} ns/day)"
                        if str(full_item.get("md_runtime_ns_per_day", "")).strip()
                        else ""
                    )
                ),
                "how_to_use": "Documents how much production MD evidence supports this review row.",
            },
            {
                "evidence_item": "AP summary file",
                "value": str(full_item.get("ap_file", "")) or "-",
                "how_to_use": "Legacy SASA(0 ns production)/SASA(t) numeric evidence.",
            },
            {
                "evidence_item": "Paper-style AP_SASA file",
                "value": str(full_item.get("paper_ap_sasa_last10ns_file", "")) or "-",
                "how_to_use": "Auditable source file for final-10-ns AP_SASA.",
            },
            {
                "evidence_item": "Paper-style AP_SASA recompute script",
                "value": str(full_item.get("paper_ap_sasa_recompute_script", "")) or "-",
                "how_to_use": "Run this on BURA/GROMACS if the SASA group or true initial non-contact SASA must be recomputed.",
            },
            {
                "evidence_item": "AP contact file",
                "value": str(full_item.get("ap_contact_file", "")) or "-",
                "how_to_use": "Contact-based companion metric for borderline AP_SASA cases.",
            },
            {
                "evidence_item": "Exact paper AP contact file",
                "value": str(full_item.get("ap_contact_same_paper_formula_file", "")) or "-",
                "how_to_use": "Auditable source file for the exact piecewise paper AP_contact calculation.",
            },
            {
                "evidence_item": "Paper path AP contact file",
                "value": str(full_item.get("paper_path_ap_contact_file", "")) or "-",
                "how_to_use": "Auditable source file for the path-based paper-formula AP_contact calculation.",
            },
            {
                "evidence_item": "Paper path AP contact last-10-ns file",
                "value": str(full_item.get("paper_path_ap_contact_last10ns_file", "")) or "-",
                "how_to_use": "Auditable source file for the final-10-ns average when available.",
            },
            {
                "evidence_item": "Paper path AP contact last-10-ns extraction script",
                "value": str(full_item.get("paper_path_ap_contact_last10ns_script", "")) or "-",
                "how_to_use": "Run this on BURA/GROMACS if the final-10-ns frames are not copied back yet.",
            },
            {
                "evidence_item": "Aggregate summary file",
                "value": str(full_item.get("aggregate_summary_file", "")) or "-",
                "how_to_use": "Cutoff-sensitivity and cluster-size diagnostics for saturated AP_contact cases.",
            },
            {
                "evidence_item": "SASA file",
                "value": str(full_item.get("sasa_file", "")) or "-",
                "how_to_use": "Supporting surface-area evidence and sanity check.",
            },
            {
                "evidence_item": "Current review evidence",
                "value": str(review_status.get("state", "")),
                "how_to_use": "Ingest stays blocked for new structured reviews until this is evidence-backed.",
            },
        ]
    )

    st.markdown("#### Review editor")
    existing_rubric = str(full_item.get("label_rubric", "")).strip()
    if not existing_rubric and label_value in {"0", "1"}:
        existing_rubric = default_rubric_for_label(label_value)
    if existing_rubric not in LABEL_RUBRIC_OPTIONS:
        existing_rubric = ""
    existing_confidence = str(full_item.get("label_confidence", "")).strip().lower()
    if existing_confidence not in LABEL_CONFIDENCE_OPTIONS:
        existing_confidence = ""
    selected_label = st.selectbox(
        "Human review label (`cgmd_label`)",
        options=["", "0", "1"],
        index=(["", "0", "1"].index(label_value) if label_value in {"", "0", "1"} else 0),
        key=f"review_label_{ladder['sequence']}",
    )
    selected_rubric = st.selectbox(
        "Thesis review rubric",
        options=list(LABEL_RUBRIC_OPTIONS),
        index=list(LABEL_RUBRIC_OPTIONS).index(existing_rubric),
        key=f"review_rubric_{ladder['sequence']}",
    )
    rubric_label = label_for_rubric(selected_rubric)
    resolved_label = rubric_label if selected_rubric else str(selected_label)
    if selected_rubric == "uncertain_rerun":
        resolved_label = ""
    if selected_rubric and str(selected_label).strip() and rubric_label and str(selected_label).strip() != rubric_label:
        st.warning("The rubric and numeric label disagree; saving will follow the rubric-derived label.")
    selected_confidence = st.selectbox(
        "Review confidence",
        options=list(LABEL_CONFIDENCE_OPTIONS),
        index=list(LABEL_CONFIDENCE_OPTIONS).index(existing_confidence),
        key=f"review_confidence_{ladder['sequence']}",
    )
    edited_tags = st.text_input(
        "Evidence tags",
        value=normalize_evidence_tags(full_item.get("label_evidence_tags", "")),
        key=f"review_evidence_tags_{ladder['sequence']}",
    )
    st.caption("Suggested tags: " + ", ".join(LABEL_EVIDENCE_TAG_OPTIONS))
    edited_summary = st.text_area(
        "Evidence summary",
        value=str(full_item.get("label_evidence_summary", "")).strip(),
        key=f"review_evidence_summary_{ladder['sequence']}",
    )
    edited_notes = st.text_area(
        "Review notes",
        value=review_notes,
        key=f"review_notes_{ladder['sequence']}",
    )
    edited_reviewer = st.text_input(
        "Reviewer",
        value=str(full_item.get("reviewer", "")).strip(),
        key=f"reviewer_{ladder['sequence']}",
    )
    draft_review = {
        **full_item,
        "cgmd_label": resolved_label,
        "label_rubric": str(selected_rubric),
        "label_confidence": str(selected_confidence),
        "label_evidence_tags": normalize_evidence_tags(edited_tags),
        "label_evidence_summary": str(edited_summary).strip(),
        "review_notes": str(edited_notes).strip(),
        "reviewer": str(edited_reviewer).strip(),
        "label_review_schema": "structured",
    }
    draft_status = review_evidence_status(draft_review)
    st.dataframe([_review_evidence_summary_row(draft_review)])
    save_review_command = (
        f"update md_review.csv for {ladder['sequence']}: "
        f"cgmd_label={resolved_label or '<empty>'} "
        f"rubric={str(selected_rubric) or '<empty>'} "
        f"confidence={str(selected_confidence) or '<empty>'} "
        f"evidence={str(edited_summary).strip() or '<empty>'!r}"
    )
    _render_launch_action(
        st,
        label=f"Save review decision for {ladder['sequence']}",
        command=save_review_command,
        key_prefix=f"save_review_{ladder['sequence']}",
        button_text="Save review locally",
        what="Write the current `cgmd_label`, rubric, confidence, evidence summary, and notes back into the campaign's `md_review.csv` without leaving the cockpit.",
        when="Use this after you have inspected the full-analysis outputs and want the dashboard state to reflect a thesis-grade review decision.",
        produces="An updated evidence-backed `md_review.csv` plus a recorded local dashboard action showing the saved review decision.",
        contract_id="update-md-review",
        readiness=build_button_readiness(state, "update-md-review", ladder=ladder),
        next_step=(
            "If the label is now final, create `cgmd_ingest.csv`, then ingest it from Model Workflow."
            if ingest_supported
            else "If the label is now final, use the result for thesis reporting, or wait until the peptide appears in a real proposed batch before ingesting."
        ),
        on_submit=lambda full_item=full_item, resolved_label=resolved_label, selected_rubric=selected_rubric, selected_confidence=selected_confidence, edited_tags=edited_tags, edited_summary=edited_summary, edited_notes=edited_notes, edited_reviewer=edited_reviewer, run_root=run_root: submit_update_md_review_action(
            run_root=run_root,
            review_csv=Path(str(full_item["review_path"])),
            sequence=str(ladder["sequence"]),
            cgmd_label=str(resolved_label),
            review_notes=str(edited_notes),
            label_rubric=str(selected_rubric),
            label_confidence=str(selected_confidence),
            label_evidence_tags=str(edited_tags),
            label_evidence_summary=str(edited_summary),
            reviewer=str(edited_reviewer),
            related_run=str(ladder["run_dir"]),
            related_campaign=str(full_item["campaign_dir"]),
        ),
    )
    if not bool(draft_status.get("ingest_ready", False)):
        st.caption("Draft evidence still needs: " + _review_evidence_missing_text(draft_status))

    st.markdown("#### Review checklist")
    checklist_rows = [
        {
            "checkpoint": "Full-analysis status",
            "state": full_status,
            "details": "Review can start only after `analysis_complete`.",
        },
        {
            "checkpoint": "Review CSV",
            "state": _path_name(str(full_item.get("review_path", ""))) or "-",
            "details": "Assign `cgmd_label` and keep any human notes here.",
        },
        {
            "checkpoint": "Source batch semantics",
            "state": source_batch_label,
            "details": (
                "This result can flow back into `ingest-round`."
                if ingest_supported
                else "This was launched from a dashboard-local MD batch, so it currently supports reporting only."
            ),
        },
        {
            "checkpoint": "Current label",
            "state": label_value or "Not assigned",
            "details": "Use your thesis review rubric, then save `md_review.csv`.",
        },
        {
            "checkpoint": "Evidence-backed review",
            "state": str(review_status.get("state", "")),
            "details": (
                "Ready for model feedback."
                if evidence_ready
                else "Complete: " + _review_evidence_missing_text(review_status)
            ),
        },
        {
            "checkpoint": "Ingest CSV",
            "state": _path_name(ingest_csv_path) if ingest_exists else "Not created",
            "details": (
                "Create this locally once the review label is final."
                if ingest_supported
                else "This stays blocked until the campaign is promoted into a real proposed batch."
            ),
        },
        {
            "checkpoint": "Next model step",
            "state": next_feedback_step,
            "details": "This is the exact handoff back into the active-learning run.",
        },
    ]
    st.dataframe(checklist_rows)

    st.markdown("#### Review files")
    st.dataframe(
        [
            {
                "artifact": "Campaign directory",
                "path": str(full_item.get("campaign_dir", "")) or "-",
                "why_it_matters": "Main folder holding the packaged MD outputs for this peptide.",
            },
            {
                "artifact": "Review CSV",
                "path": str(full_item.get("review_path", "")) or "-",
                "why_it_matters": "This is where `cgmd_label`, rubric, confidence, evidence summary, and review notes live.",
            },
            {
                "artifact": "AP summary",
                "path": str(full_item.get("ap_file", "")) or "-",
                "why_it_matters": "Helpful final-analysis output to inspect before assigning the label.",
            },
            {
                "artifact": "AP contact summary",
                "path": str(full_item.get("ap_contact_file", "")) or "-",
                "why_it_matters": "Diagnostic contact-fraction evidence; it is separate from the path-based contact metric used by the retained Phase 3 label.",
            },
            {
                "artifact": "Exact paper AP contact summary",
                "path": str(full_item.get("ap_contact_same_paper_formula_file", "")) or "-",
                "why_it_matters": "Shows the contact score calculated with the paper's piecewise distance-weight function.",
            },
            {
                "artifact": "Paper path AP contact summary",
                "path": str(full_item.get("paper_path_ap_contact_file", "")) or "-",
                "why_it_matters": "Shows the path-based contact score using the paper's distance-weight function.",
            },
            {
                "artifact": "Aggregate summary",
                "path": str(full_item.get("aggregate_summary_file", "")) or "-",
                "why_it_matters": "Cutoff sensitivity, cluster count, largest cluster fraction, and contact density for stronger self-assembly interpretation.",
            },
            {
                "artifact": "SASA file",
                "path": str(full_item.get("sasa_file", "")) or "-",
                "why_it_matters": "Useful supporting evidence during review.",
            },
        ]
    )

    if review_notes:
        st.markdown("#### Current review notes")
        st.write(review_notes)

    st.markdown("#### Handoff guide")
    handoff_rows = [
        {
            "phase": "1. Review the full-analysis outputs",
            "what_to_do": "Inspect the campaign outputs and decide the human outcome for this peptide.",
            "done_when": "You know whether the peptide should receive a `cgmd_label` of `0` or `1`.",
        },
        {
            "phase": "2. Update `md_review.csv`",
            "what_to_do": "Write the final `cgmd_label`, rubric, confidence, evidence summary, and useful notes into the campaign review CSV.",
            "done_when": "The review evidence state is `Evidence-backed label` or a legacy notes-only row you have intentionally kept.",
        },
    ]
    if ingest_supported:
        handoff_rows.extend(
            [
                {
                    "phase": "3. Create `cgmd_ingest.csv`",
                    "what_to_do": "Run the local ingest-CSV action once the label is final.",
                    "done_when": "A `cgmd_ingest.csv` file exists in the campaign directory.",
                },
                {
                    "phase": "4. Feed the label back into the model",
                    "what_to_do": "Open Model Workflow for the parent run and run Ingest returned labels.",
                    "done_when": "The run shows the returned label in its import history.",
                },
            ]
        )
    else:
        handoff_rows.extend(
            [
                {
                    "phase": "3. Promote into a real proposed batch",
                    "what_to_do": (
                        "Use the promotion bridge once a real proposed batch contains this peptide."
                        if bool(ladder.get("promotion_available", False))
                        else "Wait until a real proposed batch contains this peptide, then come back here to promote it."
                    ),
                    "done_when": "The campaign source batch points to a real proposed batch row instead of the dashboard-local MD batch.",
                },
                {
                    "phase": "4. Create `cgmd_ingest.csv`",
                    "what_to_do": "Run the local ingest-CSV action after promotion updates the round metadata.",
                    "done_when": "A `cgmd_ingest.csv` file exists in the campaign directory with the real batch round id.",
                },
                {
                    "phase": "5. Feed the label back into the model",
                    "what_to_do": "Open Model Workflow for the parent run and run Ingest returned labels.",
                    "done_when": "The run shows the returned label in its import history.",
                },
            ]
        )
    st.dataframe(handoff_rows)

    if ladder.get("make_ingest_command") and ingest_supported and label_value in {"0", "1"} and not ingest_exists:
        st.markdown("#### Copyable ingest command")
        st.code(str(ladder["make_ingest_command"]), language="bash")
        st.caption("Use this only after the review label is final. Once it succeeds, switch to Model Workflow and run Ingest returned labels.")


def _find_ladder_for_review_row(state: dict[str, object], *, run_name: str, sequence: str) -> dict[str, object] | None:
    for ladder in list(state.get("peptides", [])):
        if (
            str(ladder.get("sequence", "")) == sequence
            and str(ladder.get("run_display_name", ladder.get("run_name", ""))) == run_name
        ):
            return ladder
    return None


def _md_slate_resource_request(md_profile: str) -> str:
    if md_profile == "line_smoke":
        return "1 node | 1 task/node | 2 CPUs/task"
    if md_profile == "production_smoke":
        return "1 node | 4 tasks/node | 2 CPUs/task"
    if md_profile == "full":
        return "5 nodes | 20 tasks/node | 2 CPUs/task"
    return "-"


def _render_md_source_batch_export_action(
    st,
    state: dict[str, object],
    *,
    rows: list[dict[str, object]],
    key_prefix: str,
    label: str,
    next_step: str,
) -> None:
    if not rows:
        return
    normalized_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip()
        if not sequence or not run_dir:
            continue
        key = (_canonical_path(run_dir), sequence)
        if key in seen:
            continue
        seen.add(key)
        normalized_rows.append(row)
    if not normalized_rows:
        return
    preview = ", ".join(str(row.get("sequence", "")) for row in normalized_rows[:6])
    if len(normalized_rows) > 6:
        preview += f", +{len(normalized_rows) - 6} more"
    readiness = build_button_readiness(
        state,
        "export-md-source-batch",
        rows=normalized_rows,
    )
    _render_launch_action(
        st,
        label=label,
        command=f"dashboard-local MD source batch export for {preview}",
        key_prefix=key_prefix,
        button_text="Create local MD source batch",
        what="Create one-row, dashboard-local MD source batch CSVs so these peptides can enter MD preparation even if they were discovery-only candidates.",
        when="Use this when the peptide is thesis-relevant for MD/reporting, but there is no real proposed batch CSV row for the MD prepare step yet.",
        produces="A dashboard-local batch CSV per peptide that unlocks MD preparation and slate launch.",
        next_step=next_step,
        contract_id="export-md-source-batch",
        readiness=readiness,
        on_submit=lambda normalized_rows=normalized_rows: submit_export_md_source_batch_action(
            run_root=Path(str(state["run_root"])),
            items=[
                {
                    "sequence": str(row.get("sequence", "")),
                    "run_dir": str(row.get("run_dir", "")),
                    "run_name": str(row.get("run", "")),
                    "round_id": "" if str(row.get("proposal_round", "")) == "-" else str(row.get("proposal_round", "")),
                    "strategy": "" if str(row.get("strategy", "")) == "-" else str(row.get("strategy", "")),
                    "source": str(row.get("source", "")),
                }
                for row in normalized_rows
            ],
        ),
    )
    st.info(
        "These dashboard-local MD source batches unlock MD preparation and reporting support only. "
        "They do not by themselves make the peptide valid for `ingest-round`."
    )


def _active_md_stage_counts(md_slates: list[dict[str, object]]) -> dict[str, int]:
    counts = {key: 0 for key in SLATE_STAGE_CAPS}
    for slate in md_slates:
        if not isinstance(slate, dict):
            continue
        if str(slate.get("effective_status", "")) in {"completed", "completed_with_failures", "cancelled"}:
            continue
        for peptide in list(slate.get("peptides", [])):
            if not isinstance(peptide, dict):
                continue
            stage = str(peptide.get("current_stage", "")).strip()
            if stage in counts and str(peptide.get("status", "")) == "active":
                counts[stage] += 1
    return counts


def _md_slate_planner_candidate_payload(row: dict[str, object]) -> dict[str, str]:
    return {
        "sequence": str(row.get("sequence", "")),
        "run_dir": str(row.get("run_dir", "")),
        "run_name": str(row.get("run", "")),
        "source": str(row.get("source", "")),
        "strategy": str(row.get("strategy", "")),
        "priority_band": str(row.get("priority_band", "")),
        "proposal_round": str(row.get("proposal_round", "")),
        "source_batch_csv": "" if str(row.get("source_batch_csv", "")) == "-" else str(row.get("source_batch_csv", "")),
        "source_batch_kind": str(row.get("source_batch_kind", "")),
        "launch_ready": str(row.get("launch_ready", "")),
        "launch_blocker": "" if str(row.get("launch_blocker", "")) == "-" else str(row.get("launch_blocker", "")),
        "next_action": str(row.get("next_action", "")),
        "decision_title": str(row.get("decision_title", "")),
    }


def _save_md_slate_planner_from_rows(
    run_root: Path,
    *,
    run_dir: str,
    run_name: str,
    name: str,
    rationale: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    planner_id = uuid.uuid4().hex[:12]
    planner = {
        "planner_id": planner_id,
        "run_dir": str(run_dir),
        "run_name": str(run_name),
        "name": str(name).strip(),
        "rationale": str(rationale).strip(),
        "status": "draft",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "last_launched_slate_id": "",
        "last_launched_action_id": "",
        "last_launched_at": "",
        "candidates": [_md_slate_planner_candidate_payload(row) for row in rows],
    }
    return save_dashboard_md_slate_planner(run_root, planner)


def _mark_md_slate_planner_launched(run_root: Path, planner_id: str, action: dict[str, object]) -> None:
    if not planner_id:
        return
    planner = load_dashboard_md_slate_planner(run_root, planner_id)
    metadata = action.get("metadata", {}) if isinstance(action.get("metadata", {}), dict) else {}
    planner["status"] = "launched"
    planner["updated_at"] = datetime.now().isoformat(timespec="seconds")
    planner["last_launched_at"] = datetime.now().isoformat(timespec="seconds")
    planner["last_launched_action_id"] = str(action.get("id", "")).strip()
    planner["last_launched_slate_id"] = str(metadata.get("slate_id", "")).strip()
    save_dashboard_md_slate_planner(run_root, planner)


def _live_md_slate_planner_candidates(state: dict[str, object], planner: dict[str, object]) -> list[dict[str, object]]:
    run_root = Path(str(state.get("run_root", "")))
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    candidate_rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    candidate_lookup = {
        (_canonical_path(str(row.get("run_dir", ""))), str(row.get("sequence", ""))): row
        for row in candidate_rows
    }
    live_rows: list[dict[str, object]] = []
    for stored in list(planner.get("candidates", [])):
        if not isinstance(stored, dict):
            continue
        run_dir = str(stored.get("run_dir", "")).strip() or str(planner.get("run_dir", "")).strip()
        sequence = str(stored.get("sequence", "")).strip()
        live = candidate_lookup.get((_canonical_path(run_dir), sequence))
        if isinstance(live, dict):
            merged = dict(stored)
            merged.update(
                {
                    "source": str(live.get("source", stored.get("source", ""))),
                    "strategy": str(live.get("strategy", stored.get("strategy", ""))),
                    "priority_band": str(live.get("priority_band", stored.get("priority_band", ""))),
                    "proposal_round": str(live.get("proposal_round", stored.get("proposal_round", ""))),
                    "source_batch_csv": str(live.get("source_batch_csv", stored.get("source_batch_csv", "-"))),
                    "source_batch_kind": str(live.get("source_batch_kind", stored.get("source_batch_kind", "-"))),
                    "launch_ready": str(live.get("launch_ready", stored.get("launch_ready", "no"))),
                    "launch_blocker": str(live.get("launch_blocker", stored.get("launch_blocker", "-"))),
                    "next_action": str(live.get("next_action", stored.get("next_action", ""))),
                    "candidate_status": str(live.get("candidate_status", "-")),
                }
            )
            live_rows.append(merged)
            continue
        source_batch_csv = str(stored.get("source_batch_csv", "")).strip()
        if not source_batch_csv:
            source_batch_csv = find_md_source_batch_csv(run_root, Path(run_dir), sequence)
        launch_ready = bool(source_batch_csv)
        live_rows.append(
            {
                **stored,
                "source_batch_csv": source_batch_csv or "-",
                "source_batch_kind": _source_batch_kind_label(_source_batch_kind(run_root, source_batch_csv)) if source_batch_csv else "-",
                "launch_ready": "yes" if launch_ready else "no",
                "launch_blocker": "-" if launch_ready else (str(stored.get("launch_blocker", "")).strip() or "No source batch CSV currently contains this peptide."),
                "candidate_status": str(stored.get("candidate_status", "-")).strip() or "-",
            }
        )
    return live_rows


def _planner_capacity_rows(state: dict[str, object]) -> tuple[list[dict[str, str]], int]:
    utilization = state.get("bura_utilization", {}) if isinstance(state.get("bura_utilization", {}), dict) else {}
    tracked_counts = utilization.get("tracked_external_counts", {}) if isinstance(utilization.get("tracked_external_counts", {}), dict) else {}
    active_counts = _active_md_stage_counts(list(state.get("md_slates", [])))
    rows: list[dict[str, str]] = []
    line_available = SLATE_STAGE_CAPS["line_smoke"]
    for stage in SLATE_STAGE_CAPS:
        occupancy = active_counts.get(stage, 0) + int(tracked_counts.get(stage, 0) or 0)
        available = max(SLATE_STAGE_CAPS[stage] - occupancy, 0)
        if stage == "line_smoke":
            line_available = available
        rows.append(
            {
                "stage": _friendly_md_profile(stage),
                "active_slate_jobs": str(active_counts.get(stage, 0)),
                "other_tracked_jobs": str(int(tracked_counts.get(stage, 0) or 0)),
                "occupancy_now": f"{occupancy} / {SLATE_STAGE_CAPS[stage]}",
                "available_now": str(available),
            }
        )
    return rows, line_available


def _md_slate_launch_payloads(rows: list[dict[str, object]], *, fallback_run_dir: str = "") -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip() or str(fallback_run_dir).strip()
        if not sequence or not run_dir:
            continue
        source_batch_csv = str(row.get("source_batch_csv", "")).strip()
        if source_batch_csv == "-":
            source_batch_csv = ""
        payloads.append(
            {
                "sequence": sequence,
                "run_dir": run_dir,
                "source_batch_csv": source_batch_csv,
                "source": str(row.get("source", "")).strip(),
                "strategy": str(row.get("strategy", "")).strip(),
                "priority_band": str(row.get("priority_band", "")).strip(),
            }
        )
    return payloads


def _render_md_slate_launch_readiness_gate(
    st,
    state: dict[str, object],
    *,
    title: str,
    run_dir: str,
    run_name: str,
    rows: list[dict[str, object]],
    key_prefix: str,
) -> dict[str, object]:
    readiness = build_md_slate_launch_readiness(
        run_root=Path(str(state["run_root"])),
        run_dir=Path(str(run_dir)),
        run_name=run_name,
        peptides=_md_slate_launch_payloads(rows, fallback_run_dir=run_dir),
        profiles_payload=state.get("profiles", {}) if isinstance(state.get("profiles", {}), dict) else None,
        md_slates=list(state.get("md_slates", [])),
    )
    st.markdown(f"#### {title}")
    verdict = str(readiness.get("verdict", "Do not launch yet"))
    summary = str(readiness.get("summary", ""))
    if verdict == "Ready to launch":
        st.success(summary)
    elif verdict == "Partially ready":
        st.warning(summary)
    else:
        st.error(summary)
    _render_metric_cards(
        st,
        [
            ("Dry-run verdict", verdict),
            ("Launch-ready", readiness.get("launchable_count", 0)),
            ("Blocked", readiness.get("blocked_count", 0)),
            ("Can submit first wave", readiness.get("starts_now", 0)),
            ("Queued behind caps", readiness.get("queued_by_caps", 0)),
            ("BURA profile", readiness.get("bura_profile_ready", "no")),
        ],
    )
    st.caption("Per-peptide launch readiness")
    st.dataframe(list(readiness.get("peptide_rows", [])))
    st.write("Expected BURA occupancy against the local caps")
    st.caption("BURA cap simulation for this launch")
    st.dataframe(list(readiness.get("cap_rows", [])))
    with st.expander("Child actions after approval", expanded=False):
        st.dataframe(list(readiness.get("child_actions_preview", [])))
    blocked_sequences = list(readiness.get("blocked_sequences", []))
    if blocked_sequences:
        st.caption(
            "Blocked before approval: "
            + ", ".join(str(item) for item in blocked_sequences[:8])
            + ("" if len(blocked_sequences) <= 8 else f", +{len(blocked_sequences) - 8} more")
        )
    return readiness


def _render_selected_md_slate_panel(
    st,
    state: dict[str, object],
    *,
    selected_rows: list[dict[str, object]],
    selected_candidate: dict[str, object] | None = None,
) -> None:
    st.markdown("#### Selected MD slate")
    st.caption("This is the launch pad for GUI-driven MD automation. You can still quick-launch the current selection, but the stronger workflow is to save a named slate plan with its rationale, inspect the expected BURA fit, and then launch that plan intentionally.")
    if not selected_rows:
        st.info("No candidates are currently marked as Selected for MD. Save candidate decisions first, then launch one peptide or the run-level slate from here.")
        return
    run_options = sorted({str(row.get("run", "")) for row in selected_rows if str(row.get("run", "")).strip()})
    selected_run = st.selectbox(
        "Run for the MD slate",
        run_options,
        index=0,
        key="candidate_slate_run",
    )
    run_rows = [row for row in selected_rows if str(row.get("run", "")) == selected_run]
    launchable_rows = [row for row in run_rows if str(row.get("launch_ready", "")) == "yes"]
    blocked_rows = [row for row in run_rows if str(row.get("launch_ready", "")) != "yes"]
    _render_metric_cards(
        st,
        [
            ("Selected peptides", len(run_rows)),
            ("Launch-ready peptides", len(launchable_rows)),
            ("Blocked selected peptides", len(blocked_rows)),
            ("line_smoke cap", SLATE_STAGE_CAPS["line_smoke"]),
            ("production_smoke cap", SLATE_STAGE_CAPS["production_smoke"]),
            ("full cap", SLATE_STAGE_CAPS["full"]),
        ],
    )
    st.caption("Current selected-slate snapshot")
    st.dataframe(
        [
            {
                "sequence": row.get("sequence", ""),
                "source": row.get("source", ""),
                "strategy": row.get("strategy", ""),
                "priority_band": row.get("priority_band", ""),
                "source_batch_kind": row.get("source_batch_kind", "-"),
                "launch_ready": row.get("launch_ready", "-"),
                "source_batch_csv": row.get("source_batch_csv", "-"),
                "launch_blocker": row.get("launch_blocker", "-"),
                "next_action": row.get("next_action", ""),
            }
            for row in run_rows
        ]
    )
    if blocked_rows:
        st.warning("Some selected peptides are not launch-ready yet because the MD prepare step still has no source batch CSV for them.")
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "source": row.get("source", ""),
                    "source_batch_kind": row.get("source_batch_kind", "-"),
                    "launch_blocker": row.get("launch_blocker", ""),
                }
                for row in blocked_rows
            ]
        )
        _render_md_source_batch_export_action(
            st,
            state,
            rows=blocked_rows,
            key_prefix=f"export_md_source_batch_selected_blocked_{selected_run}",
            label=f"Create dashboard-local MD batch rows for blocked selected peptides ({len(blocked_rows)})",
            next_step="Once those local batch rows exist, save or relaunch the slate plan for this run.",
        )

    current_readiness = _render_md_slate_launch_readiness_gate(
        st,
        state,
        title="Launch readiness dry run",
        run_dir=str(run_rows[0].get("run_dir", "")),
        run_name=selected_run,
        rows=run_rows,
        key_prefix=f"current_{selected_run}",
    )

    st.markdown("#### Saved slate planner")
    st.caption("Use this to turn the current selected rows into a named MD plan with a thesis-facing rationale. The planner is saved outside the repo, can include launch blockers, and can be launched later once the blockers are cleared.")
    default_name = f"{selected_run} MD slate"
    planner_name = st.text_input(
        "Planner name",
        value=default_name,
        key=f"candidate_slate_planner_name_{selected_run}",
    )
    planner_rationale = st.text_area(
        "Why does this slate belong together?",
        value="These candidates form the next MD validation slate and should move together under the current thesis priorities.",
        key=f"candidate_slate_planner_rationale_{selected_run}",
        height=100,
    )
    if st.button("Save current selected rows as a named slate plan", key=f"save_md_slate_planner_{selected_run}"):
        if not planner_name.strip() or not planner_rationale.strip():
            st.warning("Add both a planner name and a rationale before saving the slate plan.")
        else:
            _save_md_slate_planner_from_rows(
                Path(str(state["run_root"])),
                run_dir=str(run_rows[0].get("run_dir", "")),
                run_name=selected_run,
                name=planner_name,
                rationale=planner_rationale,
                rows=run_rows,
            )
            st.success("Saved the selected candidates as a named MD slate plan.")
            st.rerun()

    planners = [
        planner
        for planner in list(state.get("md_slate_planners", []))
        if _canonical_path(str(planner.get("run_dir", ""))) == _canonical_path(str(run_rows[0].get("run_dir", "")))
    ]
    if planners:
        planner_labels = [
            f"{planner.get('name', 'Unnamed slate')} [{planner.get('status', 'draft')}]"
            for planner in planners
        ]
        selected_planner_label = st.selectbox(
            "Choose saved slate plan",
            planner_labels,
            index=0,
            key=f"candidate_slate_planner_select_{selected_run}",
        )
        planner = planners[planner_labels.index(selected_planner_label)]
        planner_candidates = _live_md_slate_planner_candidates(state, planner)
        planner_readiness = _render_md_slate_launch_readiness_gate(
            st,
            state,
            title="Saved plan launch dry run",
            run_dir=str(planner.get("run_dir", "")),
            run_name=str(planner.get("run_name", "")),
            rows=planner_candidates,
            key_prefix=f"planner_{planner.get('planner_id', '')}",
        )
        planner_launchable = [
            row
            for row in planner_candidates
            if str(row.get("sequence", "")) in {
                str(item.get("sequence", ""))
                for item in list(planner_readiness.get("launchable_peptides", []))
                if isinstance(item, dict)
            }
        ]
        planner_blocked = [
            row
            for row in planner_candidates
            if str(row.get("sequence", "")) in set(str(item) for item in list(planner_readiness.get("blocked_sequences", [])))
        ]
        _render_metric_cards(
            st,
            [
                ("Planner status", planner.get("status", "-")),
                ("Planned peptides", len(planner_candidates)),
                ("Launch-ready now", len(planner_launchable)),
                ("Blocked in plan", len(planner_blocked)),
                ("Starts immediately", planner_readiness.get("starts_now", 0)),
                ("Queued behind line_smoke cap", planner_readiness.get("queued_by_caps", 0)),
            ],
        )
        st.write(f"Planner rationale: {planner.get('rationale', '-')}")
        if str(planner.get("last_launched_at", "")).strip():
            st.caption(
                f"Last launched: {planner.get('last_launched_at', '-')} | "
                f"Slate id: {planner.get('last_launched_slate_id', '-') or '-'}"
            )
        st.write("Saved slate comparison")
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "candidate_status": row.get("candidate_status", "-"),
                    "source": row.get("source", ""),
                    "strategy": row.get("strategy", ""),
                    "priority_band": row.get("priority_band", ""),
                    "proposal_round": row.get("proposal_round", "-"),
                    "source_batch_kind": row.get("source_batch_kind", "-"),
                    "launch_ready": row.get("launch_ready", "-"),
                    "launch_blocker": row.get("launch_blocker", "-"),
                    "next_action": row.get("next_action", "-"),
                }
                for row in planner_candidates
            ]
        )
        if planner_blocked:
            _render_md_source_batch_export_action(
                st,
                state,
                rows=planner_blocked,
                key_prefix=f"export_md_source_batch_saved_planner_{planner.get('planner_id', '')}",
                label=f"Create dashboard-local MD batch rows for blocked planner peptides ({len(planner_blocked)})",
                next_step="Once those local rows exist, come back here and launch the saved slate plan.",
            )
        if planner_launchable:
            slate_preview = ", ".join(str(row.get("sequence", "")) for row in planner_launchable[:6])
            if len(planner_launchable) > 6:
                slate_preview += f", +{len(planner_launchable) - 6} more"
            launch_label = f"Launch saved slate plan: {planner.get('name', 'MD slate')}"
            if planner_blocked:
                launch_label = f"Launch ready peptides from saved plan: {planner.get('name', 'MD slate')}"
            planner_launch_readiness = build_button_readiness(
                state,
                "md-slate-run",
                rows=planner_launchable,
                run_dir=str(planner.get("run_dir", "")),
                run_name=str(planner.get("run_name", "")),
            )
            _render_draft_action(
                st,
                label=launch_label,
                command=f"dashboard-managed MD slate for {selected_run}: {slate_preview}",
                key_prefix=f"launch_md_saved_planner_{planner.get('planner_id', '')}",
                what="Launch this saved MD slate plan into the existing supervised slate runner, while preserving the plan name and rationale in the launch metadata.",
                when="Use this when the dry-run verdict shows at least one peptide is launch-ready. Blocked peptides stay out of this launch until their blockers are cleared.",
                produces="One approval-gated MD slate action tied back to this saved planner, plus the usual per-peptide child actions.",
                next_step="Approve the slate once, then monitor it from MD Validation -> Slate monitor.",
                contract_id="md-slate-run",
                readiness=planner_launch_readiness,
                on_submit=lambda planner=planner: draft_md_slate_run_action(
                    run_root=Path(str(state["run_root"])),
                    run_dir=Path(str(planner.get("run_dir", ""))),
                    run_name=str(planner.get("run_name", "")),
                    planner_id=str(planner.get("planner_id", "")),
                    planner_name=str(planner.get("name", "")),
                    operator_note=str(planner.get("rationale", "")),
                    peptides=[
                        {
                            "sequence": str(row.get("sequence", "")),
                            "run_dir": str(planner.get("run_dir", "")),
                            "source_batch_csv": "" if str(row.get("source_batch_csv", "")) == "-" else str(row.get("source_batch_csv", "")),
                            "source": str(row.get("source", "")),
                            "strategy": str(row.get("strategy", "")),
                            "priority_band": str(row.get("priority_band", "")),
                        }
                        for row in planner_launchable
                    ],
                ),
                after_submit=lambda action, planner_id=str(planner.get("planner_id", "")): _mark_md_slate_planner_launched(
                    Path(str(state["run_root"])),
                    planner_id,
                    action,
                ),
            )
        elif planner_candidates:
            st.info("This saved slate plan has no launch-ready peptides yet. Clear the dry-run blockers above, then launch it from here.")
    else:
        st.info("No saved MD slate plans exist for this run yet. Save the current selected rows first if you want a reusable, named slate.")

    gate_launchable_sequences = {
        str(item.get("sequence", ""))
        for item in list(current_readiness.get("launchable_peptides", []))
        if isinstance(item, dict)
    }
    gate_launchable_rows = [
        row for row in launchable_rows if str(row.get("sequence", "")) in gate_launchable_sequences
    ]
    st.markdown("#### Quick launch current selection")
    st.caption("This is still available as the faster escape hatch when you do not need a named saved plan first.")
    if gate_launchable_rows:
        slate_preview = ", ".join(str(row.get("sequence", "")) for row in gate_launchable_rows[:6])
        if len(gate_launchable_rows) > 6:
            slate_preview += f", +{len(gate_launchable_rows) - 6} more"
        rehearsal_readiness = build_button_readiness(
            state,
            "md-slate-rehearsal",
            rows=gate_launchable_rows,
            run_dir=str(run_rows[0].get("run_dir", "")),
            run_name=selected_run,
        )
        _render_launch_action(
            st,
            label=f"Run rehearsal for selected MD slate ({len(gate_launchable_rows)} peptides)",
            command=f"dashboard rehearsal only: simulate MD slate for {selected_run}: {slate_preview}",
            key_prefix=f"rehearse_md_slate_{selected_run}",
            button_text="Run rehearsal",
            what="Simulate the full slate ladder locally without touching BURA or SUPEK.",
            when="Use this before a real launch when you want to rehearse caps, auto-advance, review-ready handoff, and recovery visibility.",
            produces="A dashboard-local rehearsal slate with fake queue events, fake artifacts, and review-ready state. It never assigns `cgmd_label`.",
            next_step="Open MD Validation -> Slate monitor and inspect the rehearsal slate before launching the real one.",
            contract_id="md-slate-rehearsal",
            readiness=rehearsal_readiness,
            on_submit=lambda selected_run=selected_run, gate_launchable_rows=gate_launchable_rows: launch_md_slate_rehearsal_action(
                run_root=Path(str(state["run_root"])),
                run_dir=Path(str(gate_launchable_rows[0]["run_dir"])),
                run_name=selected_run,
                peptides=[
                    {
                        "sequence": str(row.get("sequence", "")),
                        "run_dir": str(row.get("run_dir", "")),
                        "source_batch_csv": str(row.get("source_batch_csv", "")) if str(row.get("source_batch_csv", "")) != "-" else "",
                        "source": str(row.get("source", "")),
                        "strategy": str(row.get("strategy", "")),
                        "priority_band": str(row.get("priority_band", "")),
                    }
                    for row in gate_launchable_rows
                ],
            ),
        )
        launch_readiness = build_button_readiness(
            state,
            "md-slate-run",
            rows=gate_launchable_rows,
            run_dir=str(run_rows[0].get("run_dir", "")),
            run_name=selected_run,
        )
        _render_draft_action(
            st,
            label=f"Launch selected MD slate ({len(gate_launchable_rows)} peptides)",
            command=f"dashboard-managed MD slate for {selected_run}: {slate_preview}",
            key_prefix=f"launch_md_slate_{selected_run}",
            what="Snapshot the currently selected peptides for this run into one supervised MD slate.",
            when="Use this when you already know which peptides should go to MD and you want the cockpit to drive the routine ladder work automatically.",
            produces="One approval-gated MD slate action plus per-peptide child actions for local prepare, BURA staging, submit, monitoring, pull-back, and finalize.",
            next_step="Approve the slate once, then monitor it from MD Validation -> Slate monitor.",
            contract_id="md-slate-run",
            readiness=launch_readiness,
            on_submit=lambda selected_run=selected_run, gate_launchable_rows=gate_launchable_rows: draft_md_slate_run_action(
                run_root=Path(str(state["run_root"])),
                run_dir=Path(str(gate_launchable_rows[0]["run_dir"])),
                run_name=selected_run,
                peptides=[
                    {
                        "sequence": str(row.get("sequence", "")),
                        "run_dir": str(row.get("run_dir", "")),
                        "source_batch_csv": str(row.get("source_batch_csv", "")) if str(row.get("source_batch_csv", "")) != "-" else "",
                        "source": str(row.get("source", "")),
                        "strategy": str(row.get("strategy", "")),
                        "priority_band": str(row.get("priority_band", "")),
                    }
                    for row in gate_launchable_rows
                ],
            ),
        )
    else:
        st.info("No selected peptides pass the launch readiness gate for this run yet. Clear the dry-run blockers above, then come back here to quick-launch the slate.")
    if (
        selected_candidate
        and str(selected_candidate.get("candidate_status", "")) == "Selected for MD"
        and str(selected_candidate.get("launch_ready", "")) == "yes"
        and str(selected_candidate.get("sequence", "")) in gate_launchable_sequences
    ):
        single_launch_readiness = build_button_readiness(
            state,
            "md-slate-run",
            rows=[selected_candidate],
            run_dir=str(selected_candidate.get("run_dir", "")),
            run_name=str(selected_candidate.get("run", "")),
        )
        _render_draft_action(
            st,
            label=f"Launch only {selected_candidate.get('sequence', '')} as a 1-peptide slate",
            command=f"dashboard-managed single-peptide MD slate for {selected_candidate.get('sequence', '')}",
            key_prefix=f"launch_single_md_slate_{selected_candidate.get('run_slug', '')}_{selected_candidate.get('sequence', '')}",
            what="Launch only the currently selected peptide, while still using the same slate runner, caps, and status tracking.",
            when="Use this when you want a single candidate to start moving immediately without waiting for the rest of the slate.",
            produces="A one-peptide MD slate that behaves exactly like the full slate path.",
            next_step="Approve the single-peptide slate once, then monitor it from MD Validation -> Slate monitor.",
            contract_id="md-slate-run",
            readiness=single_launch_readiness,
            on_submit=lambda candidate=selected_candidate: draft_md_slate_run_action(
                run_root=Path(str(state["run_root"])),
                run_dir=Path(str(candidate["run_dir"])),
                run_name=str(candidate.get("run", "")),
                peptides=[
                    {
                        "sequence": str(candidate.get("sequence", "")),
                        "run_dir": str(candidate.get("run_dir", "")),
                        "source_batch_csv": str(candidate.get("source_batch_csv", "")) if str(candidate.get("source_batch_csv", "")) != "-" else "",
                        "source": str(candidate.get("source", "")),
                        "strategy": str(candidate.get("strategy", "")),
                        "priority_band": str(candidate.get("priority_band", "")),
                    }
                ],
            ),
        )
    elif selected_candidate and str(selected_candidate.get("candidate_status", "")) == "Selected for MD":
        st.info(str(selected_candidate.get("launch_blocker", "")) or "This selected candidate is not launch-ready yet.")


def _render_md_slate_monitor(st, state: dict[str, object], ladder: dict[str, object]) -> None:
    run_dir = str(ladder.get("run_dir", ""))
    slates = [slate for slate in list(state.get("md_slates", [])) if _canonical_path(str(slate.get("run_dir", ""))) == _canonical_path(run_dir)]
    st.subheader("Slate monitor")
    st.caption("This board shows the supervised MD slate for this run: where each peptide is in the ladder, which stage is active, and whether anything is blocked or review-ready.")
    if not slates:
        st.info("No MD slate has been launched for this run yet. Use Peptides -> Candidate selection to snapshot a selected slate and launch it with one approval.")
        return
    options = [
        f"{_path_name(str(item.get('run_dir', '')))} [{str(item.get('slate_id', ''))}]"
        for item in slates
    ]
    selected_label = st.selectbox(
        "Choose MD slate",
        options,
        index=0,
        key=f"md_slate_select_{_path_name(run_dir)}",
    )
    slate = slates[options.index(selected_label)]
    execution_mode = str(slate.get("execution_mode", "live")).strip() or "live"
    utilization = build_bura_utilization_summary(Path(str(state["run_root"])), list(state.get("actions", [])), run_dir=run_dir)
    current_stage_counts = {key: 0 for key in SLATE_STAGE_CAPS}
    for peptide in list(slate.get("peptides", [])):
        if not isinstance(peptide, dict):
            continue
        current_stage = str(peptide.get("current_stage", "")).strip()
        if current_stage in current_stage_counts and str(peptide.get("status", "")) == "active":
            current_stage_counts[current_stage] += 1
    tracked_counts = utilization.get("tracked_external_counts", {}) if isinstance(utilization, dict) else {}
    _render_metric_cards(
        st,
        [
            ("Mode", execution_mode),
            ("Slate status", slate.get("effective_status", "-")),
            ("Peptides", slate.get("peptide_count", 0)),
            ("Blocked", slate.get("blocked_count", 0)),
            ("Recovery items", slate.get("exception_count", 0)),
            ("Review-ready", slate.get("review_ready_count", 0)),
            ("line_smoke active", f"{current_stage_counts.get('line_smoke', 0) + int(tracked_counts.get('line_smoke', 0) or 0)} / {SLATE_STAGE_CAPS['line_smoke']}"),
            ("production_smoke active", f"{current_stage_counts.get('production_smoke', 0) + int(tracked_counts.get('production_smoke', 0) or 0)} / {SLATE_STAGE_CAPS['production_smoke']}"),
            ("full active", f"{current_stage_counts.get('full', 0) + int(tracked_counts.get('full', 0) or 0)} / {SLATE_STAGE_CAPS['full']}"),
        ],
    )
    st.dataframe(
        [
            {
                **row,
                "stage_label": _friendly_md_profile(str(row.get("stage", ""))) if str(row.get("stage", "")) not in {"", "-"} else "-",
                "resource_request": _md_slate_resource_request(str(row.get("stage", ""))),
                "is_selected_peptide": "yes" if str(row.get("sequence", "")) == str(ladder.get("sequence", "")) else "",
            }
            for row in build_md_slate_monitor_rows(slate)
        ]
    )
    if execution_mode == "rehearsal":
        summary = slate.get("rehearsal_summary", {}) if isinstance(slate.get("rehearsal_summary", {}), dict) else {}
        st.info("This is a dashboard-local rehearsal slate. It did not submit, poll, cancel, or pull anything on BURA/SUPEK, and it did not assign `cgmd_label`.")
        if summary:
            st.dataframe(
                [
                    {
                        "completed": summary.get("completed", 0),
                        "blocked": summary.get("blocked", 0),
                        "review_ready": summary.get("review_ready", 0),
                        "max_line_smoke_active": summary.get("max_active_line_smoke", 0),
                        "max_production_smoke_active": summary.get("max_active_production_smoke", 0),
                        "max_full_active": summary.get("max_active_full", 0),
                        "touched_remote_clusters": summary.get("touched_remote_clusters", "no"),
                        "cgmd_label_assigned": summary.get("cgmd_label_assigned", "no"),
                    }
                ]
            )
        rehearsal_events = list(slate.get("rehearsal_events", [])) if isinstance(slate.get("rehearsal_events", []), list) else []
        if rehearsal_events:
            st.caption("Rehearsal event trace")
            st.dataframe(rehearsal_events[-30:])
    waiting_rows = [
        row
        for row in build_md_slate_monitor_rows(slate)
        if str(row.get("waiting_reason", "")) not in {"", "-"}
    ]
    if waiting_rows:
        st.info("Some peptides are waiting rather than blocked. The table below explains the current hold-up so you can tell the difference between normal pacing and a real failure.")
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "stage": _friendly_md_profile(str(row.get("stage", ""))) if str(row.get("stage", "")) not in {"", "-"} else "-",
                    "step": row.get("step", ""),
                    "waiting_reason": row.get("waiting_reason", ""),
                }
                for row in waiting_rows
            ]
        )
    tracked_external_rows = utilization.get("tracked_external_rows", []) if isinstance(utilization, dict) else []
    if tracked_external_rows:
        st.caption("Other dashboard-tracked BURA jobs under your user still count toward the local slate caps shown above.")
        st.dataframe(tracked_external_rows)
    control_cols = st.columns(4)
    if str(slate.get("effective_status", "")) == "paused":
        if control_cols[0].button("Resume slate", key=f"resume_slate_{slate.get('slate_id', '')}"):
            resume_md_slate(Path(str(state["run_root"])), str(slate.get("slate_id", "")))
            _stash_dashboard_flash(st, "success", "Resumed the MD slate.")
            _trigger_dashboard_rerun(st)
    else:
        if control_cols[0].button("Pause slate", key=f"pause_slate_{slate.get('slate_id', '')}"):
            pause_md_slate(Path(str(state["run_root"])), str(slate.get("slate_id", "")))
            _stash_dashboard_flash(st, "success", "Paused the MD slate. Running remote jobs were left alone.")
            _trigger_dashboard_rerun(st)
    selected_peptide = next(
        (item for item in list(slate.get("peptides", [])) if str(item.get("sequence", "")) == str(ladder.get("sequence", ""))),
        None,
    )
    if isinstance(selected_peptide, dict) and str(selected_peptide.get("status", "")) == "blocked":
        if control_cols[1].button("Retry blocked peptide", key=f"retry_slate_peptide_{slate.get('slate_id', '')}_{ladder.get('sequence', '')}"):
            retry_blocked_md_slate_peptide(
                Path(str(state["run_root"])),
                str(slate.get("slate_id", "")),
                str(ladder.get("sequence", "")),
            )
            _stash_dashboard_flash(st, "success", "Reset the blocked peptide so the slate can try that stage again.")
            _trigger_dashboard_rerun(st)
        if control_cols[2].button("Skip blocked peptide", key=f"skip_slate_peptide_{slate.get('slate_id', '')}_{ladder.get('sequence', '')}"):
            stop_blocked_md_slate_peptide(
                Path(str(state["run_root"])),
                str(slate.get("slate_id", "")),
                str(ladder.get("sequence", "")),
            )
            _stash_dashboard_flash(st, "success", "Skipped the blocked peptide and left the rest of the slate active.")
            _trigger_dashboard_rerun(st)


def _render_md_slate_recovery_actions(
    st,
    state: dict[str, object],
    exception_row: dict[str, object],
    *,
    key_prefix: str,
) -> None:
    run_root = Path(str(state["run_root"]))
    slate_id = str(exception_row.get("slate_id", "")).strip()
    sequence = str(exception_row.get("sequence", "")).strip()
    if not slate_id or not sequence:
        return
    cols = st.columns(4)
    if bool(exception_row.get("rebind_available", False)):
        if cols[0].button("Rebind latest tracked action", key=f"{key_prefix}_rebind"):
            rebind_md_slate_peptide(run_root, slate_id, sequence)
            _stash_dashboard_flash(st, "success", f"Rebound {sequence} to the latest tracked child action.")
            _trigger_dashboard_rerun(st)
    if bool(exception_row.get("recover_available", False)):
        if cols[1].button("Recover from last checkpoint", key=f"{key_prefix}_recover"):
            recover_md_slate_peptide(run_root, slate_id, sequence)
            _stash_dashboard_flash(st, "success", f"Reset {sequence} so the slate can retry the current stage cleanly.")
            _trigger_dashboard_rerun(st)
    if str(exception_row.get("exception_type", "")) == "blocked":
        if cols[2].button("Retry blocked peptide", key=f"{key_prefix}_retry"):
            retry_blocked_md_slate_peptide(run_root, slate_id, sequence)
            _stash_dashboard_flash(st, "success", f"Reset the blocked peptide {sequence} so the slate can try that stage again.")
            _trigger_dashboard_rerun(st)
        if cols[3].button("Skip blocked peptide", key=f"{key_prefix}_skip"):
            stop_blocked_md_slate_peptide(run_root, slate_id, sequence)
            _stash_dashboard_flash(st, "success", f"Skipped {sequence} and left the rest of the slate active.")
            _trigger_dashboard_rerun(st)


def _render_md_slate_recovery_center(st, state: dict[str, object], ladder: dict[str, object]) -> None:
    run_dir = _canonical_path(str(ladder.get("run_dir", "")))
    rows = [
        row
        for row in list(state.get("md_slate_exceptions", []))
        if _canonical_path(str(row.get("run_dir", ""))) == run_dir
    ]
    artifact_rows = [
        row
        for row in _artifact_rows_for_ladder(state, ladder)
        if str(row.get("verification_state", "")) == "Attention needed"
    ]
    st.subheader("Recovery center")
    st.caption("This workspace is for the peptides that need manual recovery help inside the MD slate: blocked stages, stale actions, overdue polls, drift between the slate and the tracked dashboard actions, or missing copied-back files.")
    if not rows and not artifact_rows:
        st.success("No current MD slate recovery or artifact integrity issues were detected for this run.")
        return
    if rows:
        summary = build_md_slate_exception_summary(rows)
        _render_metric_cards(
            st,
            [
                ("Recovery items", summary.get("total", 0)),
                ("Errors", summary.get("errors", 0)),
                ("Warnings", summary.get("warnings", 0)),
                ("Blocked peptides", summary.get("blocked", 0)),
                ("Stale / drifted", summary.get("stale", 0)),
            ],
        )
        st.dataframe(rows)
        preferred_sequence = str(ladder.get("sequence", "")).strip()
        focused_rows = [row for row in rows if str(row.get("sequence", "")) == preferred_sequence] or rows
        options = [
            f"{row.get('sequence', '')} [{row.get('exception_type', '')}]"
            for row in focused_rows
        ]
        selected_label = st.selectbox(
            "Inspect recovery item",
            options,
            index=0,
            key=f"md_recovery_item_{_path_name(run_dir)}",
        )
        selected_row = focused_rows[options.index(selected_label)]
        _render_recommended_card(
            st,
            eyebrow="MD recovery",
            title=str(selected_row.get("summary", "")) or "Inspect the selected recovery item",
            summary=str(selected_row.get("next_move", "")),
            why=(
                f"Stage: {_friendly_md_profile(str(selected_row.get('stage', '')))} | "
                f"Step: {selected_row.get('step', '-')} | "
                f"Last update: {selected_row.get('last_update_at', '-')}"
            ),
            do_now=(
                f"Sequence: {selected_row.get('sequence', '')} | "
                f"State: {selected_row.get('state', '-')} | "
                f"Tracked job: {selected_row.get('remote_job_id', '-') or '-'}"
            ),
            next_after="Use one of the recovery actions below, then keep watching the slate monitor to confirm the peptide starts moving again.",
        )
        _render_md_slate_recovery_actions(
            st,
            state,
            selected_row,
            key_prefix=f"md_recovery_{selected_row.get('slate_id', '')}_{selected_row.get('sequence', '')}",
        )
    if artifact_rows:
        _render_artifact_verification_workspace(
            st,
            artifact_rows,
            title="Artifact integrity issues",
            caption="These are the file-level blockers for the selected peptide/run: missing staging paths, incomplete copied-back outputs, or mismatched review/ingest files.",
            key_prefix=f"md_recovery_artifact_{_path_name(run_dir)}",
            render_export_pack=_render_export_pack,
        )


def _render_operations_md_slate_recovery_center(st, state: dict[str, object]) -> None:
    rows = list(state.get("md_slate_exceptions", []))
    artifact_rows = [
        row
        for row in list(state.get("artifact_verification", []))
        if str(row.get("verification_state", "")) == "Attention needed"
    ]
    utilization = state.get("bura_utilization", {}) if isinstance(state.get("bura_utilization", {}), dict) else {}
    st.subheader("MD slate recovery")
    st.caption("This is the operator-wide recovery board for slate issues across the visible workspace. Use it when you want one place to understand why a peptide is waiting, stale, blocked, or missing expected copied-back files.")
    if not rows and not artifact_rows:
        st.success("No current MD slate recovery or artifact integrity issues were detected in this workspace.")
        return
    if rows:
        summary = build_md_slate_exception_summary(rows)
        _render_metric_cards(
            st,
            [
                ("Recovery items", summary.get("total", 0)),
                ("Slates affected", summary.get("slates", 0)),
                ("Errors", summary.get("errors", 0)),
                ("Warnings", summary.get("warnings", 0)),
                ("Blocked peptides", summary.get("blocked", 0)),
                ("Stale / drifted", summary.get("stale", 0)),
            ],
        )
        st.dataframe(rows)
        options = [
            f"{row.get('run', '')} :: {row.get('sequence', '')} [{row.get('exception_type', '')}]"
            for row in rows
        ]
        selected_label = st.selectbox(
            "Inspect recovery item",
            options,
            index=0,
            key="operations_md_recovery_select",
        )
        selected_row = rows[options.index(selected_label)]
        _render_recommended_card(
            st,
            eyebrow="MD recovery",
            title=str(selected_row.get("summary", "")) or "Inspect the selected recovery item",
            summary=str(selected_row.get("next_move", "")),
            why=f"Run: {selected_row.get('run', '')} | Slate: {selected_row.get('slate_id', '')}",
            do_now=(
                f"Sequence: {selected_row.get('sequence', '')} | "
                f"Stage: {_friendly_md_profile(str(selected_row.get('stage', '')))} | "
                f"Step: {selected_row.get('step', '-')}"
            ),
            next_after="If this peptide really has no live job behind it, recover it from the last checkpoint. If the job is still alive, rebind it so the slate can pick the thread back up.",
        )
        _render_md_slate_recovery_actions(
            st,
            state,
            selected_row,
            key_prefix=f"operations_md_recovery_{selected_row.get('slate_id', '')}_{selected_row.get('sequence', '')}",
        )
    if artifact_rows:
        _render_artifact_verification_workspace(
            st,
            artifact_rows,
            title="Artifact integrity issues",
            caption="These rows show file-level problems that can make a transfer or finalize step look successful when the actual outputs are incomplete or mismatched.",
            key_prefix="operations_recovery_artifact",
            render_export_pack=_render_export_pack,
        )
    snapshot_summary = utilization.get("snapshot_summary", {}) if isinstance(utilization, dict) else {}
    st.caption(
        "BURA user queue snapshot: "
        f"running={snapshot_summary.get('running', 0)} | "
        f"pending={snapshot_summary.get('pending', 0)} | "
        f"held={snapshot_summary.get('held', 0)}"
    )


def _simulator_candidate_rows_for_run(state: dict[str, object], run_label: str) -> list[dict[str, object]]:
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    launchable = [
        row
        for row in rows
        if str(row.get("run", "")) == run_label
        and str(row.get("launch_ready", "")) == "yes"
    ]
    selected = [row for row in launchable if str(row.get("candidate_status", "")) == "Selected for MD"]
    if selected:
        return selected
    prioritized = [
        row
        for row in launchable
        if _candidate_priority_rank(str(row.get("priority_band", ""))) >= 2
    ]
    return prioritized or launchable


def _render_al_loop_simulator_panel(st, state: dict[str, object]) -> None:
    st.subheader("End-to-end AL loop simulator")
    st.caption(
        "Rehearse the whole operator loop locally: candidate snapshot -> MD slate rehearsal -> simulated review labels -> simulated ingest -> simulated retrain/propose. "
        "This is a safety drill, not a real model or MD run."
    )
    simulations = list(state.get("al_loop_simulations", []))
    simulation_rows = build_al_loop_simulation_rows(simulations)
    _render_metric_cards(
        st,
        [
            ("Simulations", len(simulations)),
            ("Next-round ready", sum(1 for row in simulations if str(row.get("stage", "")) == "next_round_ready")),
            ("Awaiting labels", sum(1 for row in simulations if str(row.get("stage", "")) == "md_rehearsed")),
            ("Safety mode", "local only"),
        ],
    )

    st.markdown("#### Start a local loop rehearsal")
    st.info(
        "The simulator creates dashboard-local state plus a dashboard-local MD rehearsal slate. It does not write real `cgmd_label`, `cgmd_ingest.csv`, model, metric, or next-batch files."
    )
    _render_action_contract_summary(st, "al-loop-simulation")
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    candidate_rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    launchable_rows = [row for row in candidate_rows if str(row.get("launch_ready", "")) == "yes"]
    run_options = sorted({str(row.get("run", "")) for row in launchable_rows if str(row.get("run", "")).strip()})
    if not run_options:
        st.warning("No launch-ready proposed/discovery candidates are visible. Use Peptides -> Candidate selection to select candidates or create dashboard-local MD batch rows first.")
    else:
        selected_run = st.selectbox(
            "Run to rehearse",
            run_options,
            index=0,
            key="al_loop_simulation_run",
        )
        selected_candidates = _simulator_candidate_rows_for_run(state, selected_run)[:6]
        st.caption(
            "The simulator prefers candidates already marked Selected for MD. If none are selected, it uses the highest-priority launch-ready candidates for this run."
        )
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "candidate_status": row.get("candidate_status", ""),
                    "source": row.get("source", ""),
                    "strategy": row.get("strategy", ""),
                    "priority_band": row.get("priority_band", ""),
                    "source_batch_csv": row.get("source_batch_csv", "-"),
                }
                for row in selected_candidates
            ]
        )
        if selected_candidates:
            run_dir = str(selected_candidates[0].get("run_dir", ""))
            run_key = _path_name(run_dir) or selected_run
            if st.button("Start loop rehearsal", key=f"start_al_loop_simulation_{run_key}"):
                try:
                    simulation = start_al_loop_simulation(
                        run_root=Path(str(state["run_root"])),
                        run_dir=Path(run_dir),
                        run_name=selected_run,
                        candidates=selected_candidates,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _stash_dashboard_flash(
                        st,
                        "success",
                        f"Started AL loop rehearsal {simulation.get('simulation_id', '')} and completed the local MD slate rehearsal.",
                    )
                    _trigger_dashboard_rerun(st)
        else:
            st.info("No launch-ready candidates are available for this run after filtering.")

    st.markdown("#### Simulation board")
    if simulation_rows:
        st.dataframe(simulation_rows)
    else:
        st.info("No AL loop simulations have been recorded yet.")
        return

    simulation_options = [
        f"{row.get('run', '')} | {row.get('stage', '')} | {row.get('simulation_id', '')}"
        for row in simulation_rows
    ]
    selected_label = st.selectbox(
        "Inspect simulation",
        simulation_options,
        index=0,
        key="al_loop_simulation_select",
    )
    selected_id = str(simulation_rows[simulation_options.index(selected_label)].get("simulation_id", ""))
    simulation = next((row for row in simulations if str(row.get("simulation_id", "")) == selected_id), simulations[0])
    candidate_snapshot = list(simulation.get("candidate_snapshot", []))
    labels = list(simulation.get("simulated_review_labels", []))
    ingest = simulation.get("simulated_ingest", {}) if isinstance(simulation.get("simulated_ingest", {}), dict) else {}
    retrain = simulation.get("simulated_retrain", {}) if isinstance(simulation.get("simulated_retrain", {}), dict) else {}
    slate_summary = simulation.get("slate_summary", {}) if isinstance(simulation.get("slate_summary", {}), dict) else {}

    _render_metric_cards(
        st,
        [
            ("Stage", simulation.get("stage", "-")),
            ("Candidates", len(candidate_snapshot)),
            ("Simulated labels", len(labels)),
            ("MD rehearsal slate", simulation.get("md_rehearsal_slate_id", "-")),
            ("Slate status", slate_summary.get("status", "-")),
        ],
    )
    st.write("Candidate snapshot")
    st.dataframe(candidate_snapshot)

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Inject review labels", key=f"inject_al_loop_labels_{selected_id}"):
            try:
                inject_simulated_review_labels(Path(str(state["run_root"])), selected_id)
            except Exception as exc:
                st.error(str(exc))
            else:
                _stash_dashboard_flash(st, "success", "Injected simulated review labels into local simulator state.")
                _trigger_dashboard_rerun(st)
    with action_cols[1]:
        if st.button("Simulate ingest", key=f"simulate_al_loop_ingest_{selected_id}"):
            try:
                simulate_loop_ingest(Path(str(state["run_root"])), selected_id)
            except Exception as exc:
                st.error(str(exc))
            else:
                _stash_dashboard_flash(st, "success", "Built the dry-run ingest plan without writing ingest files.")
                _trigger_dashboard_rerun(st)
    with action_cols[2]:
        if st.button("Simulate retrain/propose next round", key=f"simulate_al_loop_next_round_{selected_id}"):
            try:
                simulate_loop_retrain_and_propose(Path(str(state["run_root"])), selected_id)
            except Exception as exc:
                st.error(str(exc))
            else:
                _stash_dashboard_flash(st, "success", "Simulated retrain/propose readiness without running model code.")
                _trigger_dashboard_rerun(st)

    if labels:
        st.write("Simulated human review labels")
        st.dataframe(labels)
    if ingest:
        st.write("Dry-run ingest plan")
        st.dataframe(list(ingest.get("rows", [])) if isinstance(ingest.get("rows", []), list) else [])
        st.caption("Real path files that would be written")
        st.dataframe([{"path": path} for path in list(ingest.get("would_write", []))])
        st.code(str(ingest.get("would_run", "")), language="bash")
    if retrain:
        st.write("Dry-run retrain / next-round plan")
        st.dataframe([{"command": command} for command in list(retrain.get("would_run", []))])
        st.caption("Real path files/directories that would be updated")
        st.dataframe([{"path": path} for path in list(retrain.get("would_update", []))])
    st.write("Safety contract")
    safety = simulation.get("safety", {}) if isinstance(simulation.get("safety", {}), dict) else {}
    st.dataframe([{"guardrail": key, "value": value} for key, value in safety.items()])
    st.write("Event trace")
    st.dataframe(list(simulation.get("events", [])))


def _render_execution_readiness_panel(st, state: dict[str, object], *, compact: bool = False) -> None:
    execution_readiness = state.get("execution_readiness", {}) if isinstance(state.get("execution_readiness", {}), dict) else {}
    rows = list(execution_readiness.get("rows", [])) if isinstance(execution_readiness.get("rows", []), list) else []
    counts = execution_readiness.get("counts", {}) if isinstance(execution_readiness.get("counts", {}), dict) else {}
    if compact:
        rows = list(execution_readiness.get("top_rows", [])) if isinstance(execution_readiness.get("top_rows", []), list) else rows[:8]
    st.subheader("Execution readiness")
    st.caption(
        "This board combines action contracts with live dashboard state and answers: is it safe to run this now, what is blocking it, and what should you fix first."
    )
    _render_metric_cards(
        st,
        [
            ("Blocked", counts.get("blocked", 0)),
            ("Ready with caution", counts.get("caution", 0)),
            ("Ready", counts.get("ready", 0)),
            ("Tracked actions", counts.get("total", 0)),
        ],
    )
    if not rows:
        st.success("No execution-readiness rows are visible for the current workspace.")
        return
    if compact:
        st.dataframe(
            [
                {
                    "readiness": row.get("verdict", ""),
                    "action": row.get("label", ""),
                    "target": row.get("target", ""),
                    "summary": row.get("summary", ""),
                    "fix_now": row.get("fix_now", ""),
                }
                for row in rows
            ]
        )
        return
    focus_options = ["All", "Blocked", "Ready with caution", "Ready"]
    selected_focus = st.selectbox(
        "Execution readiness focus",
        focus_options,
        index=0,
        key="operations_execution_readiness_focus",
    )
    visible_rows = [
        row
        for row in rows
        if selected_focus == "All" or str(row.get("verdict", "")) == selected_focus
    ]
    st.dataframe(
        [
            {
                "readiness": row.get("verdict", ""),
                "view": row.get("view", ""),
                "action": row.get("label", ""),
                "run": row.get("run", "") or "-",
                "sequence": row.get("sequence", "") or "-",
                "target": row.get("target", ""),
                "summary": row.get("summary", ""),
                "fix_now": row.get("fix_now", ""),
            }
            for row in visible_rows
        ]
    )
    if not visible_rows:
        st.info("No execution-readiness rows match the current focus.")
        return
    labels = [
        f"{row.get('label', '')} | {row.get('target', '') or row.get('run', '') or row.get('sequence', '')}"
        for row in visible_rows
    ]
    selected_label = st.selectbox(
        "Inspect readiness row",
        labels,
        index=0,
        key="operations_execution_readiness_row",
    )
    selected_row = visible_rows[labels.index(selected_label)]
    blockers = list(selected_row.get("blockers", []))
    cautions = list(selected_row.get("cautions", []))
    _render_recommended_card(
        st,
        eyebrow="Execution readiness",
        title=f"{selected_row.get('label', '')}: {selected_row.get('verdict', '')}",
        summary=str(selected_row.get("summary", "")),
        why=(
            f"View: {selected_row.get('view', '')} | "
            f"Run: {selected_row.get('run', '') or '-'} | "
            f"Sequence: {selected_row.get('sequence', '') or '-'}"
        ),
        do_now=str(selected_row.get("fix_now", "")) or "No fix is needed right now.",
        next_after=(
            "Blockers: " + "; ".join(str(item) for item in blockers[:4])
            if blockers
            else ("Cautions: " + "; ".join(str(item) for item in cautions[:4]) if cautions else "No extra blockers or cautions are recorded for this row.")
        ),
    )


def _render_action_contracts_panel(st, state: dict[str, object]) -> None:
    st.subheader("Action contracts")
    st.caption(
        "These contracts describe the dashboard’s high-risk automation controls: what they write, whether approval is required, what must already be true, and the safest fallback if you are not ready to run them for real."
    )
    contracts = list_dashboard_action_contracts()
    actions = list(state.get("actions", []))
    simulations = list(state.get("al_loop_simulations", []))
    readiness_contract_rows = {
        str(row.get("contract_id", "")): row
        for row in list((state.get("execution_readiness", {}) or {}).get("contract_rows", []))
        if isinstance(row, dict)
    }
    contract_rows: list[dict[str, str]] = []
    for contract in contracts:
        latest_action = next(
            (
                action
                for action in actions
                if contract.action_kind and str(action.get("kind", "")) == contract.action_kind
            ),
            {},
        )
        latest_status = str(latest_action.get("status", "")) if isinstance(latest_action, dict) else ""
        latest_target = str(latest_action.get("title", "")) if isinstance(latest_action, dict) else ""
        if contract.contract_id == "al-loop-simulation" and simulations:
            latest_status = str(simulations[0].get("stage", "")) or latest_status
            latest_target = str(simulations[0].get("simulation_id", "")) or latest_target
        readiness_row = readiness_contract_rows.get(contract.contract_id, {})
        contract_rows.append(
            {
                "contract_id": contract.contract_id,
                "label": contract.label,
                "view": contract.view,
                "trigger": contract.trigger,
                "mode": "Action" if contract.mode == "action" else "Local-only rehearsal",
                "approval": contract.approval.replace("_", " "),
                "scope": contract.scope,
                "cluster": contract.cluster or "local",
                "execution_readiness": str(readiness_row.get("verdict", "")) or "-",
                "ready_now": str(readiness_row.get("ready_count", 0)),
                "caution": str(readiness_row.get("caution_count", 0)),
                "blocked": str(readiness_row.get("blocked_count", 0)),
                "latest_status": latest_status or "-",
                "latest_target": latest_target or "-",
            }
        )
    _render_metric_cards(
        st,
        [
            ("Tracked contracts", len(contracts)),
            ("Approval-gated", sum(1 for contract in contracts if contract.approval == "required")),
            ("Immediate local", sum(1 for contract in contracts if contract.approval == "not_required" and contract.scope == "local")),
            ("Remote controls", sum(1 for contract in contracts if contract.cluster in {"supek", "bura"})),
        ],
    )
    st.dataframe(contract_rows)
    labels = [f"{row['label']} [{row['contract_id']}]" for row in contract_rows]
    selected_label = st.selectbox(
        "Inspect contract",
        labels,
        index=0,
        key="dashboard_action_contract_select",
    )
    contract = contracts[labels.index(selected_label)]
    readiness_row = readiness_contract_rows.get(contract.contract_id, {})
    _render_recommended_card(
        st,
        eyebrow="Action contract",
        title=contract.label,
        summary=str(readiness_row.get("summary", "")).strip() or contract.postcondition,
        why=(
            f"Trigger: {contract.trigger} | Approval: {contract.approval.replace('_', ' ')} | "
            f"Scope: {contract.scope} | Cluster: {contract.cluster or 'local'}"
        ),
        do_now="Prerequisites: " + "; ".join(contract.prerequisites),
        next_after=f"Recovery path: {contract.recovery}",
    )
    if readiness_row:
        st.markdown(_readiness_badge_markup(str(readiness_row.get("verdict", ""))), unsafe_allow_html=True)
        st.caption(
            f"Ready now: {readiness_row.get('ready_count', 0)} | "
            f"Caution: {readiness_row.get('caution_count', 0)} | "
            f"Blocked: {readiness_row.get('blocked_count', 0)}"
        )
        if str(readiness_row.get("sample_target", "")).strip():
            st.caption(f"Most urgent visible context: {readiness_row.get('sample_target', '')}")
    st.write("Writes / side effects")
    st.dataframe([{"writes": item} for item in contract.writes])
    st.caption(f"Safer option: {contract.safer_option}")
    if contract.action_kind:
        matching_actions = [
            {
                "id": str(action.get("id", "")),
                "title": str(action.get("title", "")),
                "status": str(action.get("status", "")),
                "related_run": _path_name(str(action.get("related_run", ""))) or str(action.get("related_run", "")),
                "sequence": str(action.get("related_sequence", "")) or "-",
                "output_path": str(action.get("output_path", "")) or "-",
                "created_at": str(action.get("created_at", "")),
            }
            for action in actions
            if str(action.get("kind", "")) == contract.action_kind
        ]
        if matching_actions:
            st.write("Recent matching actions")
            st.dataframe(matching_actions[:8])
        else:
            st.info("No matching dashboard actions are currently recorded for this contract.")
        matching_readiness = [
            {
                "readiness": str(row.get("verdict", "")),
                "run": str(row.get("run", "")) or "-",
                "sequence": str(row.get("sequence", "")) or "-",
                "target": str(row.get("target", "")) or "-",
                "summary": str(row.get("summary", "")),
                "fix_now": str(row.get("fix_now", "")),
            }
            for row in list((state.get("execution_readiness", {}) or {}).get("rows", []))
            if isinstance(row, dict) and str(row.get("contract_id", "")) == contract.contract_id
        ]
        if matching_readiness:
            st.write("Visible readiness contexts")
            st.dataframe(matching_readiness[:10])
    elif contract.contract_id == "al-loop-simulation":
        if simulations:
            st.write("Recent rehearsal simulations")
            st.dataframe(
                [
                    {
                        "simulation_id": str(item.get("simulation_id", "")),
                        "run": _path_name(str(item.get("run_dir", ""))) or str(item.get("run_name", "")),
                        "stage": str(item.get("stage", "")),
                        "status": str(item.get("status", "")),
                        "updated_at": str(item.get("updated_at", "")),
                    }
                    for item in simulations[:8]
                ]
            )
        else:
            st.info("No AL loop rehearsals have been recorded yet.")


def _clean_reconciliation_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text in {"", "-"} else text


def _remote_reconciliation_primary_job_id(row: dict[str, object]) -> str:
    text = _clean_reconciliation_value(row.get("remote_job_id", ""))
    if not text:
        return ""
    token = text.split(",", 1)[0].split("/", 1)[0].strip()
    if " " in token:
        token = token.split(" ", 1)[0]
    return token.strip("()")


def _remote_reconciliation_id(row: dict[str, object]) -> str:
    existing = _clean_reconciliation_value(row.get("reconciliation_id", ""))
    if existing:
        return existing
    fallback = "::".join(
        [
            str(row.get("issue_type", "")),
            str(row.get("cluster", "")),
            str(row.get("remote_job_id", "")),
            str(row.get("run_dir", "")),
            str(row.get("campaign_dir", "")),
            str(row.get("sequence", "")),
            str(row.get("stage", "")),
        ]
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, fallback).hex[:16]


def _remote_reconciliation_path(row: dict[str, object], key: str) -> Path | None:
    value = _clean_reconciliation_value(row.get(key, ""))
    return Path(value) if value else None


def _remote_reconciliation_job_candidates(row: dict[str, object]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    raw_candidates = row.get("candidate_jobs", [])
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            job_id = _clean_reconciliation_value(item.get("job_id", ""))
            if not job_id:
                continue
            candidates.append(
                {
                    "job_id": job_id,
                    "campaign": _clean_reconciliation_value(item.get("campaign", "")),
                    "campaign_dir": _clean_reconciliation_value(item.get("campaign_dir", "")),
                    "dashboard_state": _clean_reconciliation_value(item.get("dashboard_state", "")),
                }
            )
    if not candidates:
        for token in _clean_reconciliation_value(row.get("remote_job_id", "")).split(","):
            job_id = _clean_reconciliation_value(token)
            if job_id:
                candidates.append(
                    {
                        "job_id": job_id,
                        "campaign": _clean_reconciliation_value(row.get("campaign", "")),
                        "campaign_dir": _clean_reconciliation_value(row.get("campaign_dir", "")),
                        "dashboard_state": _clean_reconciliation_value(row.get("dashboard_state", "")),
                    }
                )
    deduped: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        deduped.setdefault(candidate["job_id"], candidate)
    return list(deduped.values())


def _mark_remote_reconciliation_stale(run_root: Path, row: dict[str, object]) -> dict[str, object]:
    cluster = _clean_reconciliation_value(row.get("cluster", "")).lower()
    campaign_dir = _remote_reconciliation_path(row, "campaign_dir")
    run_dir = _remote_reconciliation_path(row, "run_dir")
    target_path = campaign_dir if cluster == "bura" else run_dir
    if target_path is None:
        raise ValueError("This reconciliation row does not have a local target path that can be marked stale.")
    update_sync_status(
        run_root,
        cluster=cluster,
        target_key=str(target_path),
        status="stale",
        related_run=str(run_dir or ""),
        related_campaign=str(campaign_dir or ""),
        related_sequence=_clean_reconciliation_value(row.get("sequence", "")),
        remote_job_id=None,
        metadata={
            "marked_stale_at": datetime.now().isoformat(timespec="seconds"),
            "previous_dashboard_state": _clean_reconciliation_value(row.get("dashboard_state", "")),
            "stale_remote_job_id": _remote_reconciliation_primary_job_id(row),
            "stale_reason": _clean_reconciliation_value(row.get("summary", "")),
        },
    )
    return {
        "id": f"remote-reconcile-stale:{_remote_reconciliation_id(row)}",
        "title": "Mark remote reconciliation item stale",
        "status": "manual_override",
    }


def _acknowledge_remote_reconciliation_item(run_root: Path, row: dict[str, object]) -> dict[str, object]:
    reconciliation_id = _remote_reconciliation_id(row)
    ignore_dashboard_reconciliation_item(run_root, reconciliation_id)
    return {
        "id": f"remote-reconcile-ack:{reconciliation_id}",
        "title": "Acknowledge remote reconciliation item",
        "status": "manual_override",
    }


def _remote_reconciliation_display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "severity": row.get("severity", ""),
            "issue": row.get("issue", ""),
            "cluster": row.get("cluster", ""),
            "run": row.get("run", ""),
            "sequence": row.get("sequence", ""),
            "stage": row.get("stage", ""),
            "remote_job_id": row.get("remote_job_id", ""),
            "queue_state": row.get("queue_state", ""),
            "dashboard_state": row.get("dashboard_state", ""),
            "recommended_recovery": row.get("recommended_recovery", ""),
            "summary": row.get("summary", ""),
        }
        for row in rows
    ]


def _render_remote_reconciliation_actions(
    st,
    state: dict[str, object],
    row: dict[str, object],
    *,
    key_prefix: str,
) -> None:
    run_root = Path(str(state["run_root"]))
    cluster = _clean_reconciliation_value(row.get("cluster", "")).lower()
    issue_type = _clean_reconciliation_value(row.get("issue_type", ""))
    sequence = _clean_reconciliation_value(row.get("sequence", ""))
    run_dir = _remote_reconciliation_path(row, "run_dir")
    campaign_dir = _remote_reconciliation_path(row, "campaign_dir")
    remote_job_id = _remote_reconciliation_primary_job_id(row)
    profile = get_cluster_profile(state.get("profiles", {}), cluster) if cluster in SUPPORTED_CLUSTERS else None
    rendered_any = False

    if cluster in SUPPORTED_CLUSTERS and profile is None:
        st.warning(f"{cluster.upper()} profile is not configured, so only local reconciliation actions are available here.")

    if cluster == "bura" and profile and campaign_dir and sequence:
        _render_launch_action(
            st,
            label="Fetch BURA logs for this campaign",
            command=f"ssh {profile.get('host', 'bura')} 'tail latest logs for {sequence}'",
            key_prefix=f"{key_prefix}_fetch_logs",
            button_text="Fetch logs",
            on_submit=lambda: queue_bura_fetch_logs_action(
                run_root=run_root,
                campaign_dir=campaign_dir,
                sequence=sequence,
                profile=profile,
                related_run=str(run_dir or ""),
            ),
            what="Queues a read-only log fetch against the campaign/package folders.",
            when="Use this first when a tracked BURA job disappeared, failed, or the slate/sync records disagree.",
            produces="A dashboard action with the latest remote log excerpt.",
            next_step="Use the logs to decide whether to pull artifacts, retry, or mark the stale job as evidence.",
        )
        rendered_any = True
    elif cluster == "supek" and profile and run_dir:
        _render_launch_action(
            st,
            label="Fetch SUPEK logs for this run",
            command=f"ssh {profile.get('host', 'supek')} 'tail latest logs for {_path_name(run_dir)}'",
            key_prefix=f"{key_prefix}_fetch_logs",
            button_text="Fetch logs",
            on_submit=lambda: queue_supek_fetch_logs_action(
                run_root=run_root,
                run_dir=run_dir,
                profile=profile,
            ),
            what="Queues a read-only SUPEK log fetch for the run.",
            when="Use this before deciding whether a missing or failed tracked SUPEK job is recoverable.",
            produces="A dashboard action with the latest remote log excerpt.",
            next_step="If outputs exist, pull artifacts; if the logs show failure, retry from the safe workflow step.",
        )
        rendered_any = True

    if issue_type in {"tracked_running_dashboard_waiting", "slate_job_mismatch", "slate_job_missing_sync"} and remote_job_id:
        if cluster == "bura" and profile and campaign_dir and sequence:
            _render_launch_action(
                st,
                label="Poll BURA queue again",
                command=f"ssh {profile.get('host', 'bura')} 'squeue -u {profile.get('username', '<user>')}'",
                key_prefix=f"{key_prefix}_poll",
                button_text="Poll again",
                on_submit=lambda: queue_bura_poll_action(
                    run_root=run_root,
                    campaign_dir=campaign_dir,
                    sequence=sequence,
                    profile=profile,
                    related_run=str(run_dir or ""),
                    remote_job_id=remote_job_id,
                ),
                what="Queues a read-only scheduler poll tied to this tracked BURA job id.",
                when="Use this when the scheduler and cockpit states are probably just out of sync.",
                produces="A refreshed queue snapshot and sync status after the worker runs.",
                next_step="If the job is no longer visible after polling, fetch logs or pull artifacts before advancing.",
            )
            rendered_any = True
        elif cluster == "supek" and profile and run_dir:
            _render_launch_action(
                st,
                label="Poll SUPEK queue again",
                command=f"ssh {profile.get('host', 'supek')} 'qstat -u {profile.get('username', '<user>')}'",
                key_prefix=f"{key_prefix}_poll",
                button_text="Poll again",
                on_submit=lambda: queue_supek_poll_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    profile=profile,
                    remote_job_id=remote_job_id,
                ),
                what="Queues a read-only scheduler poll tied to this tracked SUPEK job id.",
                when="Use this when SUPEK says running but the local record still says submitted.",
                produces="A refreshed queue snapshot and sync status after the worker runs.",
                next_step="If the job is gone, fetch logs and decide whether to pull or retry.",
            )
            rendered_any = True

    if issue_type in {"tracked_missing_from_queue", "tracked_failed_in_queue"}:
        if cluster == "bura" and profile and campaign_dir and sequence:
            _render_draft_action(
                st,
                label="Pull BURA artifacts into safe staging",
                command=f"scp -r {profile.get('host', 'bura')}:<campaign>/packages/{sequence} <dashboard staging>",
                key_prefix=f"{key_prefix}_pull",
                on_submit=lambda: draft_bura_pull_package_action(
                    run_root=run_root,
                    campaign_dir=campaign_dir,
                    sequence=sequence,
                    profile=profile,
                    related_run=str(run_dir or ""),
                ),
                what="Creates an approval-gated artifact pull draft. It does not finalize or label the peptide.",
                when="Use this only after logs or queue state suggest outputs may exist.",
                produces="A copied-back package in dashboard staging once approved and run.",
                next_step="Finalize the MD stage manually after checking that the staged package looks complete.",
            )
            rendered_any = True
        elif cluster == "supek" and profile and run_dir:
            _render_draft_action(
                st,
                label="Pull SUPEK run artifacts into safe staging",
                command=f"scp -r {profile.get('host', 'supek')}:<remote run> <dashboard staging>",
                key_prefix=f"{key_prefix}_pull",
                on_submit=lambda: draft_supek_pull_artifacts_action(
                    run_root=run_root,
                    run_dir=run_dir,
                    profile=profile,
                ),
                what="Creates an approval-gated SUPEK artifact pull draft.",
                when="Use this only after logs or queue state suggest outputs may exist.",
                produces="A copied-back run artifact bundle in dashboard staging once approved and run.",
                next_step="Inspect the artifacts before rerunning or advancing model workflow steps.",
            )
            rendered_any = True

        _render_launch_action(
            st,
            label="Mark tracked job stale locally",
            command="dashboard local: clear the tracked job id and mark this sync record stale",
            key_prefix=f"{key_prefix}_mark_stale",
            button_text="Mark stale",
            on_submit=lambda: _mark_remote_reconciliation_stale(run_root, row),
            what="Updates only the local dashboard sync record. It does not touch the remote cluster.",
            when="Use this after logs or queue evidence show the tracked job is no longer the live authority.",
            produces="A stale transfer/sync row that preserves the old job id in metadata.",
            next_step="Retry from the last safe stage or recover the slate peptide when you are ready.",
        )
        rendered_any = True

    if issue_type == "duplicate_tracked_job" and cluster == "bura" and profile and sequence:
        candidates = _remote_reconciliation_job_candidates(row)
        if candidates:
            candidate_labels = [
                f"{item['job_id']} :: {item.get('campaign', '') or _path_name(item.get('campaign_dir', '')) or 'campaign'}"
                for item in candidates
            ]
            selected_label = st.selectbox(
                "Duplicate job to cancel",
                candidate_labels,
                index=0,
                key=f"{key_prefix}_cancel_candidate",
            )
            selected_candidate = candidates[candidate_labels.index(selected_label)]
            selected_campaign_dir = Path(selected_candidate.get("campaign_dir", "")) if selected_candidate.get("campaign_dir", "") else campaign_dir
            if selected_campaign_dir:
                _render_draft_action(
                    st,
                    label="Create cancel draft for selected duplicate BURA job",
                    command=f"scancel {selected_candidate['job_id']}",
                    key_prefix=f"{key_prefix}_cancel",
                    on_submit=lambda: draft_bura_cancel_action(
                        run_root=run_root,
                        campaign_dir=selected_campaign_dir,
                        sequence=sequence,
                        profile=profile,
                        related_run=str(run_dir or ""),
                        remote_job_id=selected_candidate["job_id"],
                    ),
                    what="Creates an approval-gated cancel draft for the duplicate job you selected.",
                    when="Use this only after choosing which tracked job is accidental.",
                    produces="A draft action. Nothing is cancelled until you approve and run it.",
                    next_step="Keep the intended job tracked, then poll/rebind if the slate points at the wrong id.",
                )
                rendered_any = True

    if issue_type in {"slate_job_mismatch", "slate_job_missing_sync"}:
        slate_id = _clean_reconciliation_value(row.get("slate_id", ""))
        if slate_id and sequence:
            if st.button("Rebind latest tracked action", key=f"{key_prefix}_rebind"):
                rebind_md_slate_peptide(run_root, slate_id, sequence)
                _stash_dashboard_flash(st, "success", f"Rebound {sequence} to the latest tracked child action.")
                _trigger_dashboard_rerun(st)
            rendered_any = True

    if issue_type == "external_queue_job":
        _render_launch_action(
            st,
            label="Acknowledge external queue job context",
            command="dashboard local: hide this external queue row until the reconciliation id changes",
            key_prefix=f"{key_prefix}_ack",
            button_text="Acknowledge",
            on_submit=lambda: _acknowledge_remote_reconciliation_item(run_root, row),
            what="Stores a local dashboard acknowledgement for this external queue row.",
            when="Use this when the job is not part of the cockpit workflow or does not need action.",
            produces="The row is hidden from the default reconciliation board on the next refresh.",
            next_step="It will reappear only if its reconciliation identity changes.",
        )
        rendered_any = True

    if not rendered_any:
        st.info("No one-click recovery action is safe for this row yet. Use the linked run or slate view and inspect the latest logs before changing state.")


def _render_remote_reconciliation_recovery_panel(
    st,
    state: dict[str, object],
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    st.markdown("#### Remote job reconciliation")
    st.caption(
        "This compares dashboard-tracked SUPEK/BURA jobs with the latest queue snapshots. "
        "The selected row below now acts like a recovery assistant: safe read-only checks are queued directly, "
        "and potentially disruptive operations stay approval-gated."
    )
    if not rows:
        if list(state.get("snapshots", [])):
            st.success("No remote job reconciliation issues were detected from the latest queue snapshots.")
        else:
            st.info("No cluster snapshots have been captured yet, so there is nothing to reconcile.")
        return
    _render_metric_cards(
        st,
        [
            ("Reconciliation rows", summary.get("total", len(rows))),
            ("Errors", summary.get("errors", 0)),
            ("Warnings", summary.get("warnings", 0)),
            ("Tracked missing", summary.get("tracked_missing", 0)),
            ("Duplicate jobs", summary.get("duplicates", 0)),
            ("External BURA jobs", summary.get("external_bura_jobs", 0)),
        ],
    )
    st.dataframe(_remote_reconciliation_display_rows(rows))
    options = [
        (
            f"{str(row.get('severity', '')).upper()} :: {str(row.get('cluster', '')).upper()} :: "
            f"{row.get('issue', '')} :: "
            f"{row.get('sequence', row.get('run', '-'))} :: {row.get('remote_job_id', '-')}"
        )
        for row in rows
    ]
    selected_label = st.selectbox(
        "Inspect reconciliation item",
        options,
        index=0,
        key="operations_remote_reconciliation_select",
    )
    selected_row = rows[options.index(selected_label)]
    stage = _clean_reconciliation_value(selected_row.get("stage", ""))
    _render_recommended_card(
        st,
        eyebrow="Remote recovery",
        title=str(selected_row.get("issue", "")) or "Inspect the selected remote issue",
        summary=str(selected_row.get("summary", "")),
        why=(
            f"Cluster: {str(selected_row.get('cluster', '')).upper()} | "
            f"Queue: {selected_row.get('queue_state', '-')} | "
            f"Dashboard: {selected_row.get('dashboard_state', '-')}"
        ),
        do_now=str(selected_row.get("recommended_recovery", "")) or "Inspect before changing state",
        next_after=(
            f"Target: {selected_row.get('sequence', '-') or selected_row.get('run', '-')} | "
            f"Stage: {_friendly_md_profile(stage) if stage else '-'} | "
            f"Next: {selected_row.get('next_move', '')}"
        ),
    )
    _render_remote_reconciliation_actions(
        st,
        state,
        selected_row,
        key_prefix=f"reconcile_{_remote_reconciliation_id(selected_row)}",
    )


def _render_candidate_selection_workspace(st, state: dict[str, object]) -> None:
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    st.subheader("Candidate selection")
    st.caption("Use this workspace to decide which proposed or discovery peptides should move into MD, which ones should wait, and which ones should stay out of the thesis validation set.")
    if not rows:
        st.info("No proposed or discovery candidates are visible for the current workspace / filters.")
        return
    focus_options = [
        "Shortlist manager",
        "Selected for MD next",
        "Undecided pool",
        "Discovery-backed candidates",
        "Deferred / rejected",
        "Already in MD",
        "All candidates",
    ]
    source_options = [
        "All evidence sources",
        "Proposal + discovery",
        "Any discovery evidence",
        "Proposal-only evidence",
    ]
    priority_options = [
        "All priorities",
        "Top priority only",
        "High + Top",
        "Review soon or higher",
    ]
    selected_focus = st.selectbox(
        "Candidate queue focus",
        focus_options,
        index=0,
        key="candidate_selection_focus",
    )
    selected_source_focus = st.selectbox(
        "Source evidence focus",
        source_options,
        index=0,
        key="candidate_selection_source_focus",
    )
    selected_priority_focus = st.selectbox(
        "Priority focus",
        priority_options,
        index=0,
        key="candidate_selection_priority_focus",
    )
    filtered_rows = _candidate_focus_rows(
        rows,
        focus=selected_focus,
        source_focus=selected_source_focus,
        priority_focus=selected_priority_focus,
    )
    _render_metric_cards(
        st,
        [
            ("Undecided", sum(1 for row in rows if str(row.get("candidate_status", "")) == "Undecided")),
            ("Selected for MD", sum(1 for row in rows if str(row.get("candidate_status", "")) == "Selected for MD")),
            ("Already in MD", sum(1 for row in rows if str(row.get("candidate_status", "")) == "Already in MD")),
            ("Visible rows", len(filtered_rows)),
        ],
    )
    selected_rows = [
        row
        for row in rows
        if str(row.get("candidate_status", "")) == "Selected for MD"
    ]
    if selected_rows:
        st.markdown("#### Selected for the next MD batch")
        st.caption("These peptides already have an explicit thesis decision saying they should move into MD next. Use the slate launch area below when you want the cockpit to take over the routine MD ladder work with one approval.")
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "run": row.get("run", ""),
                    "source": row.get("source", ""),
                    "strategy": row.get("strategy", ""),
                    "priority_band": row.get("priority_band", ""),
                    "decision_title": row.get("decision_title", "") or "-",
                    "next_action": row.get("next_action", ""),
                }
                for row in selected_rows[:12]
            ]
        )
    recommended_rows = [
        row
        for row in rows
        if str(row.get("candidate_status", "")) == "Undecided"
        and _candidate_priority_rank(str(row.get("priority_band", ""))) >= 2
    ]
    if not recommended_rows:
        recommended_rows = [
            row
            for row in rows
            if str(row.get("candidate_status", "")) == "Undecided"
        ]
    recommended_rows = recommended_rows[:8]
    if recommended_rows:
        st.markdown("#### Recommended next MD candidates")
        st.caption("These are the highest-ranked undecided candidates based on where they appeared, whether they are backed by both proposal and discovery, and whether the strategy is uncertainty/ensemble-driven.")
        st.dataframe(
            [
                {
                    "sequence": row.get("sequence", ""),
                    "run": row.get("run", ""),
                    "priority_band": row.get("priority_band", ""),
                    "source": row.get("source", ""),
                    "strategy": row.get("strategy", ""),
                    "why_ranked": row.get("priority_reason", ""),
                    "next_action": row.get("next_action", ""),
                }
                for row in recommended_rows
            ]
        )
    shortlist_summary = _candidate_shortlist_summary_rows(filtered_rows)
    if shortlist_summary:
        st.write("Run-level shortlist overview")
        st.dataframe(shortlist_summary)
    if not filtered_rows:
        st.info("No candidates match the current shortlist filters. Relax the focus or evidence filters to inspect a wider pool.")
        return
    st.write("Visible candidate queue")
    st.dataframe(
        [
            {
                "sequence": row.get("sequence", ""),
                "run": row.get("run", ""),
                "source_focus": _candidate_source_bucket(str(row.get("source", ""))),
                "source": row.get("source", ""),
                "strategy": row.get("strategy", ""),
                "priority_band": row.get("priority_band", ""),
                "candidate_status": row.get("candidate_status", ""),
                "md_stage": row.get("md_stage", ""),
                "md_status": row.get("md_status", ""),
                "last_decision": row.get("last_decision", "-"),
                "why_ranked": row.get("priority_reason", ""),
                "next_action": row.get("next_action", ""),
            }
            for row in filtered_rows
        ]
    )
    visible_undecided_rows = [
        row for row in filtered_rows if str(row.get("candidate_status", "")) == "Undecided"
    ]
    if visible_undecided_rows:
        st.markdown("#### Batch shortlist action")
        st.caption("Use this when the visible shortlist already reflects one clear thesis decision. It applies the same decision to the currently visible undecided candidates and records the batch in the local action log.")
        batch_decision_types = ["select_candidate_for_md", "defer_candidate", "reject_candidate"]
        batch_option_labels = [_decision_type_label(item) for item in batch_decision_types]
        batch_option_map = dict(zip(batch_option_labels, batch_decision_types))
        selected_batch_label = st.selectbox(
            "Batch decision for visible undecided candidates",
            batch_option_labels,
            index=0,
            key="candidate_batch_decision_type",
        )
        batch_decision_type = batch_option_map[selected_batch_label]
        batch_spec = DECISION_TYPE_INFO.get(batch_decision_type, {})
        batch_title = st.text_input(
            "Batch decision title",
            value=str(batch_spec.get("default_title", "")) or "Apply candidate shortlist decision",
            key="candidate_batch_decision_title",
        )
        batch_rationale = st.text_area(
            "Why are we applying this batch decision?",
            value=(
                "These visible candidates belong to the same shortlist decision and should move together instead of being handled one-by-one."
            ),
            key="candidate_batch_decision_rationale",
            height=110,
        )
        batch_next_step = st.text_input(
            "What should happen after this batch decision?",
            value=str(batch_spec.get("default_next_step", "")),
            key="candidate_batch_decision_next_step",
        )
        batch_preview = ", ".join(str(row.get("sequence", "")) for row in visible_undecided_rows[:6])
        if len(visible_undecided_rows) > 6:
            batch_preview += f", +{len(visible_undecided_rows) - 6} more"
        st.write(f"Visible undecided shortlist: {len(visible_undecided_rows)} peptide(s)")
        st.caption(f"Preview: {batch_preview}")
        confirm_batch = st.checkbox(
            "Confirm batch decision for the visible undecided candidates",
            value=False,
            key="confirm_candidate_batch_decision",
        )
        if st.button("Apply batch shortlist decision", key="candidate_batch_decision_save"):
            if not confirm_batch:
                st.warning("Confirm the batch shortlist decision first.")
            elif not batch_title.strip() or not batch_rationale.strip():
                st.warning("Add both a batch title and a rationale before saving the shortlist decision.")
            else:
                submit_bulk_candidate_decision_action(
                    run_root=Path(str(state["run_root"])),
                    items=[
                        {
                            "sequence": str(row.get("sequence", "")),
                            "run_dir": str(row.get("run_dir", "")),
                            "run_name": str(row.get("run", "")),
                            "source": str(row.get("source", "")),
                            "strategy": str(row.get("strategy", "")),
                            "priority_band": str(row.get("priority_band", "")),
                            "next_step_override": str(row.get("next_action", "")) if batch_decision_type == "select_candidate_for_md" else batch_next_step,
                        }
                        for row in visible_undecided_rows
                    ],
                    decision_type=batch_decision_type,
                    title=batch_title,
                    rationale=batch_rationale,
                    next_step=batch_next_step,
                    evidence_prefix=(
                        f"focus={selected_focus} | source_focus={selected_source_focus} | priority_focus={selected_priority_focus}"
                    ),
                )
                st.success("Saved the visible shortlist decision and recorded it as a finished local dashboard action.")
                st.rerun()
    options = [
        f"{row.get('sequence', '')} | {row.get('run', '')} | {row.get('source', '')}"
        for row in filtered_rows
    ]
    selected_label = st.selectbox(
        "Choose candidate",
        options,
        index=0,
        key="peptides_candidate_target",
    )
    candidate = filtered_rows[options.index(selected_label)]
    st.markdown("#### Candidate briefing")
    _render_metric_cards(
        st,
        [
            ("Candidate status", candidate.get("candidate_status", "-")),
            ("Source", candidate.get("source", "-")),
            ("Priority", candidate.get("priority_band", "-")),
            ("Launch ready", candidate.get("launch_ready", "-")),
            ("MD stage", candidate.get("md_stage", "-")),
            ("Last decision", candidate.get("last_decision", "-")),
        ],
    )
    st.caption(f"Why this candidate is ranked here: {candidate.get('priority_reason', 'general candidate pool')}")
    rationale_text = str(candidate.get("decision_rationale", "")).strip()
    if str(candidate.get("candidate_status", "")) == "Already in MD":
        st.success("This peptide is already in the guided MD ladder. Use MD Validation to continue the remote/local execution path; record the decision here only if you want the thesis rationale captured.")
    elif str(candidate.get("candidate_status", "")) == "Selected for MD":
        if str(candidate.get("launch_ready", "")) == "yes":
            st.info("This candidate has already been marked for MD validation. The next operational step is to open MD Validation and prepare or continue the ladder.")
        else:
            st.warning(str(candidate.get("launch_blocker", "")) or "This candidate still needs a source batch CSV before it can enter MD.")
    elif str(candidate.get("candidate_status", "")) == "Deferred":
        st.info("This candidate is currently deferred. Keep the rationale current if you want to explain later why it was not prioritized.")
    elif str(candidate.get("candidate_status", "")) == "Rejected":
        st.warning("This candidate is currently marked as rejected for validation. Keep the rationale clear so the choice is traceable in thesis reporting.")
    else:
        st.info("This candidate is still undecided. Use the editor below to record whether it should move into MD, wait for later, or stay out of the validation set.")
    st.write("Selected candidate summary")
    st.dataframe(
        [
            {
                "sequence": candidate.get("sequence", ""),
                "run": candidate.get("run", ""),
                "source_focus": _candidate_source_bucket(str(candidate.get("source", ""))),
                "source": candidate.get("source", ""),
                "proposal_round": candidate.get("proposal_round", "-"),
                "strategy": candidate.get("strategy", "-"),
                "discovery_stage": candidate.get("discovery_stage", "-"),
                "candidate_status": candidate.get("candidate_status", ""),
                "source_batch_csv": candidate.get("source_batch_csv", "-"),
                "source_batch_kind": candidate.get("source_batch_kind", "-"),
                "launch_blocker": candidate.get("launch_blocker", "-"),
                "md_status": candidate.get("md_status", "-"),
                "next_action": candidate.get("next_action", ""),
            }
        ]
    )
    if str(candidate.get("launch_ready", "")) != "yes":
        _render_md_source_batch_export_action(
            st,
            state,
            rows=[candidate],
            key_prefix=f"export_md_source_batch_single_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
            label=f"Create a dashboard-local MD batch row for {candidate.get('sequence', '')}",
            next_step="After that local batch row exists, you can mark the peptide for MD or launch it as a 1-peptide slate.",
        )
    _render_selected_md_slate_panel(
        st,
        state,
        selected_rows=selected_rows,
        selected_candidate=candidate,
    )
    st.markdown("#### Candidate decision editor")
    decision_types = _decision_scope_options("candidate")
    option_labels = [_decision_type_label(item) for item in decision_types]
    option_map = dict(zip(option_labels, decision_types))
    default_decision = str(candidate.get("last_decision", "-"))
    selected_decision_label = option_labels[0]
    for label in option_labels:
        if label == default_decision:
            selected_decision_label = label
            break
    selected_label = st.selectbox(
        "Decision type",
        option_labels,
        index=option_labels.index(selected_decision_label),
        key=f"candidate_decision_type_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
    )
    decision_type = option_map[selected_label]
    spec = DECISION_TYPE_INFO.get(decision_type, {})
    title = st.text_input(
        "Short decision title",
        value=str(candidate.get("decision_title", "")).strip() or str(spec.get("default_title", "")),
        key=f"candidate_decision_title_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
    )
    rationale = st.text_area(
        "Why are we making this candidate decision?",
        value=rationale_text,
        key=f"candidate_decision_rationale_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
        height=110,
    )
    evidence = st.text_input(
        "Evidence / shortlist note",
        value=f"source={candidate.get('source', '-')} | strategy={candidate.get('strategy', '-')} | md_status={candidate.get('md_status', '-')}",
        key=f"candidate_decision_evidence_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
    )
    next_step = st.text_input(
        "What should happen next?",
        value=str(spec.get("default_next_step", "")),
        key=f"candidate_decision_next_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}",
    )
    if st.button("Save candidate decision", key=f"candidate_decision_save_{candidate.get('run_slug', '')}_{candidate.get('sequence', '')}"):
        if not title.strip() or not rationale.strip():
            st.warning("Add both a short title and a rationale before saving the candidate decision.")
        else:
            add_dashboard_decision(
                Path(str(state["run_root"])),
                scope="candidate",
                decision_type=decision_type,
                title=title,
                rationale=rationale,
                run_dir=str(candidate.get("run_dir", "")),
                run_name=str(candidate.get("run", "")),
                sequence=str(candidate.get("sequence", "")),
                evidence=evidence,
                next_step=next_step,
            )
            st.success("Saved candidate decision to the local dashboard decision log.")
            st.rerun()
    candidate_decisions = _decision_log_rows(
        state,
        run_names={str(candidate.get("run", ""))},
        sequence=str(candidate.get("sequence", "")),
        scope="candidate",
    )
    if candidate_decisions:
        st.markdown("#### Recent candidate decisions")
        st.dataframe(candidate_decisions[:8])


def _bulk_review_note_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text == "-" else text


def _render_bulk_review_ingest_workspace(st, state: dict[str, object]) -> None:
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    all_rows = list(inventory.get("review_pipeline", [])) if isinstance(inventory.get("review_pipeline", []), list) else []
    st.subheader("Bulk review & ingest")
    st.caption("Work through review-ready peptides from one queue. Update several human labels at once, save only the rows that changed, then use the detailed editor below if one peptide needs closer attention.")
    if not all_rows:
        st.info("No peptides are currently waiting in the review / ingest handoff.")
        return
    review_state_options = ["All", "Needs review / label", "Needs review evidence", "Reviewed for reporting", "Ready for ingest", "Already ingested"]
    selected_review_state = st.selectbox(
        "Review queue focus",
        review_state_options,
        index=0,
        key="bulk_review_state_filter",
    )
    rows = [
        row
        for row in all_rows
        if selected_review_state == "All" or str(row.get("review_state", "")) == selected_review_state
    ]
    _render_metric_cards(
        st,
        [
            ("Needs review", sum(1 for row in all_rows if str(row.get("review_state", "")) == "Needs review / label")),
            ("Reporting-only", sum(1 for row in all_rows if str(row.get("review_state", "")) == "Reviewed for reporting")),
            ("Ready for ingest", sum(1 for row in all_rows if str(row.get("review_state", "")) == "Ready for ingest")),
            ("Unique runs", len({str(row.get("run", "")) for row in all_rows})),
            ("Visible rows", len(rows)),
        ],
    )
    if rows:
        st.dataframe(rows)
    else:
        st.info("No review rows match the current queue focus.")
        return
    st.markdown("#### Bulk review board")
    st.caption("The board below shows the first visible review / ingest rows for the current workspace. Saving writes only the rows whose label or notes changed. Use the detailed peptide workspace when a newly structured review still needs confidence, tags, or an evidence summary.")
    board_limit = min(6, len(rows))
    board_items: list[tuple[dict[str, object], dict[str, object]]] = []
    unresolved_rows: list[str] = []
    for row in rows[:board_limit]:
        ladder = _find_ladder_for_review_row(
            state,
            run_name=str(row.get("run", "")),
            sequence=str(row.get("sequence", "")),
        )
        if ladder is None or not ladder.get("full"):
            unresolved_rows.append(f"{row.get('sequence', '')} | {row.get('run', '')}")
            continue
        board_items.append((row, ladder))
    if unresolved_rows:
        st.warning(
            "Some review rows could not be opened in the bulk board: "
            + ", ".join(unresolved_rows[:4])
            + ("." if len(unresolved_rows) <= 4 else f", and {len(unresolved_rows) - 4} more.")
        )
    pending_edits: list[dict[str, object]] = []
    ready_for_ingest_items: list[dict[str, object]] = []
    invalid_ready_rows: list[str] = []
    if board_items:
        for index, (row, ladder) in enumerate(board_items, start=1):
            full_item = ladder["full"]
            current_label = str(full_item.get("cgmd_label", "")).strip()
            if current_label not in {"", "0", "1"}:
                current_label = ""
            current_notes = _bulk_review_note_value(full_item.get("review_notes", ""))
            run_display = str(ladder.get("run_display_name", ladder.get("run_name", "")))
            stage_label = _friendly_md_status(str(full_item.get("job_root_status", "")))
            current_state = str(row.get("review_state", ""))
            current_ingest = str(row.get("ingest_csv", "-"))
            ingest_support = str(row.get("ingest_support", "AL-ingestable"))
            st.write(f"{index}. {ladder['sequence']} | {run_display}")
            st.caption(
                f"Campaign: {full_item.get('campaign', '-')} | Review state: {current_state} | "
                f"Full-analysis status: {stage_label} | Ingest support: {ingest_support} | Ingest CSV: {current_ingest} | "
                f"Evidence: {review_evidence_status(full_item).get('state', '-')}"
            )
            edit_columns = st.columns(2)
            selected_label = edit_columns[0].selectbox(
                "Human review label",
                options=["", "0", "1"],
                index=["", "0", "1"].index(current_label),
                key=f"bulk_review_label_{ladder['sequence']}_{index}",
            )
            edited_notes = edit_columns[1].text_input(
                "Review notes",
                value=current_notes,
                key=f"bulk_review_notes_{ladder['sequence']}_{index}",
            )
            label_changed = selected_label != current_label
            notes_changed = edited_notes != current_notes
            if label_changed or notes_changed:
                pending_edits.append(
                    {
                        "review_csv": Path(str(full_item["review_path"])),
                        "sequence": str(ladder["sequence"]),
                        "cgmd_label": str(selected_label),
                        "review_notes": str(edited_notes),
                        "related_run": str(ladder["run_dir"]),
                        "related_campaign": str(full_item["campaign_dir"]),
                    }
                )
                st.caption("Pending change: this row will be saved.")
            else:
                st.caption("No change pending for this row.")
            resolved_label = str(selected_label).strip()
            resolved_notes = str(edited_notes).strip()
            current_ingest_path = _ingest_csv_path(str(full_item["campaign_dir"]))
            draft_review = {
                **full_item,
                "cgmd_label": resolved_label,
                "review_notes": resolved_notes,
            }
            draft_status = review_evidence_status(draft_review)
            if resolved_label in {"0", "1"} and ingest_support == "AL-ingestable" and not current_ingest_path.exists() and bool(draft_status.get("ingest_ready", False)):
                ready_for_ingest_items.append(
                    {
                        "campaign_dir": Path(str(full_item["campaign_dir"])),
                        "review_csv": Path(str(full_item["review_path"])),
                        "sequence": str(ladder["sequence"]),
                        "related_run": str(ladder["run_dir"]),
                        "review_notes": resolved_notes,
                    }
                )
            elif resolved_label in {"0", "1"} and ingest_support == "AL-ingestable" and not current_ingest_path.exists():
                invalid_ready_rows.append(f"{ladder['sequence']} ({_review_evidence_missing_text(draft_status)})")
            st.divider()
    else:
        st.info("No visible review rows could be opened in the bulk editor yet.")

    if pending_edits:
        st.write("Pending bulk review saves")
        st.dataframe(
            [
                {
                    "sequence": str(edit["sequence"]),
                    "label": str(edit["cgmd_label"] or "<empty>"),
                    "review_notes": str(edit["review_notes"] or "-"),
                    "campaign": _path_name(str(edit["related_campaign"])),
                }
                for edit in pending_edits
            ]
        )
        bulk_command = (
            "bulk update md_review.csv for "
            + ", ".join(f"{edit['sequence']}={edit['cgmd_label'] or '<empty>'}" for edit in pending_edits[:6])
            + ("" if len(pending_edits) <= 6 else f", +{len(pending_edits) - 6} more")
        )
        _render_launch_action(
            st,
            label="Save visible review edits",
            command=bulk_command,
            key_prefix="bulk_review_save",
            button_text="Save visible edits",
            what="Write the changed review labels and notes back into the visible campaign `md_review.csv` files in one local batch.",
            when="Use this after you have inspected several full-analysis peptides and want the queue to reflect those review decisions immediately.",
            produces="Updated `md_review.csv` files plus one recorded dashboard action summarizing the saved review decisions.",
            next_step="Rows that now have a final label can move straight to Create ingest CSV or Ingest returned labels next.",
            on_submit=lambda pending_edits=pending_edits: submit_bulk_update_md_review_action(
                run_root=Path(str(state["run_root"])),
                edits=pending_edits,
            ),
        )
    else:
        st.info("No visible review edits are pending right now. Change one or more labels or notes above, then save them together.")

    if ready_for_ingest_items:
        st.markdown("#### Visible rows ready for ingest CSV creation")
        st.dataframe(
            [
                {
                    "sequence": str(item["sequence"]),
                    "campaign": _path_name(str(item["campaign_dir"])),
                    "review_notes": str(item["review_notes"] or "-"),
                }
                for item in ready_for_ingest_items
            ]
        )
        if invalid_ready_rows:
            st.warning(
                "Complete review evidence before batch ingest creation for: "
                + ", ".join(invalid_ready_rows[:6])
                + ("" if len(invalid_ready_rows) <= 6 else f", +{len(invalid_ready_rows) - 6} more")
            )
        elif pending_edits:
            st.info("Save the pending review edits first, then create ingest CSVs for the rows that are ready.")
        else:
            _render_launch_action(
                st,
                label="Create ingest CSVs for visible ready rows",
                command="bulk create cgmd_ingest.csv for visible ready peptides",
                key_prefix="bulk_make_ingest",
                button_text="Create visible ingest CSVs",
                what="Turn every visible, fully reviewed peptide without an ingest file into a local `cgmd_ingest.csv` in one batch.",
                when="Use this after the labels and review evidence are final and you want to move several peptides back toward the model together.",
                produces="One `cgmd_ingest.csv` per visible ready campaign plus a recorded local dashboard action.",
                next_step="Switch to Model Workflow and run Ingest returned labels for the affected parent runs.",
                on_submit=lambda ready_for_ingest_items=ready_for_ingest_items: submit_bulk_make_md_ingest_action(
                    run_root=Path(str(state["run_root"])),
                    items=ready_for_ingest_items,
                ),
            )

    st.markdown("#### Detailed peptide workspace")
    st.caption("Use this when one peptide needs a closer look, the exact review checklist, or the fast handoff into `cgmd_ingest.csv`.")
    options = [
        f"{row.get('sequence', '')} | {row.get('run', '')} | {row.get('campaign', '')}"
        for row in rows
    ]
    selected_row_label = st.selectbox(
        "Choose peptide from review / ingest queue",
        options,
        index=0,
        key="peptides_bulk_review_target",
    )
    selected_row = rows[options.index(selected_row_label)]
    ladder = _find_ladder_for_review_row(
        state,
        run_name=str(selected_row.get("run", "")),
        sequence=str(selected_row.get("sequence", "")),
    )
    if ladder is None:
        st.warning("The selected review queue row could not be matched back to a visible peptide ladder.")
        return
    st.write("Selected review target")
    st.dataframe(
        [
            {
                "sequence": selected_row.get("sequence", ""),
                "run": selected_row.get("run", ""),
                "campaign": selected_row.get("campaign", ""),
                "review_state": selected_row.get("review_state", ""),
                "next_action": selected_row.get("next_action", ""),
            }
        ]
    )
    _render_review_workspace(st, ladder, run_root=Path(str(state["run_root"])), state=state)
    if str(selected_row.get("review_state", "")) == "Ready for ingest":
        st.markdown("#### Fast handoff")
        _render_make_ingest_action(
            st,
            ladder,
            run_root=Path(str(state["run_root"])),
            key_prefix=f"bulk_ingest_{ladder['sequence']}",
            state=state,
        )


def _render_ladder_workflow_macros(st, state: dict[str, object], ladder: dict[str, object]) -> None:
    st.markdown("#### Guided ladder runner")
    st.caption("Use this runner when you want the cockpit to show the whole peptide-validation path, highlight the current checkpoint, and advance one safe step at a time.")
    macro_key = _recommended_ladder_macro_key(ladder)
    current = ladder.get("current")
    run_root = Path(str(state["run_root"]))
    bura_profile = get_cluster_profile(state.get("profiles", {}), "bura")
    current_campaign_dir = str((current or {}).get("campaign_dir", ""))
    ladder_rows = _ladder_plan_rows(ladder)
    ladder_current = next((row for row in ladder_rows if str(row.get("status", "")) == "current"), {})
    _render_plan_checkpoint_table(
        st,
        title="Peptide ladder checkpoints",
        rows=ladder_rows,
        summary="This path runs from local preparation through remote BURA execution, local finalization, human review, and eventual model feedback.",
    )
    ladder_blocker, ladder_blocker_level = _ladder_block_reason(state, ladder)
    _render_runner_memory_panel(
        st,
        current_checkpoint=str(ladder_current.get("checkpoint", "")),
        events=_progress_events_for_context(
            state,
            scope="peptide",
            plan_kind="ladder",
            run_dir=str(ladder["run_dir"]),
            sequence=str(ladder["sequence"]),
            campaign_dir=current_campaign_dir,
        ),
        blocker=ladder_blocker,
        blocker_level=ladder_blocker_level,
    )

    if macro_key == "prepare-local" and ladder["next_profile"] and ladder["source_batch_csv"]:
        default_campaign_name = _default_campaign_name(ladder)
        prepare_readiness = build_button_readiness(state, "prepare-md-stage", ladder=ladder)
        _render_launch_action(
            st,
            label=f"Advance ladder plan: Prepare {ladder['next_profile_label']}",
            command=(
                "python -m active_learning_thesis prepare-md-stage "
                f"--run-dir {_quote_path(ladder['run_dir'])} "
                f"--batch-csv {_quote_path(ladder['source_batch_csv'])} "
                f"--sequence {ladder['sequence']} "
                f"--campaign {default_campaign_name} "
                f"--md-profile {ladder['next_profile']}"
            ),
            key_prefix=f"macro_prepare_{ladder['sequence']}",
            button_text="Advance plan",
            what="Create the next missing local MD campaign package using the current ladder recommendation.",
            when="Use this when the next rung has not been prepared yet.",
            produces="A new local campaign folder ready for BURA staging.",
            next_step="The campaign can then be uploaded to BURA from the next macro.",
            contract_id="prepare-md-stage",
            readiness=prepare_readiness,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]): _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Prepare local campaign",
                action_label=f"Advance ladder plan: Prepare {ladder['next_profile_label']}",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                note="Prepared the next ladder campaign from the guided runner.",
            ),
            on_submit=lambda campaign_name=default_campaign_name: submit_prepare_md_stage_action(
                run_root=run_root,
                run_dir=Path(str(ladder["run_dir"])),
                batch_csv=Path(str(ladder["source_batch_csv"])),
                sequence=str(ladder["sequence"]),
                campaign=campaign_name,
                md_profile=str(ladder["next_profile"]),
                cluster=str(ladder.get("cluster", "bura") or "bura"),
                reuse_pdb_from=str(ladder.get("reuse_pdb_from", "")) or None,
                exclude_nodes=str(ladder.get("exclude_nodes", "")),
            ),
        )
    elif macro_key == "upload-bura" and current and bura_profile is not None:
        upload_readiness = build_button_readiness(state, "bura-upload-campaign", ladder=ladder)
        _render_draft_action(
            st,
            label="Advance ladder plan: Upload campaign to BURA",
            command=f"scp -r {_quote_path(current['campaign_dir'])} {bura_profile['username']}@{bura_profile['host']}:<campaign root>",
            key_prefix=f"macro_bura_upload_{ladder['sequence']}",
            what="Draft the next required remote step so the current campaign becomes staged on BURA.",
            when="Use this when the campaign exists locally but has not been uploaded yet.",
            produces="An approval-gated upload draft.",
            next_step="Once approved, the normalization/preflight macro becomes the next safe step.",
            contract_id="bura-upload-campaign",
            readiness=upload_readiness,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]), campaign_dir=current_campaign_dir: _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Upload to BURA",
                action_label="Advance ladder plan: Upload campaign to BURA",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                campaign_dir=campaign_dir,
                note="Created the BURA upload draft from the guided ladder runner.",
            ),
            on_submit=lambda current=current: draft_bura_upload_campaign_action(
                run_root=run_root,
                campaign_dir=Path(str(current["campaign_dir"])),
                sequence=str(ladder["sequence"]),
                profile=bura_profile,
                related_run=str(ladder["run_dir"]),
            ),
        )
    elif macro_key == "normalize-bura" and current and bura_profile is not None:
        normalize_readiness = build_button_readiness(state, "bura-normalize-scripts", ladder=ladder)
        _render_draft_action(
            st,
            label="Advance ladder plan: Start BURA pre-submit sequence",
            command='find . -type f -name "*.sh" -exec dos2unix {} \\; && chmod u+x *.sh',
            key_prefix=f"macro_bura_normalize_{ladder['sequence']}",
            what="Start the remote pre-submit sequence with the first required step: script normalization.",
            when="Use this when the campaign is already staged on BURA and needs the normalization -> preflight -> submit chain.",
            produces="An approval-gated normalization draft.",
            next_step="After normalization, run the preflight action and then submit the chain.",
            contract_id="bura-normalize-scripts",
            readiness=normalize_readiness,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]), campaign_dir=current_campaign_dir: _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Normalize / preflight / submit",
                action_label="Advance ladder plan: Start BURA pre-submit sequence",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                campaign_dir=campaign_dir,
                note="Created the normalization draft from the guided ladder runner.",
            ),
            on_submit=lambda current=current: draft_bura_normalize_action(
                run_root=run_root,
                campaign_dir=Path(str(current["campaign_dir"])),
                sequence=str(ladder["sequence"]),
                profile=bura_profile,
                related_run=str(ladder["run_dir"]),
            ),
        )
    elif macro_key == "poll-bura" and current and bura_profile is not None:
        poll_readiness = build_button_readiness(state, "bura-poll-squeue", ladder=ladder)
        _render_launch_action(
            st,
            label="Advance ladder plan: Monitor active BURA campaign",
            command=f"ssh {bura_profile['username']}@{bura_profile['host']} squeue -u {bura_profile['username']}",
            key_prefix=f"macro_bura_poll_{ladder['sequence']}",
            button_text="Advance plan",
            what="Poll the tracked BURA job for the current peptide ladder.",
            when="Use this while the remote chain is queued or running.",
            produces="An updated queue snapshot and ladder sync state.",
            next_step="If the chain has finished, fetch logs or copy the outputs back.",
            contract_id="bura-poll-squeue",
            readiness=poll_readiness,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]), campaign_dir=current_campaign_dir: _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Monitor remote queue",
                action_label="Advance ladder plan: Monitor active BURA campaign",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                campaign_dir=campaign_dir,
                note="Polled the tracked BURA campaign from the guided ladder runner.",
            ),
            on_submit=lambda current=current: queue_bura_poll_action(
                run_root=run_root,
                campaign_dir=Path(str(current["campaign_dir"])),
                sequence=str(ladder["sequence"]),
                profile=bura_profile,
                related_run=str(ladder["run_dir"]),
                remote_job_id=str(current.get("remote_job_id", "")),
            ),
        )
    elif macro_key == "finalize-local" and current:
        finalize_readiness = build_button_readiness(state, "finalize-md-stage", ladder=ladder)
        finalize_command = f"python -m active_learning_thesis finalize-md-stage --campaign-dir {_quote_path(current['campaign_dir'])}"
        if current.get("local_stage_path"):
            finalize_command += f" --staged-package-dir {_quote_path(current['local_stage_path'])}"
        _render_launch_action(
            st,
            label=f"Advance ladder plan: Finalize {current['campaign']}",
            command=finalize_command,
            key_prefix=f"macro_finalize_{ladder['sequence']}",
            button_text="Advance plan",
            what="Parse the latest local/staged MD outputs and advance the ladder state.",
            when="Use this after copying outputs back or when the dashboard says the next step is local re-parse/finalization.",
            produces="Updated `md_review.csv`, ladder status, and next recommendation.",
            next_step="The peptide either advances to the next rung or becomes ready for review / ingest.",
            contract_id="finalize-md-stage",
            readiness=finalize_readiness,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]), campaign_dir=current_campaign_dir, campaign=str(current["campaign"]): _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Finalize locally",
                action_label=f"Advance ladder plan: Finalize {campaign}",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                campaign_dir=campaign_dir,
                note="Finalized the copied-back ladder outputs from the guided runner.",
            ),
            on_submit=lambda current=current: submit_finalize_md_stage_action(
                run_root=run_root,
                campaign_dir=Path(str(current["campaign_dir"])),
                sequence=str(ladder["sequence"]),
                related_run=str(ladder["run_dir"]),
                staged_package_dir=str(current.get("local_stage_path", "")) or None,
            ),
        )
    elif macro_key == "make-ingest":
        _render_make_ingest_action(
            st,
            ladder,
            run_root=run_root,
            key_prefix=f"macro_make_ingest_{ladder['sequence']}",
            state=state,
            after_submit=lambda action, run_dir=str(ladder["run_dir"]), sequence=str(ladder["sequence"]), campaign_dir=str((ladder.get("full") or {}).get("campaign_dir", "")): _record_runner_progress(
                run_root,
                scope="peptide",
                plan_kind="ladder",
                checkpoint="Create ingest CSV",
                action_label=f"Create ingest CSV for {ladder['sequence']}",
                action=action,
                run_dir=run_dir,
                sequence=sequence,
                campaign_dir=campaign_dir,
                note="Created the ingest CSV from the guided ladder runner.",
            ),
        )
    else:
        st.info("No single macro can safely advance this peptide right now. The next step still needs a human review decision or a manual choice between available actions.")


def _render_peptide_ladder(st, state: dict[str, object]) -> None:
    _render_md_validation_view_impl(st, state, ns=sys.modules[__name__])


def _render_operations(st, state: dict[str, object]) -> None:
    _render_operations_view_impl(st, state, ns=sys.modules[__name__])


def _arrow_safe_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


def _arrow_safe_dataframe_value(value: object) -> object:
    if pd is None:
        return value
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, list) and all(isinstance(row, dict) for row in value):
        frame = pd.DataFrame(value)
    else:
        return value
    for column in frame.columns:
        if str(frame[column].dtype) == "object":
            frame[column] = frame[column].map(_arrow_safe_cell)
    return frame


def _install_arrow_safe_dataframe(st) -> None:
    if hasattr(st, "_view") or bool(getattr(st, "_dashboard_arrow_safe_dataframe", False)):
        return
    dataframe = getattr(st, "dataframe", None)
    if not callable(dataframe):
        return

    def _safe_dataframe(value, *args, **kwargs):
        return dataframe(_arrow_safe_dataframe_value(value), *args, **kwargs)

    try:
        setattr(st, "dataframe", _safe_dataframe)
        setattr(st, "_dashboard_arrow_safe_dataframe", True)
    except Exception:
        pass


def _dashboard_guided_mode(state: dict[str, object]) -> bool:
    return str(state.get("workflow_mode", DEFAULT_DASHBOARD_WORKFLOW_MODE)) == "Guided thesis mode"


def _dashboard_trusted_mode(state: dict[str, object]) -> bool:
    return str(state.get("approval_mode", DEFAULT_DASHBOARD_APPROVAL_MODE)) == "Trusted actions"


def _dashboard_smart_polling_mode(state: dict[str, object]) -> bool:
    return str(state.get("refresh_mode", DEFAULT_DASHBOARD_REFRESH_MODE)) == "Smart polling"


def _render_guided_next_action_panel(st, state: dict[str, object], *, view: str) -> None:
    if not _dashboard_guided_mode(state):
        return
    pending_approvals = [
        action for action in state.get("actions", []) if action.get("status") in APPROVAL_PENDING_STATUSES
    ]
    active_remote = [
        action
        for action in state.get("actions", [])
        if action.get("scope") in {"supek", "bura"} and action.get("status") in ACTIVE_ACTION_STATUSES
    ]
    if pending_approvals or active_remote:
        render_guided_state_panel(st, state, view=view)
        if pending_approvals:
            st.warning("Blocking issue: one or more actions need explicit approval before the next thesis step is safe.")
        elif active_remote:
            action = active_remote[0]
            st.info(f"Remote work in progress: {action.get('title', 'Remote action')} is {action.get('status', 'active')}.")
        return
    render_guided_state_panel(st, state, view=view)


def _render_inline_approval_panel(st, state: dict[str, object], *, key_prefix: str) -> None:
    if not _dashboard_guided_mode(state):
        return
    approval_actions = [
        action for action in state.get("actions", []) if action.get("status") in APPROVAL_PENDING_STATUSES
    ]
    if not approval_actions:
        return
    st.warning(f"{len(approval_actions)} action(s) still need explicit approval. Destructive or unknown-risk actions are intentionally not trusted-inline.")
    with st.expander("Inline approval cards", expanded=True):
        _render_action_history(
            st,
            actions=approval_actions[:4],
            run_root=str(state.get("run_root", "")),
            key_prefix=key_prefix,
        )


def render_dashboard(st, state: dict[str, object], *, refresh_seconds: int = 0) -> None:
    _install_arrow_safe_dataframe(st)
    run_root_path = Path(str(state["run_root"]))
    preferences = load_dashboard_preferences(run_root_path)
    preferred_ui_mode = str(preferences.get("ui_mode", DEFAULT_DASHBOARD_UI_MODE))
    preferred_workflow_mode = str(preferences.get("workflow_mode", DEFAULT_DASHBOARD_WORKFLOW_MODE))
    preferred_approval_mode = str(preferences.get("approval_mode", DEFAULT_DASHBOARD_APPROVAL_MODE))
    preferred_refresh_mode = str(preferences.get("refresh_mode", DEFAULT_DASHBOARD_REFRESH_MODE))
    if not bool(preferences.get("exists", False)):
        save_dashboard_preferences(run_root_path, preferences)
        preferences = load_dashboard_preferences(run_root_path)
    st.set_page_config(page_title="Thesis Dashboard", layout="wide")
    session_state = _dashboard_session_state(st)
    skip_preference_save = bool(session_state.get("dashboard_skip_preference_save", False))
    if str(session_state.get("dashboard_ui_mode", "")) not in DASHBOARD_UI_MODES:
        session_state["dashboard_ui_mode"] = preferred_ui_mode
    if str(session_state.get("dashboard_workflow_mode", "")) not in DASHBOARD_WORKFLOW_MODES:
        session_state["dashboard_workflow_mode"] = preferred_workflow_mode
    if str(session_state.get("dashboard_approval_mode", "")) not in DASHBOARD_APPROVAL_MODES:
        session_state["dashboard_approval_mode"] = preferred_approval_mode
    if str(session_state.get("dashboard_refresh_mode", "")) not in DASHBOARD_REFRESH_MODES:
        session_state["dashboard_refresh_mode"] = preferred_refresh_mode
    ui_mode = _dashboard_ui_mode(st, default=preferred_ui_mode)
    inject_dashboard_theme(st, ui_mode=ui_mode)
    render_dashboard_hero(st, ui_mode=ui_mode)
    st.caption(f"Run root: {state['run_root']}")
    st.caption(f"Generated at: {state['generated_at']}")
    _render_dashboard_flash(st)

    st.sidebar.markdown("### Navigation")
    view = _persisted_choice(
        st,
        st.sidebar.radio,
        label="View",
        options=VIEW_NAMES,
        key="dashboard_view",
        query_key="view",
        default=VIEW_NAMES[0],
        write_query=False,
    )
    workspace_scope = _persisted_choice(
        st,
        st.sidebar.radio,
        label="Workspace",
        options=WORKSPACE_SCOPES,
        key="dashboard_workspace_scope",
        query_key="workspace",
        default=WORKSPACE_SCOPES[0],
        write_query=False,
    )
    st.sidebar.markdown("### Display")
    selected_ui_mode = _persisted_choice(
        st,
        st.sidebar.radio,
        label="UI mode",
        options=DASHBOARD_UI_MODES,
        key="dashboard_ui_mode",
        query_key="ui_mode",
        default=ui_mode,
        read_query=False,
        write_query=False,
    )
    if selected_ui_mode != preferred_ui_mode and not skip_preference_save:
        save_dashboard_preferences(run_root_path, {**preferences, "ui_mode": selected_ui_mode})
    if selected_ui_mode != ui_mode:
        _trigger_dashboard_rerun(st)
        return
    if selected_ui_mode == "Stable mode":
        st.sidebar.info("Stable mode uses simpler native Streamlit layout and lighter styling to reduce foggy or duplicated rendering glitches.")
    else:
        st.sidebar.caption("Rich mode keeps the fuller cockpit styling and custom visual cards.")
    selected_workflow_mode = _persisted_choice(
        st,
        st.sidebar.radio,
        label="Workflow mode",
        options=DASHBOARD_WORKFLOW_MODES,
        key="dashboard_workflow_mode",
        query_key="workflow_mode",
        default=preferred_workflow_mode,
        read_query=False,
        write_query=False,
    )
    selected_approval_mode = _persisted_choice(
        st,
        st.sidebar.radio,
        label="Approval mode",
        options=DASHBOARD_APPROVAL_MODES,
        key="dashboard_approval_mode",
        query_key="approval_mode",
        default=preferred_approval_mode,
        read_query=False,
        write_query=False,
    )
    selected_refresh_mode = _persisted_choice(
        st,
        st.sidebar.radio,
        label="Refresh mode",
        options=DASHBOARD_REFRESH_MODES,
        key="dashboard_refresh_mode",
        query_key="refresh_mode",
        default=preferred_refresh_mode,
        read_query=False,
        write_query=False,
    )
    if (
        selected_workflow_mode != preferred_workflow_mode
        or selected_approval_mode != preferred_approval_mode
        or selected_refresh_mode != preferred_refresh_mode
    ) and not skip_preference_save:
        preferences = {
            **preferences,
            "ui_mode": selected_ui_mode,
            "workflow_mode": selected_workflow_mode,
            "approval_mode": selected_approval_mode,
            "refresh_mode": selected_refresh_mode,
        }
        save_dashboard_preferences(run_root_path, preferences)
    st.sidebar.caption(
        "Guided + Trusted is the beginner thesis cockpit. Expert + Strict keeps the old full-control workflow."
    )
    default_refresh = refresh_seconds if refresh_seconds > 0 else 0
    selected_refresh = 0 if selected_refresh_mode == "Smart polling" else _persisted_refresh_seconds(st, default_refresh)
    st.sidebar.caption(
        "Smart polling: use local status buttons near SUPEK/BURA jobs."
        if selected_refresh_mode == "Smart polling"
        else f"Auto refresh: {'Off' if selected_refresh <= 0 else f'requested every {selected_refresh}s'}"
    )
    st.sidebar.button("Refresh now", key="dashboard_manual_refresh")
    if selected_refresh > 0:
        st.sidebar.info("Timed auto refresh is currently running in safe manual mode to avoid the foggy / duplicated Streamlit rendering bug on Windows.")

    run_options = ["All", *[str(run.get("run_display_name", run.get("run_name", ""))) for run in state.get("runs", [])]]
    sequence_options = ["All", *sorted({str(peptide["sequence"]) for peptide in state.get("peptides", [])})]
    profile_filter_map = {
        "All": "All",
        _friendly_md_profile("line_smoke"): "line_smoke",
        _friendly_md_profile("production_smoke"): "production_smoke",
        _friendly_md_profile("full"): "full",
    }
    profile_options = list(profile_filter_map)
    status_values = sorted(
        {
            str(campaign.get("job_root_status", ""))
            for peptide in state.get("peptides", [])
            for campaign in peptide.get("campaigns", [])
        }
        | {
            str(campaign.get("sync_status", ""))
            for peptide in state.get("peptides", [])
            for campaign in peptide.get("campaigns", [])
        }
        | {str(action.get("status", "")) for action in state.get("actions", [])}
    )
    status_options = ["All", *[value for value in status_values if value]]

    st.sidebar.markdown("### Filters")
    selected_run = "All"
    selected_sequence = "All"
    selected_profile_label = "All"
    selected_status = "All"
    if view == "Today":
        st.sidebar.caption("Focus the home queue by run, peptide, or current status.")
        selected_run = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter run",
            options=run_options,
            key="dashboard_filter_run",
            query_key="filter_run",
            default="All",
            write_query=False,
        )
        selected_sequence = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter peptide",
            options=sequence_options,
            key="dashboard_filter_sequence",
            query_key="filter_sequence",
            default="All",
            write_query=False,
        )
        selected_status = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter status",
            options=status_options,
            key="dashboard_filter_status",
            query_key="filter_status",
            default="All",
            write_query=False,
        )
    elif view == "Results":
        st.sidebar.caption("Compare visible thesis runs here. A run filter can narrow the reporting tables if you want to inspect one result in isolation.")
        selected_run = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter run",
            options=run_options,
            key="dashboard_filter_run",
            query_key="filter_run",
            default="All",
            write_query=False,
        )
    elif view == "Peptides":
        st.sidebar.caption("Narrow the peptide inventory without hiding the in-page lifecycle buckets.")
        selected_run = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter run",
            options=run_options,
            key="dashboard_filter_run",
            query_key="filter_run",
            default="All",
            write_query=False,
        )
        selected_sequence = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter peptide",
            options=sequence_options,
            key="dashboard_filter_sequence",
            query_key="filter_sequence",
            default="All",
            write_query=False,
        )
    elif view == "MD Validation":
        st.sidebar.caption("Use one run or stage filter here, then choose the exact peptide in the page body.")
        selected_run = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter run",
            options=run_options,
            key="dashboard_filter_run",
            query_key="filter_run",
            default="All",
            write_query=False,
        )
        selected_profile_label = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter MD stage",
            options=profile_options,
            key="dashboard_filter_profile",
            query_key="filter_profile",
            default="All",
            write_query=False,
        )
        selected_status = _persisted_choice(
            st,
            st.sidebar.selectbox,
            label="Filter status",
            options=status_options,
            key="dashboard_filter_status",
            query_key="filter_status",
            default="All",
            write_query=False,
        )
    elif view == "Operations":
        st.sidebar.caption("Operations is global. Run curation, approvals, jobs, and transfers are shown for the whole visible workspace.")
    else:
        st.sidebar.caption("This page already has its own selector. Use the in-page chooser instead of adding extra global filters here.")

    state = {
        **state,
        "dashboard_preferences": {
            **preferences,
            "ui_mode": selected_ui_mode,
            "workflow_mode": selected_workflow_mode,
            "approval_mode": selected_approval_mode,
            "refresh_mode": selected_refresh_mode,
        },
        "workflow_mode": selected_workflow_mode,
        "approval_mode": selected_approval_mode,
        "refresh_mode": selected_refresh_mode,
    }
    filtered_state = apply_dashboard_filters(
        state,
        workspace_scope=workspace_scope,
        run_name=selected_run,
        sequence=selected_sequence,
        md_profile=profile_filter_map.get(selected_profile_label, "All"),
        status=selected_status,
    )
    st.sidebar.caption(
        _active_filter_caption(
            run_name=selected_run,
            sequence=selected_sequence,
            md_profile=profile_filter_map.get(selected_profile_label, "All"),
            status=selected_status,
        )
    )
    overview = filtered_state["overview"]
    st.write("Workspace snapshot")
    _render_metric_cards(
        st,
        [
            ("Visible runs", overview["run_count"]),
            ("Visible peptide ladders", overview["peptide_count"]),
            ("Approvals", len(overview["approval_queue"])),
            ("Remote actions", len(overview["active_remote_actions"])),
        ],
    )
    st.caption(
        f"Workspace view: {workspace_scope}. Run, peptide, and navigation state persists across reloads, and button actions refresh in place without kicking you back to the top."
    )
    _render_autorefresh(st, selected_refresh)

    content_host = st.empty()
    with content_host.container():
        _render_page_guide(st, view=view, state=filtered_state)
        _render_guided_next_action_panel(st, filtered_state, view=view)
        _render_inline_approval_panel(st, filtered_state, key_prefix=f"guided_{view.lower().replace(' ', '_')}")
        if view == "Today":
            _render_overview(st, filtered_state)
        elif view == "Model Workflow":
            _render_run_detail(st, filtered_state)
        elif view == "Results":
            _render_results_view(st, filtered_state)
        elif view == "Peptides":
            _render_peptides_view(st, filtered_state)
        elif view == "MD Validation":
            _render_peptide_ladder(st, filtered_state)
        else:
            _render_operations(st, filtered_state)
    _flush_query_param_updates(st)


def build_streamlit_command(
    run_root: Path,
    *,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
    refresh_seconds: int = 0,
) -> list[str]:
    app_path = Path(__file__).with_name("dashboard_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--",
        "--run-root",
        str(run_root),
        "--refresh-seconds",
        str(refresh_seconds),
    ]
    return command



def launch_dashboard(
    run_root: Path,
    *,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
    refresh_seconds: int = 0,
    skip_integrity_check: bool = False,
) -> int:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "Streamlit is not installed. Add it to the environment and rerun the dashboard command."
        )
    if not skip_integrity_check:
        from active_learning_thesis.dashboard_integrity import (
            format_dashboard_integrity_report,
            run_dashboard_integrity_check,
        )

        report = run_dashboard_integrity_check(run_root, refresh_seconds=refresh_seconds)
        print(format_dashboard_integrity_report(report))
        if int(report.get("failure_count", 0)) > 0:
            print("Dashboard launch aborted because the integrity check found render failures. Re-run with '--skip-integrity-check' only if you intentionally want to bypass that guard.")
            return 1
    return subprocess.call(
        build_streamlit_command(
            run_root,
            host=host,
            port=port,
            refresh_seconds=refresh_seconds,
        )
    )






