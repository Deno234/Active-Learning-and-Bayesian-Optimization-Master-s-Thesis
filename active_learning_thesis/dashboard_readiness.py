from __future__ import annotations

from pathlib import Path

from active_learning_thesis.dashboard_action_contracts import (
    get_dashboard_action_contract,
    list_dashboard_action_contracts,
)
from active_learning_thesis.dashboard_md_slate import build_md_slate_launch_readiness
from active_learning_thesis.md_review_evidence import review_evidence_status


READINESS_PRIORITY = {
    "Blocked": 0,
    "Ready with caution": 1,
    "Ready": 2,
}


def _unique_messages(values: list[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text == "-" or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _review_evidence_issue_text(status: dict[str, object]) -> str:
    missing = status.get("missing", [])
    blockers = status.get("blockers", [])
    missing_items = missing if isinstance(missing, list) else []
    blocker_items = blockers if isinstance(blockers, list) else []
    issues = [
        str(item)
        for item in [*missing_items, *blocker_items]
        if str(item).strip()
    ]
    return ", ".join(issues) if issues else "review evidence"


def _run_display(run: dict[str, object]) -> str:
    return str(run.get("run_display_name", run.get("run_name", run.get("run_slug", "")))).strip()


def _cluster_health(state: dict[str, object], cluster: str) -> dict[str, str]:
    for row in list(state.get("cluster_health", [])):
        if str(row.get("cluster", "")).strip() == cluster:
            return {
                "status": str(row.get("overall_status", "unknown")).strip() or "unknown",
                "summary": str(row.get("summary", "")).strip(),
                "hint": str(row.get("hint", "")).strip(),
            }
    return {
        "status": "unknown",
        "summary": f"{cluster.upper()} health has not been checked yet.",
        "hint": f"Run the {cluster.upper()} health check before relying on remote actions.",
    }


def _profile_ready(state: dict[str, object], cluster: str) -> bool:
    for row in list(state.get("profile_rows", [])):
        if str(row.get("cluster", "")).strip() != cluster:
            continue
        return (
            str(row.get("configured", "")).strip().lower() == "yes"
            and str(row.get("enabled", "")).strip().lower() == "yes"
        )
    return False


def _artifact_attention_row(state: dict[str, object], campaign_dir: str) -> dict[str, object] | None:
    campaign_key = str(Path(campaign_dir).resolve()) if campaign_dir else ""
    if not campaign_key:
        return None
    for row in list(state.get("artifact_verification", [])):
        try:
            row_key = str(Path(str(row.get("campaign_dir", ""))).resolve())
        except Exception:
            row_key = str(row.get("campaign_dir", ""))
        if row_key == campaign_key:
            return row
    return None


def _is_manual_md_sandbox_campaign(campaign_dir: str) -> bool:
    if not campaign_dir:
        return False
    try:
        return (Path(campaign_dir) / "manual_md_sandbox.json").exists()
    except Exception:
        return False


def _recommended_run_contract(run: dict[str, object]) -> str:
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
    ml_status = str(run.get("ml_status", "")).strip()
    if bool(feedback_queue.get("can_continue", False)):
        return "continue-al-feedback"
    if list(run.get("available_ingest_csvs", [])):
        return "ingest-round"
    if ml_status == "replay-complete":
        return "propose-round"
    if ml_status == "discovery-complete":
        return "evaluate-final"
    return ""


def _ladder_macro_key(ladder: dict[str, object]) -> str:
    next_step = ladder.get("next_step", {}) if isinstance(ladder.get("next_step"), dict) else {}
    title = str(next_step.get("title", "")).strip()
    if title.startswith("Prepare "):
        return "prepare-md-stage"
    if title == "Upload the campaign to BURA":
        return "bura-upload-campaign"
    if title == "Normalize, preflight, then submit":
        return "bura-normalize-scripts"
    if title == "Monitor the active BURA campaign":
        return "bura-poll-squeue"
    if title in {"Re-parse the staged outputs locally", "Finalize the copied-back outputs"}:
        return "finalize-md-stage"
    if title == "Create the ingest CSV":
        return "make-md-ingest-csv"
    return ""


def _readiness_payload(
    contract_id: str,
    verdict: str,
    summary: str,
    *,
    run: str = "",
    sequence: str = "",
    target: str = "",
    blockers: list[str] | None = None,
    cautions: list[str] | None = None,
    fix_now: str = "",
    disable_button: bool | None = None,
) -> dict[str, object]:
    contract = get_dashboard_action_contract(contract_id)
    resolved_blockers = _unique_messages(list(blockers or []))
    resolved_cautions = _unique_messages(list(cautions or []))
    if disable_button is None:
        disable_button = verdict == "Blocked"
    return {
        "contract_id": contract_id,
        "label": contract.label if contract else contract_id,
        "view": contract.view if contract else "",
        "approval": contract.approval if contract else "",
        "scope": contract.scope if contract else "",
        "cluster": contract.cluster if contract else "",
        "verdict": verdict,
        "summary": str(summary).strip(),
        "run": str(run).strip(),
        "sequence": str(sequence).strip(),
        "target": str(target or sequence or run or (contract.label if contract else contract_id)).strip(),
        "blockers": resolved_blockers,
        "cautions": resolved_cautions,
        "fix_now": str(fix_now).strip(),
        "disable_button": bool(disable_button),
    }


def _export_batch_readiness(rows: list[dict[str, object]]) -> dict[str, object]:
    normalized_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip()
        if not sequence or not run_dir:
            continue
        key = (str(Path(run_dir).resolve()), sequence)
        if key in seen:
            continue
        seen.add(key)
        normalized_rows.append(row)
    if not normalized_rows:
        return _readiness_payload(
            "export-md-source-batch",
            "Blocked",
            "No peptide rows are available for dashboard-local MD batch export.",
            fix_now="Select one or more candidate rows first, then create the local batch rows.",
        )
    missing_rows = [
        row
        for row in normalized_rows
        if str(row.get("launch_ready", "")).strip() != "yes"
        or str(row.get("source_batch_csv", "")).strip() in {"", "-"}
    ]
    if not missing_rows:
        return _readiness_payload(
            "export-md-source-batch",
            "Ready with caution",
            "All selected peptides already have a source batch row, so this export is optional.",
            run=str(normalized_rows[0].get("run", "")),
            target=f"{len(normalized_rows)} peptide(s)",
            cautions=["The selected peptides already look launch-ready for MD."],
            fix_now="Skip this export and go straight to the MD slate launch or rehearsal path.",
            disable_button=False,
        )
    return _readiness_payload(
        "export-md-source-batch",
        "Ready",
        f"{len(missing_rows)} peptide(s) can be unlocked for MD by creating dashboard-local source batch rows.",
        run=str(missing_rows[0].get("run", "")),
        target=f"{len(missing_rows)} peptide(s)",
        fix_now="Run this local export, then re-check the MD slate launch readiness gate.",
        disable_button=False,
    )


def _md_slate_rehearsal_readiness(
    state: dict[str, object],
    *,
    run_dir: str,
    run_name: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        return _readiness_payload(
            "md-slate-rehearsal",
            "Blocked",
            "No peptides are selected for MD rehearsal yet.",
            run=run_name,
            fix_now="Mark one or more peptides as Selected for MD, or choose a launch-ready candidate first.",
        )
    blockers: list[str] = []
    launchable_rows: list[dict[str, object]] = []
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        source_batch_csv = str(row.get("source_batch_csv", "")).strip()
        if source_batch_csv in {"", "-"}:
            blockers.append(f"{sequence}: no source batch CSV currently contains this peptide.")
            continue
        batch_path = Path(source_batch_csv)
        if not batch_path.exists():
            blockers.append(f"{sequence}: saved source batch CSV no longer exists.")
            continue
        launchable_rows.append(row)
    if not launchable_rows:
        return _readiness_payload(
            "md-slate-rehearsal",
            "Blocked",
            "The local rehearsal is blocked because none of the selected peptides has a valid source batch CSV.",
            run=run_name,
            target=f"{len(rows)} selected peptide(s)",
            blockers=blockers,
            fix_now="Create dashboard-local MD batch rows for the blocked peptides first.",
        )
    verdict = "Ready"
    summary = f"The rehearsal can simulate {len(launchable_rows)} peptide(s) locally without touching BURA."
    cautions: list[str] = []
    if blockers:
        verdict = "Ready with caution"
        summary = (
            f"{len(launchable_rows)}/{len(rows)} selected peptide(s) can be rehearsed locally; "
            f"{len(blockers)} are still missing a source batch row."
        )
        cautions.extend(blockers[:3])
    return _readiness_payload(
        "md-slate-rehearsal",
        verdict,
        summary,
        run=run_name,
        target=f"{len(launchable_rows)} peptide(s)",
        cautions=cautions,
        fix_now=(
            "Run the rehearsal now, or create local batch rows first if you want every selected peptide included."
        ),
        disable_button=False,
    )


def _md_slate_run_readiness(
    state: dict[str, object],
    *,
    run_dir: str,
    run_name: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        return _readiness_payload(
            "md-slate-run",
            "Blocked",
            "No peptides are selected for this MD slate yet.",
            run=run_name,
            fix_now="Mark one or more peptides as Selected for MD first.",
        )
    readiness = build_md_slate_launch_readiness(
        run_root=Path(str(state.get("run_root", ""))),
        run_dir=Path(run_dir),
        run_name=run_name,
        peptides=[dict(row) for row in rows],
        profiles_payload=state.get("profiles", {}) if isinstance(state.get("profiles", {}), dict) else {},
        md_slates=list(state.get("md_slates", [])),
    )
    blocked_rows = [
        row
        for row in list(readiness.get("peptide_rows", []))
        if isinstance(row, dict) and str(row.get("launch_state", "")) != "Launch-ready"
    ]
    blockers = [
        str(row.get("blocker", ""))
        for row in blocked_rows
        if str(row.get("blocker", "")).strip() not in {"", "-"}
    ]
    cautions = [
        str(row.get("warning", ""))
        for row in list(readiness.get("peptide_rows", []))
        if isinstance(row, dict) and str(row.get("warning", "")).strip() not in {"", "-"}
    ]
    verdict_text = str(readiness.get("verdict", "")).strip()
    if verdict_text == "Ready to launch":
        verdict = "Ready"
    elif verdict_text == "Partially ready":
        verdict = "Ready with caution"
    else:
        verdict = "Blocked"
    fix_now = "Approve the slate once, then watch the shared slate board in MD Validation."
    if verdict == "Blocked":
        fix_now = "Create source batch rows for blocked peptides or clear the dry-run blockers before launching."
    elif int(readiness.get("queued_by_caps", 0) or 0) > 0:
        cautions.append("Some peptides will wait behind the current local line_smoke cap after approval.")
        fix_now = "Launch only if you are comfortable with later peptides waiting behind the local BURA cap."
    return _readiness_payload(
        "md-slate-run",
        verdict,
        str(readiness.get("summary", "")).strip() or "Check the dry-run launch gate before launching.",
        run=run_name,
        target=f"{int(readiness.get('launchable_count', 0) or 0)} launch-ready peptide(s)",
        blockers=blockers[:4],
        cautions=cautions[:4],
        fix_now=fix_now,
    )


def _run_action_readiness(state: dict[str, object], run: dict[str, object], contract_id: str) -> dict[str, object]:
    run_name = _run_display(run)
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
    pending_round_id = str(feedback_queue.get("pending_round_id", "")).strip()
    available_ingest_csvs = list(run.get("available_ingest_csvs", []))
    ml_status = str(run.get("ml_status", "")).strip()
    remote_status = str(run.get("remote_sync_status", "not_synced")).strip()
    remote_job_id = str(run.get("remote_job_id", "")).strip()
    discovery_ready = ml_status in {"replay-complete", "batch-proposed"} or bool(run.get("import_rows"))
    recommended_contract = _recommended_run_contract(run)

    if contract_id == "continue-al-feedback":
        if bool(feedback_queue.get("can_continue", False)):
            return _readiness_payload(
                contract_id,
                "Ready",
                str(feedback_queue.get("summary", "")).strip() or "The pending proposed batch is fully reviewed and can re-enter the model now.",
                run=run_name,
                target=f"Round {pending_round_id or '-'}",
                fix_now="Run the full feedback handoff now to build the import CSV, ingest it, and refresh the model state.",
            )
        if pending_round_id:
            return _readiness_payload(
                contract_id,
                "Blocked",
                str(feedback_queue.get("summary", "")).strip() or "The pending proposed batch still needs more MD review work.",
                run=run_name,
                target=f"Round {pending_round_id}",
                blockers=[
                    str(row.get("blocker", ""))
                    for row in list(feedback_queue.get("blocked_rows", []))
                    if isinstance(row, dict)
                ][:4],
                fix_now="Open the Model Workflow feedback queue and finish the missing review, promotion, or full-analysis work first.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "No pending reviewed AL batch is visible for this run.",
            run=run_name,
            fix_now="Create or ingest reviewed MD results first before continuing the full feedback loop.",
        )

    if contract_id == "ingest-round":
        if available_ingest_csvs:
            cautions: list[str] = []
            if bool(feedback_queue.get("can_continue", False)):
                cautions.append("A full-batch Continue AL handoff is also available for this run.")
            return _readiness_payload(
                contract_id,
                "Ready" if not cautions else "Ready with caution",
                "A reviewed cgmd_ingest.csv is available for this run.",
                run=run_name,
                target=Path(str(available_ingest_csvs[0])).name,
                cautions=cautions,
                fix_now=(
                    "Use Ingest returned labels for the selected CSV, or switch to Continue AL if you want the full pending batch handled together."
                ),
            )
        if pending_round_id:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The run still has a pending proposed batch, but no reviewed cgmd_ingest.csv is visible yet.",
                run=run_name,
                target=f"Round {pending_round_id}",
                fix_now="Create cgmd_ingest.csv from MD Validation after the human review label is saved.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "No reviewed cgmd_ingest.csv is available for this run yet.",
            run=run_name,
            fix_now="Review a completed full-analysis peptide first, then build the ingest CSV.",
        )

    if contract_id == "propose-round":
        if recommended_contract == "propose-round":
            return _readiness_payload(
                contract_id,
                "Ready",
                "The run has a trained replay state and is ready to propose the next peptide batch.",
                run=run_name,
                fix_now="Run the next-batch proposal now, then review the resulting candidates in Peptides.",
            )
        if pending_round_id:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The current proposed batch still needs feedback before another batch should be proposed.",
                run=run_name,
                target=f"Round {pending_round_id}",
                fix_now="Finish the current feedback loop first, then propose again.",
            )
        if ml_status in {"discovery-complete", "final-evaluated"}:
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "The run already reached a later thesis stage, so a fresh proposal would be a branch choice rather than the normal next step.",
                run=run_name,
                fix_now="Only propose another batch if you intentionally want to keep exploring beyond the current reporting state.",
                disable_button=False,
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The run is not yet at a trained post-replay checkpoint that should propose a new batch.",
            run=run_name,
            fix_now="Finish replay benchmarking or ingest reviewed labels first.",
        )

    if contract_id == "run-discovery":
        if ml_status == "discovery-complete":
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "Discovery artifacts already exist for this run, so rerunning discovery is optional rather than required.",
                run=run_name,
                fix_now="Use Results if you only need the existing discovery outputs, or rerun discovery intentionally.",
                disable_button=False,
            )
        if discovery_ready:
            return _readiness_payload(
                contract_id,
                "Ready",
                "The run is mature enough to generate discovery evidence now.",
                run=run_name,
                fix_now="Run discovery if you want exploratory thesis evidence alongside the main AL loop.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The run is not mature enough for discovery yet.",
            run=run_name,
            fix_now="Build the replay/proposal state or ingest additional labels before running discovery.",
        )

    if contract_id == "evaluate-final":
        if ml_status == "final-evaluated":
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "Final holdout metrics already exist for this run, so another frozen evaluation is optional.",
                run=run_name,
                fix_now="Use Results if you only need the current reporting artifacts, or rerun intentionally.",
                disable_button=False,
            )
        if recommended_contract == "evaluate-final":
            return _readiness_payload(
                contract_id,
                "Ready",
                "Discovery outputs already exist, so the run can freeze a thesis-ready final evaluation now.",
                run=run_name,
                fix_now="Run the frozen holdout evaluation now if this run is moving into reporting mode.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The run is not at the discovery-complete checkpoint required for a frozen final evaluation yet.",
            run=run_name,
            fix_now="Finish discovery first, then come back to final evaluation.",
        )

    if contract_id == "freeze-final":
        run_dir_text = str(run.get("run_dir", "")).strip()
        run_dir = Path(run_dir_text) if run_dir_text else None
        freeze_json = run_dir / "final_freeze" / "final_freeze.json" if run_dir is not None else None
        if freeze_json is not None and freeze_json.exists():
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "A final freeze already exists for this run.",
                run=run_name,
                target=freeze_json.name,
                cautions=["Leave Force unchecked unless you intentionally want to replace the freeze bundle."],
                fix_now="Use Results -> Thesis output builder if the existing freeze is the one you want to report.",
                disable_button=False,
            )
        if run.get("final_metrics"):
            return _readiness_payload(
                contract_id,
                "Ready",
                "Final holdout metrics exist, so this run can be frozen for thesis reporting.",
                run=run_name,
                fix_now="Freeze now to write the reproducibility checks and model card.",
                disable_button=False,
            )
        if ml_status in {"discovery-complete", "final-evaluated"}:
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "Final metrics are missing, but the freeze action can run final evaluation first if that option is enabled.",
                run=run_name,
                cautions=["Keep 'Run final evaluation first if needed' enabled from the GUI."],
                fix_now="Run the freeze with final evaluation enabled, or run Final evaluation first and return here.",
                disable_button=False,
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The run is not ready for a final thesis freeze yet.",
            run=run_name,
            fix_now="Finish replay/proposal/MD feedback or discovery before freezing this run.",
        )

    if contract_id in {"supek-sync-run", "supek-submit-workflow", "supek-poll-qstat"}:
        health = _cluster_health(state, "supek")
        health_status = str(health.get("status", "unknown")).strip()
        profile_ready = _profile_ready(state, "supek")
        remote_issue = " ".join(part for part in [health.get("summary", ""), health.get("hint", "")] if str(part).strip())
        if not profile_ready:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The SUPEK profile is not configured or enabled.",
                run=run_name,
                blockers=["SUPEK profile/config is missing or disabled."],
                fix_now="Configure the SUPEK profile in Operations -> Cluster health first.",
            )
        if health_status not in {"ok", "unknown"}:
            return _readiness_payload(
                contract_id,
                "Blocked",
                remote_issue or "SUPEK health is degraded, so remote actions should not be launched right now.",
                run=run_name,
                fix_now="Re-run the SUPEK health check and clear the connectivity/auth blocker first.",
            )
        caution_messages = [remote_issue] if health_status == "unknown" and remote_issue else []

        if contract_id == "supek-sync-run":
            if remote_status == "not_synced":
                return _readiness_payload(
                    contract_id,
                    "Ready" if not caution_messages else "Ready with caution",
                    "The run can be staged on SUPEK.",
                    run=run_name,
                    cautions=caution_messages,
                    fix_now="Stage the run on SUPEK now if the next workflow step needs remote execution.",
                )
            return _readiness_payload(
                contract_id,
                "Blocked",
                "This run is already staged or active on SUPEK, so another staging draft is not the next safe step.",
                run=run_name,
                target=remote_status or "remote state",
                fix_now="Use the SUPEK submit or monitor step that matches the current remote state instead.",
            )

        if contract_id == "supek-submit-workflow":
            if remote_status == "staged_remote" and _recommended_run_contract(run):
                return _readiness_payload(
                    contract_id,
                    "Ready" if not caution_messages else "Ready with caution",
                    "The run is staged on SUPEK and the next remote workflow is known.",
                    run=run_name,
                    cautions=caution_messages,
                    fix_now="Create the draft now, then approve it once you are satisfied with the next workflow step.",
                )
            if remote_status == "staged_remote":
                return _readiness_payload(
                    contract_id,
                    "Blocked",
                    "The run is staged remotely, but the dashboard cannot infer a safe next SUPEK workflow to submit.",
                    run=run_name,
                    fix_now="Resolve the local run state first so the cockpit knows which workflow should run next.",
                )
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The run must be staged on SUPEK before a remote workflow can be submitted.",
                run=run_name,
                fix_now="Stage the run on SUPEK first.",
            )

        if remote_status in {"submitted", "running"} and remote_job_id:
            return _readiness_payload(
                contract_id,
                "Ready" if not caution_messages else "Ready with caution",
                "A tracked SUPEK job is active for this run, so queue polling is safe.",
                run=run_name,
                target=remote_job_id,
                cautions=caution_messages,
                fix_now="Poll the queue now, then fetch logs or pull artifacts when the job leaves the queue.",
                disable_button=False,
            )
        if remote_status in {"submitted", "running"}:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The run looks remotely active, but there is no tracked SUPEK job id attached yet.",
                run=run_name,
                fix_now="Fetch logs or reconcile the remote state before relying on queue polling.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "No active SUPEK job is currently tracked for this run.",
            run=run_name,
            fix_now="Stage and submit the run first, or use Results/Model Workflow if the run already finished remotely.",
        )

    return _readiness_payload(
        contract_id,
        "Blocked",
        "No readiness rule is available for this action yet.",
        run=run_name,
    )


def _ladder_action_readiness(state: dict[str, object], ladder: dict[str, object], contract_id: str) -> dict[str, object]:
    sequence = str(ladder.get("sequence", "")).strip()
    run_name = str(ladder.get("run_display_name", "") or ladder.get("run_name", "")).strip()
    macro_contract = _ladder_macro_key(ladder)
    full_item = ladder.get("full") or {}
    current = ladder.get("current") or {}
    ingest_supported = bool(ladder.get("ingest_supported", True))
    promotion_available = bool(ladder.get("promotion_available", False))
    label_value = str(full_item.get("cgmd_label", "")).strip()
    campaign_dir = str((current or full_item).get("campaign_dir", "")).strip()
    artifact_row = _artifact_attention_row(state, campaign_dir) if campaign_dir else None

    if contract_id == "prepare-md-stage":
        source_batch_csv = str(ladder.get("source_batch_csv", "")).strip()
        if macro_contract == contract_id and source_batch_csv and Path(source_batch_csv).exists():
            return _readiness_payload(
                contract_id,
                "Ready",
                "The next MD stage can be prepared locally for this peptide.",
                run=run_name,
                sequence=sequence,
                fix_now="Prepare the local campaign now, then move to the BURA upload step.",
            )
        blocker = "No source batch CSV currently contains this peptide."
        if source_batch_csv and not Path(source_batch_csv).exists():
            blocker = f"Saved source batch CSV no longer exists: {source_batch_csv}"
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The next MD stage cannot be prepared yet.",
            run=run_name,
            sequence=sequence,
            blockers=[blocker],
            fix_now="Create or restore the source batch row first, then prepare the stage.",
        )

    if contract_id in {"bura-upload-campaign", "bura-normalize-scripts", "bura-poll-squeue"}:
        health = _cluster_health(state, "bura")
        health_status = str(health.get("status", "unknown")).strip()
        profile_ready = _profile_ready(state, "bura")
        remote_issue = " ".join(part for part in [health.get("summary", ""), health.get("hint", "")] if str(part).strip())
        if not profile_ready:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The BURA profile is not configured or enabled.",
                run=run_name,
                sequence=sequence,
                fix_now="Configure the BURA profile in Operations -> Cluster health first.",
            )
        if health_status not in {"ok", "unknown"}:
            return _readiness_payload(
                contract_id,
                "Blocked",
                remote_issue or "BURA health is degraded, so remote MD actions should wait.",
                run=run_name,
                sequence=sequence,
                fix_now="Re-run the BURA health check and clear the connectivity/scheduler blocker first.",
            )
        cautions = [remote_issue] if health_status == "unknown" and remote_issue else []

        if contract_id == "bura-upload-campaign":
            manual_sandbox = _is_manual_md_sandbox_campaign(campaign_dir)
            campaign_sync_status = str((current or full_item).get("sync_status", "")).strip() or str(ladder.get("sync_status", "")).strip()
            if campaign_dir and (macro_contract == contract_id or (manual_sandbox and campaign_sync_status in {"", "not_synced"})):
                sandbox_cautions = ["This is a manual MD sandbox campaign, not an AL-selected peptide. Keep it out of ingest unless you intentionally promote it."] if manual_sandbox else []
                return _readiness_payload(
                    contract_id,
                    "Ready" if not cautions and not sandbox_cautions else "Ready with caution",
                    "The prepared campaign can be uploaded to BURA now.",
                    run=run_name,
                    sequence=sequence,
                    cautions=[*cautions, *sandbox_cautions],
                    fix_now="Create the upload draft now if you want this peptide to move remotely.",
                )
            return _readiness_payload(
                contract_id,
                "Blocked",
                "Upload is not the next safe ladder step for this peptide yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Prepare the local campaign first, or continue from the current remote step instead.",
            )

        if contract_id == "bura-normalize-scripts":
            campaign_sync_status = str((current or full_item).get("sync_status", "")).strip() or str(ladder.get("sync_status", "")).strip()
            if campaign_dir and (macro_contract == contract_id or campaign_sync_status == "staged_remote"):
                return _readiness_payload(
                    contract_id,
                    "Ready" if not cautions else "Ready with caution",
                    "The staged campaign can start the BURA normalize -> preflight -> submit chain.",
                    run=run_name,
                    sequence=sequence,
                    cautions=cautions,
                    fix_now="Create the normalization draft now, then follow with readiness/preflight/submit.",
                )
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The campaign is not at the staged-remote checkpoint needed for normalization/preflight yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Upload the campaign to BURA first.",
            )

        if macro_contract == contract_id or str(ladder.get("sync_status", "")).strip() in {"submitted", "running"}:
            if not str((current or {}).get("remote_job_id", "")).strip():
                cautions.append("The ladder looks remotely active, but there is no tracked BURA job id attached yet.")
            return _readiness_payload(
                contract_id,
                "Ready" if not cautions else "Ready with caution",
                "The active BURA campaign can be polled safely.",
                run=run_name,
                sequence=sequence,
                cautions=cautions,
                fix_now="Poll the queue now, then fetch logs or pull/finalize when the job finishes.",
                disable_button=False,
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "No active BURA campaign is currently tracked for this peptide.",
            run=run_name,
            sequence=sequence,
            fix_now="Upload and submit the current stage first, or use Recovery if the tracked job drifted.",
        )

    if contract_id == "finalize-md-stage":
        if artifact_row and str(artifact_row.get("verification_state", "")).strip() == "Attention needed":
            return _readiness_payload(
                contract_id,
                "Blocked",
                str(artifact_row.get("summary", "")).strip() or "Artifact verification still shows attention-needed problems.",
                run=run_name,
                sequence=sequence,
                blockers=[str(artifact_row.get("retry_action", "")).strip()],
                fix_now=str(artifact_row.get("retry_action", "")).strip() or "Repair the copied-back artifacts first, then finalize.",
            )
        if macro_contract == contract_id or str(ladder.get("sync_status", "")).strip() in {"outputs_staged", "outputs_returned", "finalized_local"}:
            return _readiness_payload(
                contract_id,
                "Ready",
                "Copied-back or staged outputs are present, so local finalization can run now.",
                run=run_name,
                sequence=sequence,
                fix_now="Finalize now to refresh the ladder state and the next recommendation.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "No copied-back outputs are staged for local finalization yet.",
            run=run_name,
            sequence=sequence,
            fix_now="Wait for the remote outputs to return, then finalize.",
        )

    if contract_id == "update-md-review":
        if not full_item:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "No returned full-analysis row is visible for this peptide yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Finish the full stage and copy the outputs back before saving a review row.",
            )
        if str(full_item.get("job_root_status", "")).strip() != "analysis_complete":
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "A review row exists, but the full-analysis stage is not marked analysis_complete yet.",
                run=run_name,
                sequence=sequence,
                cautions=["Save notes if helpful, but wait for analysis_complete before treating the label as final."],
                fix_now="Inspect the returned outputs first, then save the final human label once the analysis is complete.",
                disable_button=False,
            )
        return _readiness_payload(
            contract_id,
            "Ready",
            "The returned full-analysis evidence is ready for a structured human review and label entry.",
            run=run_name,
            sequence=sequence,
            fix_now="Save the cgmd_label, rubric, confidence, evidence summary, and notes once you are confident in the manual call.",
            disable_button=False,
        )

    if contract_id == "promote-reporting-md-campaign":
        if not full_item:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "No full-analysis campaign is visible to promote yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Finish the full analysis first.",
            )
        if ingest_supported:
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "This campaign already points at an ingest-supported batch, so promotion is not needed.",
                run=run_name,
                sequence=sequence,
                fix_now="Skip promotion and go straight to Create ingest CSV if the review label is final.",
                disable_button=False,
            )
        if promotion_available:
            return _readiness_payload(
                contract_id,
                "Ready",
                "A real proposed batch now contains this peptide, so the reporting-only campaign can be promoted.",
                run=run_name,
                sequence=sequence,
                fix_now="Promote the campaign now, then create cgmd_ingest.csv against the real batch.",
            )
        return _readiness_payload(
            contract_id,
            "Blocked",
            "The campaign is still reporting-only and no real proposed batch currently contains this peptide.",
            run=run_name,
            sequence=sequence,
            fix_now="Wait until a real proposed batch contains this peptide, then come back to promotion.",
        )

    if contract_id == "make-md-ingest-csv":
        if not full_item:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "No full-analysis campaign exists for this peptide yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Finish the full stage and save the review row first.",
            )
        if str(full_item.get("job_root_status", "")).strip() != "analysis_complete":
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The full-analysis outputs are not complete yet.",
                run=run_name,
                sequence=sequence,
                fix_now="Wait for analysis_complete, then review the evidence and create the ingest CSV.",
            )
        ingest_path = Path(str(full_item.get("campaign_dir", ""))) / "cgmd_ingest.csv"
        if ingest_path.exists():
            return _readiness_payload(
                contract_id,
                "Ready with caution",
                "cgmd_ingest.csv already exists for this campaign, so this action is not needed again.",
                run=run_name,
                sequence=sequence,
                target=ingest_path.name,
                fix_now="Use Model Workflow -> Ingest returned labels for the parent run instead.",
                disable_button=False,
            )
        if label_value not in {"0", "1"}:
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The human cgmd_label is still missing for this peptide.",
                run=run_name,
                sequence=sequence,
                fix_now="Save the review decision first, then create the ingest CSV.",
            )
        if not ingest_supported:
            fix_now = (
                "Promote the reporting-only campaign into a real proposed batch first."
                if promotion_available
                else "This campaign remains reporting-only until the peptide appears in a real proposed batch."
            )
            return _readiness_payload(
                contract_id,
                "Blocked",
                "This campaign is still reporting-only, so it should not create cgmd_ingest.csv yet.",
                run=run_name,
                sequence=sequence,
                fix_now=fix_now,
            )
        review_status = review_evidence_status(full_item)
        if not bool(review_status.get("ingest_ready", False)):
            return _readiness_payload(
                contract_id,
                "Blocked",
                "The review label is not evidence-backed enough for model feedback yet.",
                run=run_name,
                sequence=sequence,
                blockers=[_review_evidence_issue_text(review_status)],
                fix_now="Complete the review rubric, confidence, evidence summary, and notes first.",
            )
        return _readiness_payload(
            contract_id,
            "Ready",
            "The evidence-backed full-analysis result can now become cgmd_ingest.csv.",
            run=run_name,
            sequence=sequence,
            fix_now="Create the ingest CSV now, then switch to Model Workflow for model ingest.",
        )

    return _readiness_payload(
        contract_id,
        "Blocked",
        "No readiness rule is available for this peptide action yet.",
        run=run_name,
        sequence=sequence,
    )


def build_button_readiness(
    state: dict[str, object],
    contract_id: str,
    *,
    run: dict[str, object] | None = None,
    ladder: dict[str, object] | None = None,
    rows: list[dict[str, object]] | None = None,
    run_dir: str = "",
    run_name: str = "",
) -> dict[str, object]:
    contract_id = str(contract_id or "").strip()
    if not contract_id:
        return {}
    if get_dashboard_action_contract(contract_id) is None:
        return {}
    if contract_id == "export-md-source-batch":
        return _export_batch_readiness(list(rows or []))
    if contract_id == "md-slate-rehearsal":
        return _md_slate_rehearsal_readiness(
            state,
            run_dir=run_dir or str((rows or [{}])[0].get("run_dir", "")),
            run_name=run_name or str((rows or [{}])[0].get("run", "")),
            rows=list(rows or []),
        )
    if contract_id == "md-slate-run":
        return _md_slate_run_readiness(
            state,
            run_dir=run_dir or str((rows or [{}])[0].get("run_dir", "")),
            run_name=run_name or str((rows or [{}])[0].get("run", "")),
            rows=list(rows or []),
        )
    if run is not None:
        return _run_action_readiness(state, run, contract_id)
    if ladder is not None:
        return _ladder_action_readiness(state, ladder, contract_id)
    return _readiness_payload(
        contract_id,
        "Blocked",
        "The readiness context for this action is not available in the current view.",
    )


def _add_row(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    verdict = str(row.get("verdict", "")).strip()
    if verdict not in READINESS_PRIORITY:
        return
    rows.append(row)


def _contract_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    contract_map = {contract.contract_id: contract for contract in list_dashboard_action_contracts()}
    aggregated: dict[str, dict[str, object]] = {}
    for row in rows:
        contract_id = str(row.get("contract_id", "")).strip()
        if not contract_id:
            continue
        current = aggregated.get(contract_id)
        verdict = str(row.get("verdict", "")).strip()
        verdict_rank = READINESS_PRIORITY.get(verdict, 99)
        if current is None:
            aggregated[contract_id] = {
                "contract_id": contract_id,
                "verdict": verdict,
                "summary": str(row.get("summary", "")).strip(),
                "sample_target": str(row.get("target", "")).strip(),
                "ready_count": 1 if verdict == "Ready" else 0,
                "caution_count": 1 if verdict == "Ready with caution" else 0,
                "blocked_count": 1 if verdict == "Blocked" else 0,
                "_rank": verdict_rank,
            }
            continue
        current["ready_count"] = int(current.get("ready_count", 0)) + (1 if verdict == "Ready" else 0)
        current["caution_count"] = int(current.get("caution_count", 0)) + (1 if verdict == "Ready with caution" else 0)
        current["blocked_count"] = int(current.get("blocked_count", 0)) + (1 if verdict == "Blocked" else 0)
        if verdict_rank < int(current.get("_rank", 99)):
            current["verdict"] = verdict
            current["summary"] = str(row.get("summary", "")).strip()
            current["sample_target"] = str(row.get("target", "")).strip()
            current["_rank"] = verdict_rank
    summary_rows: list[dict[str, object]] = []
    for contract_id, row in aggregated.items():
        contract = contract_map.get(contract_id)
        summary_rows.append(
            {
                "contract_id": contract_id,
                "label": contract.label if contract else contract_id,
                "view": contract.view if contract else "",
                "verdict": row["verdict"],
                "summary": row["summary"],
                "sample_target": row["sample_target"],
                "ready_count": row["ready_count"],
                "caution_count": row["caution_count"],
                "blocked_count": row["blocked_count"],
            }
        )
    summary_rows.sort(
        key=lambda row: (
            READINESS_PRIORITY.get(str(row.get("verdict", "")), 99),
            str(row.get("label", "")),
        )
    )
    return summary_rows


def build_execution_readiness(state: dict[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    candidate_rows = list(inventory.get("candidate_selection", [])) if isinstance(inventory.get("candidate_selection", []), list) else []
    selected_candidates_by_run: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        if str(row.get("candidate_status", "")).strip() != "Selected for MD":
            continue
        run_label = str(row.get("run", "")).strip()
        if not run_label:
            continue
        selected_candidates_by_run.setdefault(run_label, []).append(row)
    for run_label, selected_rows in selected_candidates_by_run.items():
        _add_row(
            rows,
            _md_slate_run_readiness(
                state,
                run_dir=str(selected_rows[0].get("run_dir", "")),
                run_name=run_label,
                rows=selected_rows,
            ),
        )
        blocked_rows = [
            row
            for row in selected_rows
            if str(row.get("launch_ready", "")).strip() != "yes"
        ]
        if blocked_rows:
            _add_row(rows, _export_batch_readiness(blocked_rows))

    for run in list(state.get("runs", [])):
        recommended_contract = _recommended_run_contract(run)
        if recommended_contract:
            _add_row(rows, _run_action_readiness(state, run, recommended_contract))
        run_dir_text = str(run.get("run_dir", "")).strip()
        run_dir = Path(run_dir_text) if run_dir_text else None
        if bool(run.get("final_metrics")) and (run_dir is None or not (run_dir / "final_freeze" / "final_freeze.json").exists()):
            _add_row(rows, _run_action_readiness(state, run, "freeze-final"))
        if str(run.get("ml_status", "")).strip() != "discovery-complete":
            discovery_row = _run_action_readiness(state, run, "run-discovery")
            if discovery_row["verdict"] != "Blocked":
                _add_row(rows, discovery_row)
        remote_status = str(run.get("remote_sync_status", "not_synced")).strip()
        if remote_status == "not_synced":
            _add_row(rows, _run_action_readiness(state, run, "supek-sync-run"))
        elif remote_status == "staged_remote":
            _add_row(rows, _run_action_readiness(state, run, "supek-submit-workflow"))
        elif remote_status in {"submitted", "running"}:
            _add_row(rows, _run_action_readiness(state, run, "supek-poll-qstat"))

    for ladder in list(state.get("peptides", [])):
        macro_contract = _ladder_macro_key(ladder)
        if macro_contract:
            _add_row(rows, _ladder_action_readiness(state, ladder, macro_contract))
            continue
        full_item = ladder.get("full") or {}
        if full_item:
            if str(full_item.get("cgmd_label", "")).strip() not in {"0", "1"}:
                _add_row(rows, _ladder_action_readiness(state, ladder, "update-md-review"))
            elif not bool(ladder.get("ingest_supported", True)):
                _add_row(rows, _ladder_action_readiness(state, ladder, "promote-reporting-md-campaign"))
            else:
                ingest_row = _ladder_action_readiness(state, ladder, "make-md-ingest-csv")
                if ingest_row["verdict"] != "Ready with caution" or "already exists" not in str(ingest_row.get("summary", "")).lower():
                    _add_row(rows, ingest_row)

    rows.sort(
        key=lambda row: (
            READINESS_PRIORITY.get(str(row.get("verdict", "")), 99),
            str(row.get("view", "")),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
            str(row.get("label", "")),
        )
    )
    counts = {
        "blocked": sum(1 for row in rows if str(row.get("verdict", "")) == "Blocked"),
        "caution": sum(1 for row in rows if str(row.get("verdict", "")) == "Ready with caution"),
        "ready": sum(1 for row in rows if str(row.get("verdict", "")) == "Ready"),
        "total": len(rows),
    }
    return {
        "rows": rows,
        "counts": counts,
        "contract_rows": _contract_summary_rows(rows),
        "top_rows": rows[:8],
    }


def with_execution_readiness(state: dict[str, object]) -> dict[str, object]:
    enriched = dict(state)
    enriched["execution_readiness"] = build_execution_readiness(enriched)
    return enriched
