from __future__ import annotations

import csv
from pathlib import Path
from types import ModuleType


MD_SECTIONS = [
    "Manual MD sandbox",
    "Review & ingest",
    "BURA performance",
    "Slate monitor",
    "Recovery center",
    "Artifact verification",
    "Ladder overview",
    "Local MD actions",
    "Remote BURA",
    "Decision log",
    "Recent actions",
    "Command desk",
]
GUIDED_MD_SECTIONS = [
    "Manual MD sandbox",
    "Review & ingest",
    "Ladder overview",
    "Local MD actions",
    "Remote BURA",
    "Advanced / debug tools",
]
ADVANCED_MD_SECTIONS = [
    section
    for section in MD_SECTIONS
    if section not in GUIDED_MD_SECTIONS and section != "Advanced / debug tools"
]

BURA_FULL_RUNNER_ACTION_KINDS = {
    "bura-upload-campaign",
    "bura-normalize-scripts",
    "bura-preflight",
    "bura-submit-chain",
    "bura-full-autopilot",
    "bura-md-workflow",
    "bura-poll-squeue",
    "bura-pull-package",
    "finalize-md-stage",
}
BURA_FULL_RUNNER_BUSY_STATUSES = {"awaiting_approval", "queued", "running"}


def _same_dashboard_path(left: str, right: str) -> bool:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return False
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except OSError:
        normalized_left = left.replace("\\", "/").rstrip("/").lower()
        normalized_right = right.replace("\\", "/").rstrip("/").lower()
        return normalized_left == normalized_right


def _campaign_action_matches(action: dict[str, object], current: dict[str, object], sequence: str) -> bool:
    campaign_dir = str(current.get("campaign_dir", "")).strip()
    metadata = action.get("metadata", {})
    metadata_target = str(metadata.get("target_key", "")).strip() if isinstance(metadata, dict) else ""
    action_campaigns = [
        str(action.get("related_campaign", "")).strip(),
        metadata_target,
    ]
    if campaign_dir and any(_same_dashboard_path(action_campaign, campaign_dir) for action_campaign in action_campaigns):
        return True
    return str(action.get("related_sequence", "")).strip().upper() == sequence.strip().upper()


def _campaign_actions(
    actions: list[dict[str, object]],
    current: dict[str, object],
    sequence: str,
    *,
    kinds: set[str] | None = None,
) -> list[dict[str, object]]:
    wanted = kinds or BURA_FULL_RUNNER_ACTION_KINDS
    return [
        action
        for action in actions
        if str(action.get("kind", "")) in wanted and _campaign_action_matches(action, current, sequence)
    ]


def _latest_action(actions: list[dict[str, object]], *, kind: str | None = None) -> dict[str, object] | None:
    for action in actions:
        if kind is None or str(action.get("kind", "")) == kind:
            return action
    return None


def _latest_successful_pull_stage_path(actions: list[dict[str, object]]) -> str:
    for action in actions:
        if str(action.get("kind", "")) != "bura-pull-package" or str(action.get("status", "")) != "succeeded":
            continue
        metadata = action.get("metadata", {})
        candidates = [
            str(action.get("local_stage_path", "") or ""),
            str(metadata.get("local_stage_path", "") or "") if isinstance(metadata, dict) else "",
            str(action.get("output_path", "") or ""),
        ]
        for candidate in candidates:
            if candidate.strip():
                return candidate.strip()
    return ""


def _has_success(actions: list[dict[str, object]], kind: str) -> bool:
    return any(str(action.get("kind", "")) == kind and str(action.get("status", "")) == "succeeded" for action in actions)


def _has_busy_runner_action(actions: list[dict[str, object]]) -> dict[str, object] | None:
    for action in actions:
        if str(action.get("status", "")) in BURA_FULL_RUNNER_BUSY_STATUSES:
            return action
    return None


def _tail_action_log(action: dict[str, object] | None, field: str, *, max_lines: int = 8) -> str:
    if not action:
        return ""
    path_text = str(action.get(field, "")).strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:]).strip()


def _latest_poll_missing_tracked_job(actions: list[dict[str, object]], remote_job_id: str) -> bool:
    if not remote_job_id:
        return False
    poll = _latest_action(actions, kind="bura-poll-squeue")
    if not poll or str(poll.get("status", "")) != "succeeded":
        return False
    stdout = _tail_action_log(poll, "stdout_log", max_lines=80)
    return bool(stdout) and remote_job_id not in stdout


def _bura_full_runner_decision(
    *,
    current: dict[str, object],
    sequence: str,
    actions: list[dict[str, object]],
) -> dict[str, object]:
    profile = str(current.get("md_profile", "")).strip()
    sync_status = str(current.get("sync_status", "not_synced") or "not_synced")
    remote_job_id = str(current.get("remote_job_id", "") or "")
    job_root_status = str(current.get("job_root_status", "") or "")

    if not current.get("campaign_dir"):
        return {"step": "blocked", "label": "No selected campaign", "message": "Choose or prepare a campaign first."}
    if profile != "full":
        return {
            "step": "blocked",
            "label": "Full campaign required",
            "message": "This is line_smoke/production_smoke; select or prepare a full campaign first.",
        }

    runner_actions = _campaign_actions(actions, current, sequence)
    staged_package_path = str(current.get("local_stage_path", "") or _latest_successful_pull_stage_path(runner_actions))
    busy = _has_busy_runner_action(runner_actions)
    if busy:
        return {
            "step": "waiting",
            "label": "Waiting for active action",
            "message": f"{busy.get('title', busy.get('kind', 'Action'))} is {busy.get('status', 'active')}.",
            "action": busy,
        }

    if sync_status == "finalized_local" or job_root_status in {"analysis_complete", "review_ready"}:
        return {
            "step": "done",
            "label": "Finalized locally",
            "message": "The copied-back MD outputs have been parsed. Open Review & ingest to inspect the evidence.",
        }
    if sync_status == "not_synced":
        return {"step": "upload", "label": "Upload campaign", "message": "Next safe step: upload this full campaign to BURA."}
    if sync_status == "staged_remote":
        if not _has_success(runner_actions, "bura-normalize-scripts"):
            return {"step": "normalize", "label": "Normalize scripts", "message": "Next safe step: normalize uploaded BURA scripts."}
        if not _has_success(runner_actions, "bura-preflight"):
            return {"step": "preflight", "label": "Run preflight", "message": "Next safe step: run the BURA preflight check."}
        if not _has_success(runner_actions, "bura-submit-chain"):
            return {"step": "submit", "label": "Submit chain", "message": "Next safe step: submit the BURA MD chain."}
        return {"step": "poll", "label": "Poll queue", "message": "Submission exists; poll BURA to refresh the tracked state."}
    if sync_status in {"outputs_staged", "outputs_returned"} or staged_package_path:
        return {"step": "finalize", "label": "Parse returned outputs", "message": "Copied-back outputs are available; parse/finalize locally."}
    if sync_status in {"submitted", "running"}:
        if _latest_poll_missing_tracked_job(runner_actions, remote_job_id):
            return {"step": "pull", "label": "Copy outputs back", "message": "The tracked job no longer appears in queue output; copy outputs back."}
        return {"step": "poll", "label": "Poll queue", "message": "The chain is submitted/running; poll BURA for status."}
    return {
        "step": "blocked",
        "label": "Unknown remote state",
        "message": f"Remote state '{sync_status}' is not safe to advance automatically. Use Advanced / individual BURA actions.",
    }


def render_md_validation_view(st, state: dict[str, object], *, ns: ModuleType) -> None:
    MD_PROFILE_INFO = ns.MD_PROFILE_INFO
    _artifact_rows_for_ladder = ns._artifact_rows_for_ladder
    build_button_readiness = ns.build_button_readiness
    _cluster_profile_warning = ns._cluster_profile_warning
    _default_campaign_name = ns._default_campaign_name
    _friendly_bool = ns._friendly_bool
    _friendly_md_profile = ns._friendly_md_profile
    _friendly_md_status = ns._friendly_md_status
    _friendly_remote_sync = ns._friendly_remote_sync
    _path_name = ns._path_name
    _persisted_choice = ns._persisted_choice
    _quote_path = ns._quote_path
    _render_action_history = ns._render_action_history
    _render_artifact_verification_workspace = ns._render_artifact_verification_workspace
    _render_badges = ns._render_badges
    _render_cluster_health_notice = ns._render_cluster_health_notice
    _render_decision_workspace = ns._render_decision_workspace
    _base_render_draft_action = ns._render_draft_action
    _render_ladder_workflow_macros = ns._render_ladder_workflow_macros
    _render_latest_preflight_summary = ns._render_latest_preflight_summary
    _base_render_launch_action = ns._render_launch_action
    _render_make_ingest_action = ns._render_make_ingest_action
    _render_metric_cards = ns._render_metric_cards
    _render_md_slate_monitor = ns._render_md_slate_monitor
    _render_md_slate_recovery_center = ns._render_md_slate_recovery_center
    _render_recommended_card = ns._render_recommended_card
    _render_remote_console = ns._render_remote_console
    _render_review_workspace = ns._render_review_workspace
    _render_stage_progress = ns._render_stage_progress
    _render_export_pack = ns._render_export_pack
    _report_action_result = ns._report_action_result
    _ladder_next_step_copy = ns._ladder_next_step_copy
    draft_bura_cancel_action = ns.draft_bura_cancel_action
    draft_bura_normalize_action = ns.draft_bura_normalize_action
    draft_bura_preflight_action = ns.draft_bura_preflight_action
    draft_bura_pull_package_action = ns.draft_bura_pull_package_action
    draft_bura_submit_action = ns.draft_bura_submit_action
    draft_bura_upload_campaign_action = ns.draft_bura_upload_campaign_action
    get_cluster_profile = ns.get_cluster_profile
    queue_bura_fetch_logs_action = ns.queue_bura_fetch_logs_action
    queue_bura_poll_action = ns.queue_bura_poll_action
    queue_bura_readiness_action = ns.queue_bura_readiness_action
    queue_bura_reconcile_campaign_action = ns.queue_bura_reconcile_campaign_action
    submit_bura_full_autopilot_action = ns.submit_bura_full_autopilot_action
    submit_finalize_md_stage_action = ns.submit_finalize_md_stage_action
    submit_parse_bura_md_benchmark_action = ns.submit_parse_bura_md_benchmark_action
    submit_prepare_bura_md_benchmark_action = ns.submit_prepare_bura_md_benchmark_action
    submit_prepare_manual_md_stage_action = ns.submit_prepare_manual_md_stage_action
    submit_prepare_md_stage_action = ns.submit_prepare_md_stage_action

    ladders = list(state.get("peptides", []))
    st.header("MD Validation")
    st.caption("This page explains the guided MD ladder in plain language and shows exactly which BURA step should happen next.")
    md_default_section = (
        "Manual MD sandbox"
        if not ladders
        else "Review & ingest"
        if any(bool(item.get("ready_for_review")) for item in ladders)
        else "Ladder overview"
    )
    guided_mode = str(state.get("workflow_mode", "Expert mode")) == "Guided thesis mode"
    if guided_mode:
        issue_sections: list[str] = []
        artifact_rows = state.get("artifact_verification", [])
        artifact_attention = any(
            isinstance(row, dict)
            and (
                str(row.get("verification_state", "")) == "Attention needed"
                or str(row.get("severity", "")) in {"warning", "error"}
            )
            for row in artifact_rows
        ) if isinstance(artifact_rows, list) else False
        if state.get("md_slate_exceptions") or artifact_attention:
            issue_sections.extend(["Recovery center", "Artifact verification"])
        section_options = [
            section
            for section in GUIDED_MD_SECTIONS
            if section != "Advanced / debug tools"
        ]
        for section in issue_sections:
            if section not in section_options:
                section_options.append(section)
        section_options.append("Advanced / debug tools")
        requested_advanced_section = ""
        session_state = getattr(st, "session_state", {})
        stored_section = str(session_state.get("dashboard_md_section", "")) if isinstance(session_state, dict) else ""
        radio_values = getattr(st, "_radio_values", {})
        radio_section = str(radio_values.get("dashboard_md_section", "")) if isinstance(radio_values, dict) else ""
        for candidate in (radio_section, stored_section):
            if candidate in ADVANCED_MD_SECTIONS and candidate not in section_options:
                requested_advanced_section = candidate
                break
        if requested_advanced_section and isinstance(session_state, dict):
            session_state["dashboard_md_section"] = "Advanced / debug tools"
            session_state["dashboard_md_advanced_section"] = requested_advanced_section
    else:
        section_options = MD_SECTIONS
    selected_section = _persisted_choice(
        st,
        st.radio,
        label="MD validation section",
        options=section_options,
        key="dashboard_md_section",
        query_key="md_section",
        default=md_default_section,
        write_query=True,
    )
    if selected_section == "Advanced / debug tools":
        requested_advanced_section = ""
        if guided_mode:
            session_state = getattr(st, "session_state", {})
            requested_advanced_section = (
                str(session_state.get("dashboard_md_advanced_section", ""))
                if isinstance(session_state, dict)
                else ""
            )
        selected_section = st.selectbox(
            "Advanced MD Validation section",
            ADVANCED_MD_SECTIONS,
            index=ADVANCED_MD_SECTIONS.index(requested_advanced_section)
            if requested_advanced_section in ADVANCED_MD_SECTIONS
            else 0,
            key="dashboard_md_advanced_section",
        )
        st.info(
            "Advanced MD sections remain available for benchmark, recovery, history, and command inspection. Guided mode keeps normal execution in the main MD sections."
        )

    def _render_launch_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "MD Validation")
        kwargs.setdefault("section", selected_section)
        return _base_render_launch_action(*args, **kwargs)

    def _render_draft_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "MD Validation")
        kwargs.setdefault("section", selected_section)
        return _base_render_draft_action(*args, **kwargs)

    if selected_section == "Manual MD sandbox":
        st.subheader("Manual MD sandbox")
        st.info(
            "Use this for infrastructure tests or by-the-way peptide checks. It prepares a normal MD campaign, "
            "but the source batch is marked as manual sandbox evidence instead of AL proposal evidence."
        )
        runs = list(state.get("runs", []))
        if not runs:
            st.warning("Create or select a run first. Manual MD sandbox campaigns still need a run folder to store the campaign files.")
            return
        run_options = [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs]
        selected_run = st.selectbox("Store sandbox campaign under run", run_options, key="manual_md_run")
        run = next(item for item in runs if str(item.get("run_display_name", item.get("run_name", ""))) == selected_run)
        manual_sequence = st.text_input("Peptide sequence", value="AAAAA", key="manual_md_sequence").strip().upper()
        manual_profile = st.selectbox(
            "MD profile",
            ["line_smoke", "production_smoke", "full"],
            index=2,
            key="manual_md_profile",
        )
        safe_sequence = "".join(ch for ch in manual_sequence if ch.isalnum())[:16] or "peptide"
        default_campaign = f"manual_{manual_profile}_{safe_sequence}".lower()
        manual_campaign = st.text_input("Campaign name", value=default_campaign, key="manual_md_campaign")
        bura_profile = get_cluster_profile(state.get("profiles", {}), "bura")
        default_exclude = str((bura_profile or {}).get("default_exclude_nodes", ""))
        manual_exclude_nodes = st.text_input("Exclude BURA nodes", value=default_exclude, key="manual_md_exclude_nodes")
        reuse_pdb_from = st.text_input(
            "Reuse PDB from campaign folder (optional)",
            value="",
            key="manual_md_reuse_pdb",
            help="Leave empty unless you already have a compatible PDB in another campaign's PDBs folder.",
        )
        if manual_profile == "full":
            st.warning("Full MD is the expensive path. This is allowed, but use it intentionally because it can consume real BURA queue time.")
        command = (
            "python -m active_learning_thesis prepare-manual-md-stage "
            f"--run-dir {_quote_path(run['run_dir'])} "
            f"--sequence {manual_sequence or '<sequence>'} "
            f"--campaign {manual_campaign or '<campaign>'} "
            f"--md-profile {manual_profile} --cluster bura"
        )
        if manual_exclude_nodes:
            command += f" --exclude-nodes {manual_exclude_nodes}"
        if reuse_pdb_from:
            command += f" --reuse-pdb-from {_quote_path(reuse_pdb_from)}"
        _render_launch_action(
            st,
            label="Prepare manual MD sandbox campaign",
            command=command,
            key_prefix="manual_md_sandbox_prepare",
            button_text="Prepare locally",
            what="Create a local MD campaign for a peptide you typed manually, without requiring it to come from an AL proposal batch.",
            when="Use this to test BURA/MD plumbing or run an optional by-the-way peptide experiment.",
            produces="A campaign folder under the selected run plus a `manual_md_sandbox.json` marker and BURA command hints.",
            next_step="After the action succeeds and the dashboard refreshes, choose this peptide and use Remote BURA to upload, preflight, submit, poll, and copy back.",
            action_kind="prepare-manual-md-stage",
            on_submit=lambda run=run, manual_sequence=manual_sequence, manual_campaign=manual_campaign, manual_profile=manual_profile, manual_exclude_nodes=manual_exclude_nodes, reuse_pdb_from=reuse_pdb_from: submit_prepare_manual_md_stage_action(
                run_root=Path(str(state["run_root"])),
                run_dir=Path(str(run["run_dir"])),
                sequence=manual_sequence,
                campaign=manual_campaign,
                md_profile=manual_profile,
                cluster="bura",
                reuse_pdb_from=reuse_pdb_from or None,
                exclude_nodes=manual_exclude_nodes,
            ),
        )
        return

    if not ladders:
        st.info("No guided or legacy MD campaigns are visible in this workspace view yet.")
        return

    sequence_options = [str(item["sequence"]) for item in ladders]
    selected_sequence = _persisted_choice(
        st,
        st.selectbox,
        label="Choose peptide",
        options=sequence_options,
        key="dashboard_peptide_sequence",
        query_key="peptide",
        default=sequence_options[0],
        write_query=True,
    )
    ladder = next(item for item in ladders if item["sequence"] == selected_sequence)
    current = ladder.get("current") or ladder.get("full") or ladder.get("production_smoke") or ladder.get("line_smoke")
    campaign_options = list(ladder.get("campaign_options", []))
    if campaign_options:
        label_to_campaign = {}
        for campaign in campaign_options:
            label = (
                f"{campaign.get('campaign', '')} | "
                f"{_friendly_md_profile(str(campaign.get('md_profile', '')), short=True)} | "
                f"{_friendly_md_status(str(campaign.get('job_root_status', '')))} | "
                f"{_friendly_remote_sync(str(campaign.get('sync_status', 'not_synced')))}"
            )
            run_label = str(campaign.get("run_display_name", ""))
            if run_label:
                label = f"{label} | {run_label}"
            label_to_campaign[label] = campaign
        campaign_labels = list(label_to_campaign)
        default_campaign_label = next(
            (
                label
                for label, campaign in label_to_campaign.items()
                if current and str(campaign.get("campaign_dir", "")) == str(current.get("campaign_dir", ""))
            ),
            campaign_labels[0],
        )
        selected_campaign_label = _persisted_choice(
            st,
            st.selectbox,
            label="Choose campaign",
            options=campaign_labels,
            key=f"dashboard_md_campaign_{selected_sequence}",
            query_key="md_campaign",
            default=default_campaign_label,
            write_query=True,
        )
        current = label_to_campaign.get(selected_campaign_label, current)
    selected_ladder = dict(ladder)
    if current:
        selected_ladder["current"] = current
        if str(current.get("md_profile", "")) == "full":
            selected_ladder["full"] = current
        elif str(current.get("md_profile", "")) == "production_smoke":
            selected_ladder["production_smoke"] = current
        elif str(current.get("md_profile", "")) == "line_smoke":
            selected_ladder["line_smoke"] = current
        selected_ladder["run_dir"] = str(current.get("run_dir", selected_ladder.get("run_dir", "")))
        selected_ladder["run_display_name"] = str(current.get("run_display_name", selected_ladder.get("run_display_name", "")))
        selected_ladder["sync_status"] = str(current.get("sync_status", selected_ladder.get("sync_status", "not_synced")))
        selected_ladder["remote_job_id"] = str(current.get("remote_job_id", selected_ladder.get("remote_job_id", "")))
        selected_ladder["cluster"] = str(current.get("cluster", selected_ladder.get("cluster", "")))
        selected_ladder["source_batch_csv"] = str(current.get("source_batch_csv", selected_ladder.get("source_batch_csv", "")))
        selected_ladder["source_batch_kind"] = str(current.get("source_batch_kind", selected_ladder.get("source_batch_kind", "")))
        selected_ladder["reuse_pdb_from"] = str(current.get("campaign_dir", selected_ladder.get("reuse_pdb_from", "")))
        selected_ladder["exclude_nodes"] = str(current.get("exclude_nodes", selected_ladder.get("exclude_nodes", "")))

    st.subheader(f"Peptide: {ladder['sequence']}")
    st.caption(f"Run context: {ladder.get('run_display_name', _path_name(ladder.get('run_dir', '')))}")
    _render_badges(st, list(ladder.get("badges", [])))
    _render_recommended_card(st, **_ladder_next_step_copy(ladder))
    if not (guided_mode and selected_section == "Remote BURA"):
        _render_ladder_workflow_macros(st, state, ladder)
    _render_metric_cards(
        st,
        [
            ("Next ladder stage", ladder.get("next_profile_label", "-") or "-"),
            ("Remote state", _friendly_remote_sync(str(ladder.get("sync_status", "not_synced")))),
            ("Tracked remote job", ladder.get("remote_job_id", "-") or "-"),
            ("Ready for review", _friendly_bool(bool(ladder["ready_for_review"]))),
        ],
    )
    if current and current.get("local_stage_path"):
        st.caption(f"Staged download path: {current['local_stage_path']}")

    if selected_section == "Review & ingest":
        review_pipeline = (
            list(state.get("peptide_inventory", {}).get("review_pipeline", []))
            if isinstance(state.get("peptide_inventory", {}), dict)
            else []
        )
        run_review_rows = [
            row
            for row in review_pipeline
            if str(row.get("run", "")) == str(ladder.get("run_display_name", ladder.get("run_name", "")))
        ]
        if run_review_rows:
            st.markdown("#### Returned full-analysis peptides")
            st.caption("This queue shows all returned full-analysis peptides for the current run, so you can keep the selected peptide card while still seeing the broader review / ingest handoff.")
            st.dataframe(run_review_rows)
        _render_review_workspace(st, ladder, run_root=Path(str(state["run_root"])), state=state)
        return

    if selected_section == "BURA performance":
        st.subheader("BURA performance benchmark")
        st.info(
            "This is optional and separate from your thesis MD result. It creates short production-only test jobs "
            "from the current campaign package so we can compare BURA resource layouts before changing the real full-MD launcher."
        )
        if not current or not current.get("campaign_dir"):
            st.warning("Choose a peptide with a prepared campaign first.")
            return
        campaign_dir = Path(str(current["campaign_dir"]))
        default_benchmark_name = f"perf_{ladder['sequence'].lower()}"
        benchmark_name = st.text_input(
            "Benchmark folder name",
            value=default_benchmark_name,
            key=f"bura_perf_name_{ladder['sequence']}",
        )
        benchmark_nsteps_text = st.text_input(
            "Benchmark production steps",
            value="50000",
            key=f"bura_perf_nsteps_{ladder['sequence']}",
            help="50,000 steps is about 1 ns for your current full profile and is enough to compare speed without spending a day.",
        )
        try:
            benchmark_nsteps = max(1000, min(1000000, int(str(benchmark_nsteps_text).strip())))
        except ValueError:
            benchmark_nsteps = 50000
            st.warning("Benchmark production steps must be a number. Using 50000 for the generated command.")
        benchmark_walltime = st.text_input(
            "Benchmark walltime",
            value="02:00:00",
            key=f"bura_perf_walltime_{ladder['sequence']}",
        )
        layout_options = [
            "1n_1mpi_48omp",
            "1n_2mpi_24omp",
            "1n_4mpi_12omp",
            "1n_6mpi_8omp",
            "1n_8mpi_6omp",
            "2n_2mpi_24omp",
            "2n_4mpi_12omp",
            "2n_6mpi_8omp",
            "2n_8mpi_6omp",
            "4n_4mpi_12omp",
            "4n_6mpi_8omp",
            "4n_8mpi_6omp",
            "6n_4mpi_12omp",
            "6n_6mpi_8omp",
            "6n_8mpi_6omp",
        ]
        layout_arg = st.text_input(
            "Layouts to test",
            value="1n_1mpi_48omp,1n_4mpi_12omp,1n_6mpi_8omp,1n_8mpi_6omp,2n_4mpi_12omp",
            key=f"bura_perf_layouts_{ladder['sequence']}",
            help="Comma-separated. Safe defaults compare one-node OpenMP-heavy, one-node mixed MPI/OMP, and a two-node candidate.",
        )
        command = (
            "python -m active_learning_thesis prepare-bura-md-benchmark "
            f"--campaign-dir {_quote_path(campaign_dir)} "
            f"--sequence {ladder['sequence']} "
            f"--benchmark-name {benchmark_name or '<benchmark>'} "
            f"--nsteps {int(benchmark_nsteps)} "
            f"--walltime {benchmark_walltime or '02:00:00'}"
        )
        if layout_arg:
            command += f" --layouts {layout_arg}"
        _render_launch_action(
            st,
            label="Prepare BURA benchmark package",
            command=command,
            key_prefix=f"bura_perf_prepare_{ladder['sequence']}",
            button_text="Prepare benchmark locally",
            what="Create a separate benchmark folder with short production jobs and multiple SLURM/MPI/OpenMP layouts.",
            when="Use this after a package has at least `equi2.gro` and `equi2.cpt`, usually after a completed or copied-back full run.",
            produces="A `bura_benchmarks/<name>` folder with submit commands and a manifest.",
            next_step="Upload that benchmark folder to BURA, run `submit_bura_benchmarks.sh`, copy it back, then parse results below.",
            action_kind="prepare-bura-md-benchmark",
            on_submit=lambda benchmark_name=benchmark_name, benchmark_nsteps=benchmark_nsteps, layout_arg=layout_arg, benchmark_walltime=benchmark_walltime: submit_prepare_bura_md_benchmark_action(
                run_root=Path(str(state["run_root"])),
                campaign_dir=campaign_dir,
                sequence=str(ladder["sequence"]),
                benchmark_name=benchmark_name,
                nsteps=int(benchmark_nsteps),
                layouts=layout_arg,
                walltime=benchmark_walltime or "02:00:00",
                related_run=str(ladder["run_dir"]),
            ),
        )

        benchmark_root = campaign_dir / "bura_benchmarks"
        benchmark_dirs = sorted(path for path in benchmark_root.glob("*") if path.is_dir()) if benchmark_root.exists() else []
        if not benchmark_dirs:
            st.caption("No benchmark folders found yet for this campaign.")
            return

        selected_benchmark = st.selectbox(
            "Existing benchmark folder",
            [path.name for path in benchmark_dirs],
            key=f"bura_perf_existing_{ladder['sequence']}",
        )
        benchmark_dir = next(path for path in benchmark_dirs if path.name == selected_benchmark)
        st.caption(f"Benchmark path: `{benchmark_dir}`")
        next_commands = benchmark_dir / "NEXT_BURA_BENCHMARK_COMMANDS.md"
        if next_commands.exists():
            with st.expander("BURA commands", expanded=False):
                st.code(next_commands.read_text(encoding="utf-8"), language="markdown")

        parse_command = (
            "python -m active_learning_thesis parse-bura-md-benchmark "
            f"--benchmark-dir {_quote_path(benchmark_dir)}"
        )
        _render_launch_action(
            st,
            label="Parse copied-back benchmark results",
            command=parse_command,
            key_prefix=f"bura_perf_parse_{ladder['sequence']}_{selected_benchmark}",
            button_text="Parse benchmark",
            what="Read GROMACS benchmark logs and extract wall time, ns/day, observed MPI ranks, and OpenMP threads.",
            when="Use this after the benchmark folder has been copied back from BURA.",
            produces="A `benchmark_results.csv` table inside the benchmark folder.",
            next_step="Use the fastest successful row as the candidate layout for a future full-MD launcher update.",
            action_kind="parse-bura-md-benchmark",
            on_submit=lambda benchmark_dir=benchmark_dir: submit_parse_bura_md_benchmark_action(
                run_root=Path(str(state["run_root"])),
                benchmark_dir=benchmark_dir,
                sequence=str(ladder["sequence"]),
                campaign_dir=campaign_dir,
                related_run=str(ladder["run_dir"]),
            ),
        )
        results_path = benchmark_dir / "benchmark_results.csv"
        if results_path.exists():
            with results_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            st.markdown("#### Parsed benchmark results")
            st.dataframe(rows)
        else:
            st.caption("No parsed benchmark results yet.")
        return

    if selected_section == "Slate monitor":
        _render_md_slate_monitor(st, state, ladder)
        return

    if selected_section == "Recovery center":
        _render_md_slate_recovery_center(st, state, ladder)
        return

    if selected_section == "Artifact verification":
        _render_artifact_verification_workspace(
            st,
            _artifact_rows_for_ladder(state, ladder),
            title="Artifact verification",
            caption="Use this to confirm the expected local files really exist for the current peptide: package inputs, staged downloads, runtime outputs, SASA/AP summaries, and ingest files.",
            key_prefix=f"md_artifact_{ladder['sequence']}",
            render_export_pack=_render_export_pack,
        )
        return

    if selected_section == "Ladder overview":
        left_col, right_col = st.columns([1.2, 1])
        with left_col:
            st.subheader("Stage progression")
            _render_stage_progress(st, ladder)
            st.subheader("Campaign history")
            st.dataframe(ladder["campaigns"])
        with right_col:
            st.subheader("Stage notes")
            stage_guide = [
                {
                    "stage": info["label"],
                    "purpose": info["description"],
                    "ingest_ready": "yes" if profile_key == "full" else "no",
                }
                for profile_key, info in MD_PROFILE_INFO.items()
            ]
            st.write("Stage guide")
            st.dataframe(stage_guide)
            st.write(f"Current campaign: `{current['campaign']}`" if current else "Current campaign: `-`")
            st.write(f"Source batch: `{ladder['source_batch_csv'] or '-'}`")
            st.write(f"Reuse PDB from: `{ladder['reuse_pdb_from'] or '-'}`")
            st.write(f"Default exclude nodes: `{ladder['exclude_nodes'] or '-'}`")
            if current and current.get("review_notes"):
                st.write(f"Review notes: {current['review_notes']}")
            if current and current.get("cgmd_label") not in {"", None}:
                st.write(f"Current label: `{current['cgmd_label']}`")
        return

    if selected_section == "Local MD actions":
        st.subheader("Local MD actions")
        if ladder["next_profile"] and ladder["source_batch_csv"]:
            default_campaign_name = _default_campaign_name(ladder)
            campaign_name = st.text_input(
                "Campaign name for next stage",
                value=default_campaign_name,
                key=f"campaign_name_{ladder['sequence']}",
            )
            exclude_nodes = st.text_input(
                "Exclude nodes",
                value=str(ladder.get("exclude_nodes", "")),
                key=f"exclude_nodes_{ladder['sequence']}",
            )
            command = (
                "python -m active_learning_thesis prepare-md-stage "
                f"--run-dir {_quote_path(ladder['run_dir'])} "
                f"--batch-csv {_quote_path(ladder['source_batch_csv'])} "
                f"--sequence {ladder['sequence']} "
                f"--campaign {campaign_name} "
                f"--md-profile {ladder['next_profile']} "
                f"--cluster {ladder.get('cluster', 'bura') or 'bura'}"
            )
            reuse_pdb_from = str(ladder.get("reuse_pdb_from", ""))
            if reuse_pdb_from:
                command += f" --reuse-pdb-from {_quote_path(reuse_pdb_from)}"
            if exclude_nodes:
                command += f" --exclude-nodes {exclude_nodes}"
            _render_launch_action(
                st,
                label=f"Prepare {_friendly_md_profile(ladder['next_profile'])}",
                command=command,
                key_prefix=f"prepare_{ladder['sequence']}",
                button_text="Run locally",
                what="Create the next guided MD campaign locally from the source batch and peptide sequence.",
                when="Use this when the ladder says the next missing stage is not prepared yet.",
                produces="A new local campaign folder plus the remote BURA command hints for that stage.",
                next_step="The campaign can then be uploaded to BURA for remote execution.",
                contract_id="prepare-md-stage",
                readiness=build_button_readiness(state, "prepare-md-stage", ladder=selected_ladder),
                on_submit=lambda campaign_name=campaign_name, exclude_nodes=exclude_nodes, reuse_pdb_from=reuse_pdb_from: submit_prepare_md_stage_action(
                    run_root=Path(str(state["run_root"])),
                    run_dir=Path(str(ladder["run_dir"])),
                    batch_csv=Path(str(ladder["source_batch_csv"])),
                    sequence=str(ladder["sequence"]),
                    campaign=campaign_name,
                    md_profile=str(ladder["next_profile"]),
                    cluster=str(ladder.get("cluster", "bura") or "bura"),
                    reuse_pdb_from=reuse_pdb_from or None,
                    exclude_nodes=exclude_nodes,
                ),
            )

        if current and not current.get("legacy"):
            finalize_command = (
                "python -m active_learning_thesis finalize-md-stage "
                f"--campaign-dir {_quote_path(current['campaign_dir'])}"
            )
            if current.get("local_stage_path"):
                finalize_command += f" --staged-package-dir {_quote_path(current['local_stage_path'])}"
            finalize_label = f"Finalize local outputs for {current['campaign']}"
            if str(current.get("sync_status", "")) != "outputs_returned":
                finalize_label = f"Re-parse local outputs for {current['campaign']}"
            _render_launch_action(
                st,
                label=finalize_label,
                command=finalize_command,
                key_prefix=f"finalize_{current['campaign']}",
                button_text="Run locally",
                what="Parse the copied-back guided MD outputs and update `md_review.csv` plus the next ladder recommendation.",
                when="Use this after remote outputs have been copied back, or when you manually placed outputs into the campaign directory.",
                produces="Updated review status and the next guided MD recommendation for this peptide.",
                next_step="If the full stage is complete, the peptide may become ready for review and ingest.",
                contract_id="finalize-md-stage",
                readiness=build_button_readiness(state, "finalize-md-stage", ladder=selected_ladder),
                on_submit=lambda current=current: submit_finalize_md_stage_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    related_run=str(ladder["run_dir"]),
                    staged_package_dir=str(current.get("local_stage_path", "")) or None,
                ),
            )

        _render_make_ingest_action(
            st,
            selected_ladder,
            run_root=Path(str(state["run_root"])),
            key_prefix=f"ingest_{ladder['sequence']}",
            state=state,
        )
        return

    if selected_section == "Remote BURA":
        st.subheader("Remote BURA actions")
        st.info("Recommended order: upload campaign -> normalize scripts -> preflight -> submit -> poll queue -> copy back -> re-parse local outputs.")
        bura_profile = get_cluster_profile(state.get("profiles", {}), "bura")
        if bura_profile is None:
            st.warning(_cluster_profile_warning(state, "bura"))
            return
        if not current:
            return
        _render_cluster_health_notice(st, state, "bura")
        _render_metric_cards(
            st,
            [
                ("Remote state", current.get("sync_status", "not_synced")),
                ("Tracked job", current.get("remote_job_id", "-") or "-"),
                ("Remote path", _path_name(current.get("remote_path", "")) or "-"),
            ],
        )
        exclude_nodes = st.text_input(
            "BURA exclude nodes override",
            value=str(current.get("exclude_nodes") or bura_profile.get("default_exclude_nodes", "")),
            key=f"bura_exclude_{ladder['sequence']}",
        )
        current_sync_status = str(current.get("sync_status", "not_synced"))
        if str(current.get("md_profile", "")) != "full":
            st.warning("Profile guard: this selected campaign is not full MD. The one-button runner will not submit smoke campaigns.")
        st.markdown("#### Full simulation runner")
        runner_actions = _campaign_actions(list(ladder.get("dashboard_actions", [])), current, str(ladder["sequence"]))
        inferred_stage_path = str(current.get("local_stage_path", "") or _latest_successful_pull_stage_path(runner_actions))
        runner_decision = _bura_full_runner_decision(
            current=current,
            sequence=str(ladder["sequence"]),
            actions=list(ladder.get("dashboard_actions", [])),
        )
        runner_step = str(runner_decision.get("step", "blocked"))
        latest_runner_action = _latest_action(runner_actions)
        _render_metric_cards(
            st,
            [
                ("Selected peptide", str(ladder["sequence"])),
                ("Selected campaign", str(current.get("campaign", _path_name(current.get("campaign_dir", "")))) or "-"),
                ("Profile", _friendly_md_profile(str(current.get("md_profile", "")))),
                ("Next safe step", str(runner_decision.get("label", "-"))),
            ],
        )
        st.caption(str(runner_decision.get("message", "")))
        if runner_step not in {"blocked", "waiting", "done"}:
            st.caption(
                "Autopilot will continue past this checkpoint automatically: upload, normalize, preflight, submit, poll until done, copy back, and parse."
            )
            st.caption(
                "If a remote/network step fails, the state is kept at the last safe checkpoint; click this button again to continue from the failed checkpoint."
            )
        if latest_runner_action:
            st.caption(
                "Latest BURA action: "
                f"{latest_runner_action.get('title', latest_runner_action.get('kind', '-'))} "
                f"({latest_runner_action.get('status', '-')})"
            )
            latest_stdout = _tail_action_log(latest_runner_action, "stdout_log")
            latest_stderr = _tail_action_log(latest_runner_action, "stderr_log")
            if latest_stdout or latest_stderr:
                with st.expander("Latest stdout/stderr summary", expanded=False):
                    if latest_stdout:
                        st.markdown("**stdout**")
                        st.code(latest_stdout, language="text")
                    if latest_stderr:
                        st.markdown("**stderr**")
                        st.code(latest_stderr, language="text")
        runner_disabled = runner_step in {"blocked", "waiting", "done"}
        if st.button(
            "Do full simulation",
            key=f"bura_full_runner_{ladder['sequence']}_{current.get('campaign', '')}",
            disabled=runner_disabled,
            help="Queues a long-running autopilot worker for the selected full campaign.",
        ):
            action = submit_bura_full_autopilot_action(
                run_root=Path(str(state["run_root"])),
                campaign_dir=Path(str(current["campaign_dir"])),
                sequence=str(ladder["sequence"]),
                related_run=str(ladder["run_dir"]),
                exclude_nodes=exclude_nodes,
            )
            _report_action_result(st, action)
        if guided_mode:
            with st.expander("Advanced / individual BURA actions", expanded=False):
                st.caption(
                    "Guided mode keeps the old upload, normalize, preflight, submit, poll, pull, parse, logs, and cancel controls hidden by default."
                )
                show_individual_controls = st.checkbox(
                    "Show individual BURA action controls",
                    value=False,
                    key=f"show_bura_individual_controls_{ladder['sequence']}",
                )
            if not show_individual_controls:
                return
        st.markdown("#### Upload and staging")
        if current_sync_status == "not_synced":
            _render_draft_action(
                st,
                label="Upload campaign to BURA",
                command=f"scp -r {_quote_path(current['campaign_dir'])} {bura_profile['username']}@{bura_profile['host']}:<campaign root>",
                key_prefix=f"bura_upload_{ladder['sequence']}",
                what="Copy the prepared guided MD campaign folder from your machine to the configured BURA campaign root.",
                when="Use this after preparing a local stage and before any remote BURA commands.",
                produces="A staged remote campaign directory on BURA.",
                next_step="Normalization, preflight, and submit become available once the upload succeeds.",
                contract_id="bura-upload-campaign",
                readiness=build_button_readiness(state, "bura-upload-campaign", ladder=selected_ladder),
                on_submit=lambda current=current: draft_bura_upload_campaign_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
            _render_launch_action(
                st,
                label="Reconcile existing BURA upload",
                command="ssh <bura> <verify existing campaign folder/scripts/package>",
                key_prefix=f"bura_reconcile_{ladder['sequence']}",
                button_text="Verify existing upload",
                what="Check whether this campaign already exists on BURA and bind the dashboard to that remote folder without copying files again.",
                when="Use this if you uploaded the campaign manually or can already see the campaign folder on BURA.",
                produces="A staged remote campaign record if the BURA folder, scripts, and peptide package are present.",
                next_step="After the check succeeds, Normalize BURA scripts, Preflight, and Submit become available.",
                action_kind="bura-reconcile-campaign",
                on_submit=lambda current=current: queue_bura_reconcile_campaign_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
        else:
            st.info("Upload is only needed while the campaign exists locally and has not yet been staged on BURA.")

        st.markdown("#### Prepare and submit")
        if current_sync_status == "staged_remote":
            _render_launch_action(
                st,
                label="Run BURA submit readiness check",
                command="ssh <bura> <check campaign/scripts/module/package>",
                key_prefix=f"bura_readiness_{ladder['sequence']}",
                button_text="Run remote check",
                what="Verify that the remote campaign, required scripts, package directory, and module load command are all in place before consuming queue time.",
                when="Use this after upload and before normalization / preflight / submit, or any time you want a quick sanity check on the staged BURA campaign.",
                produces="A read-only readiness result for the current BURA campaign.",
                next_step="If it passes, continue with normalization, the real BURA preflight script, and then submission.",
                action_kind="bura-submit-readiness",
                on_submit=lambda current=current: queue_bura_readiness_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
            _render_latest_preflight_summary(
                st,
                title="Latest BURA readiness status",
                actions=list(ladder.get("dashboard_actions", [])),
                kind="bura-submit-readiness",
            )
            _render_draft_action(
                st,
                label="Normalize BURA scripts",
                command='find . -type f -name "*.sh" -exec dos2unix {} \\; && find . -type f -name "*.sh" -exec chmod u+x {} \\;',
                key_prefix=f"bura_normalize_{ladder['sequence']}",
                what="Fix line endings and executable bits on staged shell scripts so the BURA environment can run them cleanly.",
                when="Use this immediately after a fresh upload to BURA.",
                produces="Normalized remote scripts ready for preflight.",
                next_step="The campaign can then run the BURA preflight check.",
                contract_id="bura-normalize-scripts",
                readiness=build_button_readiness(state, "bura-normalize-scripts", ladder=selected_ladder),
                on_submit=lambda current=current: draft_bura_normalize_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
            _render_draft_action(
                st,
                label="Run BURA preflight",
                command="bash ./preflight_bura.sh",
                key_prefix=f"bura_preflight_{ladder['sequence']}",
                what="Run the campaign's preflight script on BURA before consuming queue time.",
                when="Use this after normalization and before submission.",
                produces="A preflight check result that tells you whether the campaign is safe to submit.",
                next_step="Once preflight is clean, you can submit the chain to the queue.",
                action_kind="bura-preflight",
                on_submit=lambda current=current: draft_bura_preflight_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
            _render_draft_action(
                st,
                label="Submit BURA chain",
                command=f"bash ./submit_chain.sh --exclude {exclude_nodes or '<none>'} {ladder['sequence']}",
                key_prefix=f"bura_submit_{ladder['sequence']}",
                what="Submit the prepared MD chain for this peptide to the BURA scheduler.",
                when="Use this only after the upload, normalization, and preflight steps are already complete.",
                produces="A tracked remote BURA job / chain state in the dashboard.",
                next_step="Queue polling and copy-back become the next actions.",
                action_kind="bura-submit-chain",
                on_submit=lambda current=current, exclude_nodes=exclude_nodes: draft_bura_submit_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                    exclude_nodes=exclude_nodes,
                ),
            )
        else:
            st.info("Normalization, preflight, and submit become available only after the campaign has been uploaded to BURA.")

        st.markdown("#### Monitor and copy back")
        if current.get("remote_job_id") or current_sync_status in {"submitted", "running"}:
            _render_launch_action(
                st,
                label="Poll BURA queue",
                command=f"ssh {bura_profile['username']}@{bura_profile['host']} squeue -u {bura_profile['username']}",
                key_prefix=f"bura_poll_{ladder['sequence']}",
                button_text="Poll remote queue",
                what="Refresh the tracked BURA queue state for this peptide's active remote campaign.",
                when="Use this after a chain submission is active on BURA.",
                produces="An updated queue snapshot and sync status for the ladder.",
                next_step="When the chain finishes, copy the outputs back and finalize them locally.",
                contract_id="bura-poll-squeue",
                readiness=build_button_readiness(state, "bura-poll-squeue", ladder=selected_ladder),
                on_submit=lambda current=current: queue_bura_poll_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                    remote_job_id=str(current.get("remote_job_id", "")),
                ),
            )
        else:
            st.info("Queue polling becomes available after a tracked BURA submission exists.")
        if current_sync_status in {"submitted", "running", "outputs_staged", "outputs_returned"}:
            _render_draft_action(
                st,
                label="Copy BURA outputs back",
                command="scp -r <remote package dir> <dashboard staging dir>",
                key_prefix=f"bura_pull_{ladder['sequence']}",
                what="Copy the finished remote BURA package back into the dashboard's safe local staging area.",
                when="Use this after the remote chain has produced outputs you want to parse locally.",
                produces="A staged local copy of the remote package outputs.",
                next_step="The next local step is re-parsing / finalizing the copied-back outputs.",
                action_kind="bura-pull-package",
                on_submit=lambda current=current: draft_bura_pull_package_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
        else:
            st.info("Copy-back becomes available after a BURA chain has been submitted.")

        st.markdown("#### Parse returned outputs")
        if current_sync_status in {"outputs_staged", "outputs_returned", "finalized_local"} or inferred_stage_path:
            finalize_command = (
                "python -m active_learning_thesis finalize-md-stage "
                f"--campaign-dir {_quote_path(current['campaign_dir'])}"
            )
            if inferred_stage_path:
                finalize_command += f" --staged-package-dir {_quote_path(inferred_stage_path)}"
            _render_launch_action(
                st,
                label=f"Parse / finalize copied-back outputs for {current['campaign']}",
                command=finalize_command,
                key_prefix=f"bura_finalize_{current['campaign']}_{ladder['sequence']}",
                button_text="Parse returned outputs",
                what="Import the copied-back BURA package into the live campaign folder and parse the MD evidence.",
                when="Use this after BURA outputs have been copied back into dashboard staging.",
                produces="Updated `md_review.csv` values such as AP, SASA paths, MD runtime, and final campaign status.",
                next_step="After this succeeds, open Review & ingest to inspect the parsed values and decide the human label.",
                contract_id="finalize-md-stage",
                readiness=build_button_readiness(state, "finalize-md-stage", ladder=selected_ladder),
                on_submit=lambda current=current: submit_finalize_md_stage_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    related_run=str(ladder["run_dir"]),
                    staged_package_dir=inferred_stage_path or None,
                ),
            )
        else:
            st.info("Parsing becomes available after copied-back BURA outputs are staged locally.")

        st.markdown("#### Logs and job control")
        bura_reference_rows = [
            {"artifact": "Tracked job", "path": str(current.get("remote_job_id", "")) or "-", "why_it_matters": "Use this job id when polling or cancelling the active BURA chain."},
            {"artifact": "Campaign root", "path": str(current.get("remote_path", "")) or "-", "why_it_matters": "Remote campaign directory currently associated with this peptide on BURA."},
            {"artifact": "Copied-back staging", "path": str(current.get("local_stage_path", "")) or "-", "why_it_matters": "Safe local staging area for copied-back BURA outputs."},
        ]
        st.dataframe(bura_reference_rows)
        if current_sync_status in {"submitted", "running", "outputs_staged", "outputs_returned"}:
            _render_launch_action(
                st,
                label="Fetch latest BURA logs",
                command="ssh <bura> 'tail campaign log files'",
                key_prefix=f"bura_logs_{ladder['sequence']}",
                button_text="Fetch remote logs",
                what="Inspect the latest queue or package logs on BURA without copying the full outputs back first.",
                when="Use this when a chain is running, recently failed, or finished but you want a quick remote sanity check.",
                produces="A read-only dashboard action whose stdout contains the latest matching BURA log excerpts.",
                next_step="Use the excerpts to decide whether to poll again, cancel the chain, or copy outputs back.",
                action_kind="bura-fetch-logs",
                on_submit=lambda current=current: queue_bura_fetch_logs_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                ),
            )
        else:
            st.info("Remote log fetch becomes useful after the campaign has at least been uploaded and started generating remote files.")
        if current.get("remote_job_id") and current_sync_status in {"submitted", "running"}:
            _render_draft_action(
                st,
                label="Cancel tracked BURA job",
                command=f"ssh {bura_profile['username']}@{bura_profile['host']} scancel {current.get('remote_job_id', '')}",
                key_prefix=f"bura_cancel_{ladder['sequence']}",
                what="Request scheduler-side cancellation of the currently tracked BURA chain for this peptide.",
                when="Use this only when the chain is clearly wrong, stuck, or no longer worth queue time.",
                produces="A scheduler cancellation request plus a reset back to a staged campaign state if the cancellation succeeds.",
                next_step="You can inspect remote logs, adjust the campaign, and submit again when ready.",
                action_kind="bura-cancel-job",
                on_submit=lambda current=current, remote_job_id=str(current.get("remote_job_id", "")): draft_bura_cancel_action(
                    run_root=Path(str(state["run_root"])),
                    campaign_dir=Path(str(current["campaign_dir"])),
                    sequence=str(ladder["sequence"]),
                    profile=bura_profile,
                    related_run=str(ladder["run_dir"]),
                    remote_job_id=remote_job_id,
                ),
            )
        else:
            st.info("Remote cancel becomes available only while a tracked BURA chain is still queued or running.")
        _render_remote_console(
            st,
            title="Latest BURA console snapshot",
            actions=list(ladder.get("dashboard_actions", [])),
            kinds={"bura-submit-readiness", "bura-poll-squeue", "bura-inspect-logs", "bura-submit-chain", "bura-cancel-job", "bura-preflight"},
            key_prefix=f"bura_console_{ladder['sequence']}",
        )
        return

    if selected_section == "Decision log":
        _render_decision_workspace(st, state, scope="peptide", ladder=ladder)
        return

    if selected_section == "Recent actions":
        st.subheader("Recent actions")
        _render_action_history(
            st,
            actions=list(ladder.get("dashboard_actions", [])),
            run_root=str(state["run_root"]),
            key_prefix=f"ladder_{ladder['sequence']}",
        )
        return

    st.subheader("Reference commands")
    if ladder["prepare_next_command"]:
        st.markdown("#### Local next-stage command")
        st.code(ladder["prepare_next_command"], language="bash")
    if ladder["make_ingest_command"]:
        st.markdown("#### Ingest CSV generation command")
        st.code(ladder["make_ingest_command"], language="bash")
    if ladder["next_bura_commands"]:
        st.markdown("#### Copyable remote BURA commands")
        st.code(ladder["next_bura_commands"], language="markdown")
