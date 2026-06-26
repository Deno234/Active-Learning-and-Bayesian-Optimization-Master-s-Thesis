from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


MODEL_SECTIONS = [
    "Workflow summary",
    "Local model actions",
    "Thesis freeze",
    "Remote SUPEK",
    "Decision log",
    "Recent actions",
    "Technical details",
]
GUIDED_MODEL_SECTIONS = [
    "Workflow summary",
    "Local model actions",
    "Remote SUPEK",
    "Thesis freeze",
    "Advanced / debug tools",
]
ADVANCED_MODEL_SECTIONS = [
    section
    for section in MODEL_SECTIONS
    if section not in GUIDED_MODEL_SECTIONS and section != "Advanced / debug tools"
]


def _form_container(st, *, key: str):
    form = getattr(st, "form", None)
    if callable(form):
        try:
            return form(key=key, clear_on_submit=False, border=False)
        except TypeError:
            return form(key=key, clear_on_submit=False)
    return st.container()


def _form_submit_button(st, label: str, *, key: str) -> bool:
    form_submit_button = getattr(st, "form_submit_button", None)
    if callable(form_submit_button):
        return bool(form_submit_button(label))
    return bool(st.button(label, key=key))


def _session_value(session_state, key: str) -> str:
    try:
        return str(session_state.get(key, ""))
    except Exception:
        return ""


def _set_session_value(session_state, key: str, value: str) -> None:
    try:
        session_state[key] = value
    except Exception:
        pass


def _rich_panel_choice(st, *, key: str, options: list[str], default: str) -> str:
    if not options:
        return ""
    session_state = getattr(st, "session_state", {})
    ui_mode = _session_value(session_state, "dashboard_ui_mode")
    if ui_mode != "Rich mode" or hasattr(st, "_view"):
        return "__all__"
    if _session_value(session_state, key) not in options:
        _set_session_value(session_state, key, default if default in options else options[0])
    return str(st.selectbox("Open section", options, key=key))


def _show_panel(selected_panel: str, title: str) -> bool:
    return selected_panel in {"", "__all__", title}


def _remember_model_workflow_location(
    st,
    *,
    run_name: str,
    section: str,
    panel_key: str = "",
    panel_value: str = "",
) -> None:
    session_state = getattr(st, "session_state", {})
    _set_session_value(session_state, "dashboard_run_detail_name", run_name)
    _set_session_value(session_state, "dashboard_model_section", section)
    if panel_key and panel_value:
        _set_session_value(session_state, panel_key, panel_value)
    try:
        pending = session_state.get("_dashboard_query_updates", {})
        if not isinstance(pending, dict):
            pending = {}
        pending["run_detail"] = run_name
        pending["model_section"] = section
        session_state["_dashboard_query_updates"] = pending
    except Exception:
        pass


@contextmanager
def _dropdown_section(st, title: str, *, expanded: bool = False):
    st.markdown(f"#### {title}")
    with st.container():
        yield


def render_model_workflow_view(st, state: dict[str, object], *, ns: ModuleType) -> None:
    METRIC_FIELDS = ns.METRIC_FIELDS
    RUN_ACTION_INFO = ns.RUN_ACTION_INFO
    build_button_readiness = ns.build_button_readiness
    _cluster_profile_warning = ns._cluster_profile_warning
    _frame_empty = ns._frame_empty
    _frame_records = ns._frame_records
    _friendly_remote_sync = ns._friendly_remote_sync
    _persisted_choice = ns._persisted_choice
    _quote_path = ns._quote_path
    _render_action_history = ns._render_action_history
    _render_cluster_health_notice = ns._render_cluster_health_notice
    _render_decision_workspace = ns._render_decision_workspace
    _base_render_draft_action = ns._render_draft_action
    _render_latest_preflight_summary = ns._render_latest_preflight_summary
    _base_render_launch_action = ns._render_launch_action
    _render_metric_cards = ns._render_metric_cards
    _render_recommended_card = ns._render_recommended_card
    _render_remote_console = ns._render_remote_console
    _render_run_workflow_macros = ns._render_run_workflow_macros
    _run_next_step_copy = ns._run_next_step_copy
    _wide_chart = ns._wide_chart
    discovery_frame = ns.discovery_frame
    draft_supek_cancel_action = ns.draft_supek_cancel_action
    draft_supek_pull_artifacts_action = ns.draft_supek_pull_artifacts_action
    draft_supek_submit_action = ns.draft_supek_submit_action
    draft_supek_sync_repo_action = ns.draft_supek_sync_repo_action
    draft_supek_sync_run_action = ns.draft_supek_sync_run_action
    draft_supek_verify_action = ns.draft_supek_verify_action
    get_cluster_profile = ns.get_cluster_profile
    queue_supek_fetch_logs_action = ns.queue_supek_fetch_logs_action
    queue_supek_poll_action = ns.queue_supek_poll_action
    queue_supek_preflight_action = ns.queue_supek_preflight_action
    replay_curve_frame = ns.replay_curve_frame
    run_metric_history = ns.run_metric_history
    submit_continue_feedback_action = ns.submit_continue_feedback_action
    submit_freeze_final_action = ns.submit_freeze_final_action
    submit_ingest_round_action = ns.submit_ingest_round_action
    submit_phase3_ingest_action = ns.submit_phase3_ingest_action
    submit_phase3_make_ingest_action = ns.submit_phase3_make_ingest_action
    submit_run_workflow_action = ns.submit_run_workflow_action

    runs = list(state.get("runs", []))
    if not runs:
        st.header("Model Workflow")
        st.info("No runs are visible in this workspace view yet.")
        return

    phase3_mode = any(str(run.get("phase", "")) == "phase3_real_al" for run in runs)
    run_options = [str(run.get("run_display_name", run.get("run_name", ""))) for run in runs]
    st.caption("Navigation updates immediately so the selected Model Workflow section always matches the content below.")
    nav_cols = st.columns([1.2, 1.4])
    with nav_cols[0]:
        selected_name = _persisted_choice(
            st,
            st.selectbox,
            label="Choose Phase 3 branch" if phase3_mode else "Choose thesis run",
            options=run_options,
            key="dashboard_run_detail_name",
            query_key="run_detail",
            default=run_options[0],
            write_query=False,
        )
    guided_mode = str(state.get("workflow_mode", "Expert mode")) == "Guided thesis mode"
    if guided_mode:
        requested_advanced_section = ""
        session_state = getattr(st, "session_state", {})
        stored_section = str(session_state.get("dashboard_model_section", "")) if isinstance(session_state, dict) else ""
        radio_values = getattr(st, "_radio_values", {})
        radio_section = str(radio_values.get("dashboard_model_section", "")) if isinstance(radio_values, dict) else ""
        for candidate in (radio_section, stored_section):
            if candidate in ADVANCED_MODEL_SECTIONS:
                requested_advanced_section = candidate
                break
        if requested_advanced_section and isinstance(session_state, dict):
            session_state["dashboard_model_section"] = "Advanced / debug tools"
            session_state["dashboard_model_advanced_section"] = requested_advanced_section
    section_options = GUIDED_MODEL_SECTIONS if guided_mode else MODEL_SECTIONS
    with nav_cols[1]:
        selected_section = _persisted_choice(
            st,
            st.radio,
            label="Model workflow section",
            options=section_options,
            key="dashboard_model_section",
            query_key="model_section",
            default=section_options[0],
            write_query=False,
        )
    if selected_section == "Advanced / debug tools":
        requested_advanced_section = ""
        if guided_mode:
            session_state = getattr(st, "session_state", {})
            requested_advanced_section = (
                str(session_state.get("dashboard_model_advanced_section", ""))
                if isinstance(session_state, dict)
                else ""
            )
        selected_section = st.selectbox(
            "Advanced Model Workflow section",
            ADVANCED_MODEL_SECTIONS,
            index=ADVANCED_MODEL_SECTIONS.index(requested_advanced_section)
            if requested_advanced_section in ADVANCED_MODEL_SECTIONS
            else 0,
            key="dashboard_model_advanced_section",
        )
        st.info(
            "Advanced sections are still here for debugging and audit history, but Guided mode keeps normal execution on the main workflow sections."
        )
    run = next(item for item in runs if str(item.get("run_display_name", item.get("run_name", ""))) == selected_name)

    st.header(f"Model Workflow: {run['run_display_name']}")
    st.caption(run.get("run_identity", ""))
    _render_recommended_card(st, **_run_next_step_copy(run))
    feedback_queue = run.get("feedback_queue", {}) if isinstance(run.get("feedback_queue", {}), dict) else {}
    run_display_name = str(run.get("run_display_name", run.get("run_name", "")))

    def preserve_location(section: str, *, panel_key: str = "", panel_value: str = ""):
        return lambda _action: _remember_model_workflow_location(
            st,
            run_name=run_display_name,
            section=section,
            panel_key=panel_key,
            panel_value=panel_value,
        )

    def _render_launch_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "Model Workflow")
        kwargs.setdefault("section", selected_section)
        return _base_render_launch_action(*args, **kwargs)

    def _render_draft_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "Model Workflow")
        kwargs.setdefault("section", selected_section)
        return _base_render_draft_action(*args, **kwargs)

    with _dropdown_section(st, "Run snapshot", expanded=True):
        metric_cards = [
            ("Model state", run.get("ml_status_label", run["ml_status"])),
            ("Latest round", run.get("latest_round_id", 0)),
            ("Suggested peptides", len((run.get("latest_batch") or {}).get("rows", []))),
            ("Returned labels", len(run.get("import_rows", []))),
            ("Remote state", _friendly_remote_sync(str(run.get("remote_sync_status", "not_synced")))),
        ]
        if str(run.get("phase", "")) == "phase3_real_al":
            round_status = run.get("phase3_round_status", {}) if isinstance(run.get("phase3_round_status", {}), dict) else {}
            continuation = run.get("phase3_continuation_status", {}) if isinstance(run.get("phase3_continuation_status", {}), dict) else {}
            metric_cards.insert(1, ("Current round", continuation.get("current_round", round_status.get("round_id", "round_001"))))
            metric_cards.insert(2, ("Proposal", round_status.get("status", "missing")))
            ingest_status = run.get("phase3_ingest_status", {}) if isinstance(run.get("phase3_ingest_status", {}), dict) else {}
            metric_cards.insert(3, ("Ingest", continuation.get("latest_ingest_status", ingest_status.get("ingest_status", "missing"))))
        _render_metric_cards(st, metric_cards)
        st.caption(f"Run directory: {run['run_dir']}")
        st.caption(f"Last modified: {run['last_modified']}")

    if selected_section == "Workflow summary":
        st.subheader("Workflow summary")
        metric_frame = run_metric_history(run)
        metric_rows = _frame_records(metric_frame)
        metric_comparison: list[dict[str, object]] = []
        for metric_name in METRIC_FIELDS:
            baseline_value = run.get("baseline_metrics", {}).get(metric_name, "")
            final_value = run.get("final_metrics", {}).get(metric_name, "")
            if baseline_value in {"", None} and final_value in {"", None}:
                continue
            metric_comparison.append(
                {
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "final_holdout": final_value,
                }
            )
        latest_batch = run.get("latest_batch") or {}
        replay = replay_curve_frame(run)
        discovery = discovery_frame(run)
        summary_panels = ["Guided workflow runner", "Workflow status"]
        if metric_rows or metric_comparison:
            summary_panels.append("Baseline and final metrics")
        if latest_batch:
            summary_panels.append("Suggested peptides from the latest batch")
        if str(feedback_queue.get("pending_round_id", "")).strip():
            summary_panels.append("Feedback queue")
        if run.get("import_rows"):
            summary_panels.append("Labels already fed back into the model")
        if not _frame_empty(replay):
            summary_panels.append("Replay benchmark analytics")
        if not _frame_empty(discovery):
            summary_panels.append("Discovery summaries")
        summary_panel = _rich_panel_choice(
            st,
            key=f"model_summary_panel_{run['run_slug']}",
            options=summary_panels,
            default="Guided workflow runner",
        )
        if _show_panel(summary_panel, "Guided workflow runner"):
            with _dropdown_section(st, "Guided workflow runner", expanded=True):
                _render_run_workflow_macros(st, state, run)
        if _show_panel(summary_panel, "Workflow status"):
            with _dropdown_section(st, "Workflow status", expanded=True):
                st.write(run.get("ml_status_summary", ""))
        if (metric_rows or metric_comparison) and _show_panel(summary_panel, "Baseline and final metrics"):
            with _dropdown_section(st, "Baseline and final metrics", expanded=True):
                if metric_comparison:
                    st.dataframe(metric_comparison)
                for metric_name in [metric for metric in METRIC_FIELDS if any(row.get(metric) not in {None, ""} for row in metric_rows)]:
                    chart = _wide_chart(metric_rows, index_key="round_id", series_key="stage", value_key=metric_name)
                    if chart:
                        st.line_chart(chart)

        if latest_batch and _show_panel(summary_panel, "Suggested peptides from the latest batch"):
            with _dropdown_section(st, "Suggested peptides from the latest batch", expanded=False):
                st.dataframe(latest_batch.get("rows", []))
        if str(feedback_queue.get("pending_round_id", "")).strip() and _show_panel(summary_panel, "Feedback queue"):
            with _dropdown_section(st, "Feedback queue", expanded=True):
                st.caption("Use this to see whether the whole pending proposed batch is ready to re-enter the active-learning loop, not just whether one peptide has a local ingest CSV.")
                st.dataframe(
                    [
                        {
                            "pending_round": str(feedback_queue.get("pending_round_id", "")) or "-",
                            "proposed_peptides": len(list(feedback_queue.get("pending_sequences", []))),
                            "ready_now": int(feedback_queue.get("ready_count", 0) or 0),
                            "blocked": int(feedback_queue.get("blocked_count", 0) or 0),
                            "state": (
                                "Ready to continue AL"
                                if bool(feedback_queue.get("can_continue", False))
                                else "Waiting on more MD feedback"
                            ),
                            "next_move": (
                                "Run Continue AL from reviewed peptides"
                                if bool(feedback_queue.get("can_continue", False))
                                else str(feedback_queue.get("summary", "")) or "Finish the missing review / promotion / full-analysis work."
                            ),
                        }
                    ]
                )
                blocked_rows = list(feedback_queue.get("blocked_rows", []))
                if blocked_rows:
                    st.markdown("##### Feedback blockers")
                    st.dataframe(blocked_rows)
        if run.get("import_rows") and _show_panel(summary_panel, "Labels already fed back into the model"):
            with _dropdown_section(st, "Labels already fed back into the model", expanded=False):
                st.dataframe(run.get("import_rows", []))
        if not _frame_empty(replay) and _show_panel(summary_panel, "Replay benchmark analytics"):
            with _dropdown_section(st, "Replay benchmark analytics", expanded=False):
                st.dataframe(replay)
        if not _frame_empty(discovery) and _show_panel(summary_panel, "Discovery summaries"):
            with _dropdown_section(st, "Discovery summaries", expanded=False):
                st.dataframe(discovery)
        return

    if selected_section == "Local model actions":
        st.subheader("Local model actions")
        if run["ml_status"] == "config-only":
            st.info("This run is still config-only. The first thesis action is usually the replay benchmark on the initial dataset.")
        ingest_candidates = list(run.get("available_ingest_csvs", []))
        local_panels = ["Guided workflow runner"]
        is_phase3_branch = str(run.get("phase", "")) == "phase3_real_al" and str(run.get("branch_strategy", "")).strip()
        if is_phase3_branch:
            local_panels.append("Phase 3 branch ingest")
        if str(feedback_queue.get("pending_round_id", "")).strip():
            local_panels.append("Closed-loop feedback runner")
        local_panels.extend(["Core local model actions", "Reviewed ingest CSV"])
        local_panel = _rich_panel_choice(
            st,
            key=f"model_local_panel_{run['run_slug']}",
            options=local_panels,
            default="Guided workflow runner",
        )
        if _show_panel(local_panel, "Guided workflow runner"):
            with _dropdown_section(st, "Guided workflow runner", expanded=True):
                _render_run_workflow_macros(st, state, run)
        if str(feedback_queue.get("pending_round_id", "")).strip() and _show_panel(local_panel, "Closed-loop feedback runner"):
            with _dropdown_section(st, "Closed-loop feedback runner", expanded=True):
                if bool(feedback_queue.get("can_continue", False)):
                    feedback_info = RUN_ACTION_INFO["continue-feedback"]
                    feedback_command = (
                        "python -m active_learning_thesis dashboard-continue-feedback "
                        f"--run-dir {_quote_path(run['run_dir'])}"
                    )
                    _render_launch_action(
                        st,
                        label=feedback_info["label"],
                        command=feedback_command,
                        key_prefix=f"{run['run_slug']}_continue_feedback",
                        button_text="Run locally",
                        what=feedback_info["what"],
                        when=feedback_info["when"],
                        produces=feedback_info["produces"],
                        next_step=feedback_info["next"],
                        contract_id="continue-al-feedback",
                        readiness=build_button_readiness(state, "continue-al-feedback", run=run),
                        after_submit=preserve_location(
                            "Local model actions",
                            panel_key=f"model_local_panel_{run['run_slug']}",
                            panel_value=local_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])): submit_continue_feedback_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            propose_next_batch=False,
                        ),
                    )
                    _render_launch_action(
                        st,
                        label="Continue AL and immediately propose the next batch",
                        command=feedback_command + " --propose-next-batch",
                        key_prefix=f"{run['run_slug']}_continue_feedback_propose",
                        button_text="Run extended loop",
                        what="Close the current MD feedback loop, retrain on the reviewed labels, then immediately export the next proposed batch from the updated model.",
                        when="Use this when you already know you want the next AL round right after ingest, without stopping in between for another manual check.",
                        produces="An ingest/retrain cycle plus a fresh `round_XXX_batch.csv` from the updated model state.",
                        next_step="The next proposed peptides will appear in Peptides and can move straight into candidate selection / MD preparation.",
                        contract_id="continue-al-feedback",
                        readiness=build_button_readiness(state, "continue-al-feedback", run=run),
                        after_submit=preserve_location(
                            "Local model actions",
                            panel_key=f"model_local_panel_{run['run_slug']}",
                            panel_value=local_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])): submit_continue_feedback_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            propose_next_batch=True,
                        ),
                    )
                else:
                    st.info(str(feedback_queue.get("summary", "")) or "The pending proposed batch is not fully ready for model feedback yet.")
                    blocked_rows = list(feedback_queue.get("blocked_rows", []))
                    if blocked_rows:
                        st.dataframe(blocked_rows)
        if is_phase3_branch and _show_panel(local_panel, "Phase 3 branch ingest"):
            with _dropdown_section(st, "Phase 3 branch ingest", expanded=True):
                branch = str(run.get("branch_strategy", ""))
                output_root = Path(str(state["run_root"]))
                ingest_status = run.get("phase3_ingest_status", {}) if isinstance(run.get("phase3_ingest_status", {}), dict) else {}
                continuation = run.get("phase3_continuation_status", {}) if isinstance(run.get("phase3_continuation_status", {}), dict) else {}
                round_number = int(continuation.get("next_round_number", 1) or 1)
                st.dataframe(
                    [
                        {
                            "branch": branch,
                            "round": continuation.get("current_round", ingest_status.get("round_id", "round_001")),
                            "proposal_status": continuation.get("latest_proposal_status", ingest_status.get("proposal_status", "")),
                            "md_returned": ingest_status.get("md_returned_count", 0),
                            "reviewed": ingest_status.get("reviewed_count", 0),
                            "ingest_ready": ingest_status.get("ingest_ready_count", 0),
                            "ingested": ingest_status.get("ingested_count", 0),
                            "blocked_rows": ingest_status.get("blocked_rows_count", 0),
                            "labeled_count": continuation.get("labeled_count", ""),
                            "cgmd_labels": continuation.get("acquired_cgmd_labels_count", ""),
                            "next_action": continuation.get("next_action", ingest_status.get("next_action", "")),
                            "blocked_reason": continuation.get("blocked_reason", ""),
                        }
                    ]
                )
                make_command = (
                    "python -m active_learning_thesis phase3-real-al make-ingest-csv "
                    f"--output-root {_quote_path(output_root)} --branch {branch} --round {round_number}"
                )
                _render_launch_action(
                    st,
                    label="Create Phase 3 ingest CSV",
                    command=make_command,
                    key_prefix=f"{run['run_slug']}_phase3_make_ingest",
                    button_text="Create ingest CSV",
                    what="Build a branch-local ingest CSV from reviewed MD evidence for this Phase 3 branch.",
                    when="Use after the selected peptides have human-reviewed CG-MD labels.",
                    produces="`rounds/round_001/ingest/cgmd_ingest.csv` plus blocker/status JSON files.",
                    next_step="Inspect the CSV and blockers, then ingest labels into this branch only.",
                    contract_id="phase3-make-ingest-csv",
                    action_kind="phase3-make-ingest-csv",
                    after_submit=preserve_location(
                        "Local model actions",
                        panel_key=f"model_local_panel_{run['run_slug']}",
                        panel_value=local_panel,
                    ),
                    on_submit=lambda output_root=output_root, branch=branch, round_number=round_number: submit_phase3_make_ingest_action(
                        run_root=Path(str(state["run_root"])),
                        output_root=output_root,
                        branch=branch,
                        round_id=round_number,
                    ),
                )
                ingest_csv = Path(str(ingest_status.get("ingest_csv", output_root / "branches" / branch / "rounds" / f"round_{round_number:03d}" / "ingest" / "cgmd_ingest.csv")))
                ingest_command = (
                    "python -m active_learning_thesis phase3-real-al ingest "
                    f"--output-root {_quote_path(output_root)} --branch {branch} --round {round_number} "
                    f"--import-csv {_quote_path(ingest_csv)}"
                )
                _render_launch_action(
                    st,
                    label="Ingest Phase 3 labels into selected branch",
                    command=ingest_command,
                    key_prefix=f"{run['run_slug']}_phase3_ingest",
                    button_text="Ingest labels",
                    what="Update only this branch ledger with labels from the branch-local ingest CSV.",
                    when="Use after the ingest CSV has been reviewed and contains the intended labels.",
                    produces="Updated branch ledger, current labeled ledger, ingest log, and shared provenance inventory events.",
                    next_step="Confirm no other branch ledger changed before starting any future round.",
                    contract_id="phase3-ingest",
                    action_kind="phase3-ingest",
                    after_submit=preserve_location(
                        "Local model actions",
                        panel_key=f"model_local_panel_{run['run_slug']}",
                        panel_value=local_panel,
                    ),
                    on_submit=lambda output_root=output_root, branch=branch, round_number=round_number, ingest_csv=ingest_csv: submit_phase3_ingest_action(
                        run_root=Path(str(state["run_root"])),
                        output_root=output_root,
                        branch=branch,
                        round_id=round_number,
                        import_csv=ingest_csv,
                    ),
                )
        if _show_panel(local_panel, "Core local model actions"):
            with _dropdown_section(st, "Core local model actions", expanded=True):
                for command_name in ("run-replay", "propose-round", "run-discovery", "evaluate-final"):
                    info = RUN_ACTION_INFO[command_name]
                    command = f"python -m active_learning_thesis {command_name} --run-dir {_quote_path(run['run_dir'])}"
                    contract_id = command_name if command_name != "run-replay" else ""
                    readiness = build_button_readiness(state, contract_id, run=run) if contract_id else None
                    _render_launch_action(
                        st,
                        label=info["label"],
                        command=command,
                        key_prefix=f"{run['run_slug']}_{command_name}",
                        button_text="Run locally",
                        what=info["what"],
                        when=info["when"],
                        produces=info["produces"],
                        next_step=info["next"],
                        contract_id=contract_id,
                        action_kind=contract_id or command_name,
                        readiness=readiness,
                        after_submit=preserve_location(
                            "Local model actions",
                            panel_key=f"model_local_panel_{run['run_slug']}",
                            panel_value=local_panel,
                        ),
                        on_submit=lambda command_name=command_name, run_dir=Path(str(run["run_dir"])): submit_run_workflow_action(
                            run_root=Path(str(state["run_root"])),
                            command_name=command_name,
                            run_dir=run_dir,
                        ),
                    )

        if _show_panel(local_panel, "Reviewed ingest CSV"):
            with _dropdown_section(st, "Reviewed ingest CSV", expanded=bool(ingest_candidates)):
                if ingest_candidates:
                    with _form_container(st, key=f"ingest_csv_form_{run['run_slug']}"):
                        selected_ingest_csv = st.selectbox(
                            "Reviewed ingest CSV",
                            ingest_candidates,
                            index=0,
                            key=f"ingest_csv_{run['run_slug']}",
                        )
                        _form_submit_button(st, "Update ingest CSV", key=f"ingest_csv_update_{run['run_slug']}")
                    ingest_info = RUN_ACTION_INFO["ingest-round"]
                    ingest_command = (
                        "python -m active_learning_thesis ingest-round "
                        f"--run-dir {_quote_path(run['run_dir'])} "
                        f"--import-csv {_quote_path(selected_ingest_csv)}"
                    )
                    _render_launch_action(
                        st,
                        label=ingest_info["label"],
                        command=ingest_command,
                        key_prefix=f"{run['run_slug']}_ingest-round",
                        button_text="Run locally",
                        what=ingest_info["what"],
                        when=ingest_info["when"],
                        produces=ingest_info["produces"],
                        next_step=ingest_info["next"],
                        contract_id="ingest-round",
                        readiness=build_button_readiness(state, "ingest-round", run=run),
                        after_submit=preserve_location(
                            "Local model actions",
                            panel_key=f"model_local_panel_{run['run_slug']}",
                            panel_value=local_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])), import_csv=Path(selected_ingest_csv): submit_ingest_round_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            import_csv=import_csv,
                        ),
                    )
                else:
                    st.info("Ingest returned labels becomes available after a reviewed `cgmd_ingest.csv` exists for this run.")
        return

    if selected_section == "Thesis freeze":
        st.subheader("Thesis freeze")
        st.caption("Use this section when you are done experimenting with this run and want a reproducible thesis handoff. It writes a freeze bundle and model card; it does not delete or rewrite the active run unless Force is explicitly enabled.")
        freeze_info = RUN_ACTION_INFO["freeze-final"]
        freeze_path = Path(str(run["run_dir"])) / "final_freeze" / "final_freeze.json"
        final_metrics = run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {}
        if freeze_path.exists():
            st.success(f"Final freeze already exists: {freeze_path}")
        elif final_metrics:
            st.info("Final holdout metrics are present, so this run can be frozen now.")
        else:
            st.warning("Final holdout metrics are not visible yet. Keep the evaluation checkbox enabled if you want the freeze action to create them first.")
        with _form_container(st, key=f"freeze_options_form_{run['run_slug']}"):
            st.caption("Adjust freeze options, then press Update freeze preview before launching the freeze action.")
            freeze_metric = st.selectbox(
                "Freeze metric",
                METRIC_FIELDS,
                index=METRIC_FIELDS.index("f1") if "f1" in METRIC_FIELDS else 0,
                key=f"freeze_metric_{run['run_slug']}",
            )
            run_evaluation = st.checkbox(
                "Run final evaluation first if needed",
                value=not bool(final_metrics),
                key=f"freeze_run_eval_{run['run_slug']}",
            )
            force_freeze = st.checkbox(
                "Replace existing freeze bundle",
                value=False,
                key=f"freeze_force_{run['run_slug']}",
            )
            allow_unresolved = st.checkbox(
                "Allow unresolved checks in the freeze report",
                value=False,
                key=f"freeze_allow_unresolved_{run['run_slug']}",
            )
            _form_submit_button(st, "Update freeze preview", key=f"freeze_update_{run['run_slug']}")
        freeze_command = (
            "python -m active_learning_thesis freeze-final "
            f"--run-dir {_quote_path(run['run_dir'])} "
            f"--metric {freeze_metric}"
        )
        if run_evaluation:
            freeze_command += " --run-evaluation"
        if force_freeze:
            freeze_command += " --force"
        if allow_unresolved:
            freeze_command += " --allow-unresolved"
        freeze_readiness = build_button_readiness(state, "freeze-final", run=run)
        _render_launch_action(
            st,
            label=freeze_info["label"],
            command=freeze_command,
            key_prefix=f"{run['run_slug']}_freeze-final",
            button_text="Freeze final result",
            what=freeze_info["what"],
            when=freeze_info["when"],
            produces=freeze_info["produces"],
            next_step=freeze_info["next"],
            contract_id="freeze-final",
            readiness=freeze_readiness,
            after_submit=preserve_location("Thesis freeze"),
            on_submit=lambda run_dir=Path(str(run["run_dir"])), freeze_metric=freeze_metric, run_evaluation=run_evaluation, force_freeze=force_freeze, allow_unresolved=allow_unresolved: submit_freeze_final_action(
                run_root=Path(str(state["run_root"])),
                run_dir=run_dir,
                metric=str(freeze_metric),
                run_evaluation=bool(run_evaluation),
                force=bool(force_freeze),
                allow_unresolved=bool(allow_unresolved),
            ),
        )
        return

    if selected_section == "Remote SUPEK":
        st.subheader("Remote SUPEK actions")
        st.info("Recommended order: verify environment -> sync repo -> upload run state -> submit remote workflow -> poll queue -> pull artifacts.")
        supek_profile = get_cluster_profile(state.get("profiles", {}), "supek")
        if supek_profile is None:
            st.warning(_cluster_profile_warning(state, "supek"))
            return
        remote_sync_status = str(run.get("remote_sync_status", "not_synced"))
        branch_default = supek_profile.get("default_branch", "codex/active-learning-thesis")
        remote_panel = _rich_panel_choice(
            st,
            key=f"model_supek_panel_{run['run_slug']}",
            options=[
                "Environment and sync",
                "Remote model workflows",
                "Monitor and pull results back",
                "Logs and job control",
                "Latest SUPEK console snapshot",
            ],
            default="Environment and sync",
        )
        if _show_panel(remote_panel, "Environment and sync"):
            with _dropdown_section(st, "Environment and sync", expanded=True):
                _render_cluster_health_notice(st, state, "supek")
                with _form_container(st, key=f"supek_branch_form_{run['run_slug']}"):
                    branch_name = st.text_input("SUPEK branch", value=branch_default, key=f"supek_branch_{run['run_slug']}")
                    _form_submit_button(st, "Update SUPEK branch", key=f"supek_branch_update_{run['run_slug']}")
                _render_launch_action(
                    st,
                    label="Verify SUPEK environment",
                    command=f"ssh {supek_profile['username']}@{supek_profile['host']} <verify allowlisted command>",
                    key_prefix=f"supek_verify_{run['run_slug']}",
                    button_text="Run remote check",
                    what="Confirm that the remote repo path, conda initialization script, and environment name all exist on SUPEK.",
                    when="Use this before the first remote SUPEK action or after changing the profile.",
                    produces="A read-only connectivity / environment result stored in recent actions.",
                    next_step="You can safely draft a repo sync or run-state upload once this check succeeds.",
                    action_kind="supek-verify-env",
                    after_submit=preserve_location(
                        "Remote SUPEK",
                        panel_key=f"model_supek_panel_{run['run_slug']}",
                        panel_value=remote_panel,
                    ),
                    on_submit=lambda run_dir=Path(str(run["run_dir"])): draft_supek_verify_action(
                        run_root=Path(str(state["run_root"])),
                        run_dir=run_dir,
                        profile=supek_profile,
                    ),
                )
                _render_draft_action(
                    st,
                    label="Sync repo on SUPEK",
                    command=f"git fetch/pull remote branch {branch_name} on SUPEK",
                    key_prefix=f"supek_sync_repo_{run['run_slug']}",
                    what="Update the remote SUPEK clone to the selected Git branch before launching jobs there.",
                    when="Use this when your desired code is already committed and pushed to the branch you want SUPEK to run.",
                    produces="A synchronized remote repo checkout on SUPEK.",
                    next_step="The run directory upload and remote workflow submissions become trustworthy after the repo is in sync.",
                    action_kind="supek-sync-repo",
                    after_submit=preserve_location(
                        "Remote SUPEK",
                        panel_key=f"model_supek_panel_{run['run_slug']}",
                        panel_value=remote_panel,
                    ),
                    on_submit=lambda run_dir=Path(str(run["run_dir"])), branch_name=branch_name: draft_supek_sync_repo_action(
                        run_root=Path(str(state["run_root"])),
                        run_dir=run_dir,
                        profile=supek_profile,
                        branch=branch_name,
                    ),
                )
                _render_draft_action(
                    st,
                    label="Upload run state to SUPEK",
                    command=f"scp -r {_quote_path(run['run_dir'])} {supek_profile['username']}@{supek_profile['host']}:<scratch>",
                    key_prefix=f"supek_sync_run_{run['run_slug']}",
                    what="Copy the filtered run directory state that SUPEK needs in order to execute model workflows remotely.",
                    when="Use this before the first remote SUPEK workflow submission and whenever the local run state has materially changed.",
                    produces="A staged remote run directory under the configured SUPEK scratch area.",
                    next_step="Remote workflow submission buttons become available after the run state is staged.",
                    contract_id="supek-sync-run",
                    readiness=build_button_readiness(state, "supek-sync-run", run=run),
                    after_submit=preserve_location(
                        "Remote SUPEK",
                        panel_key=f"model_supek_panel_{run['run_slug']}",
                        panel_value=remote_panel,
                    ),
                    on_submit=lambda run_dir=Path(str(run["run_dir"])): draft_supek_sync_run_action(
                        run_root=Path(str(state["run_root"])),
                        run_dir=run_dir,
                        profile=supek_profile,
                    ),
                )
                if remote_sync_status != "not_synced":
                    _render_launch_action(
                        st,
                        label="Run SUPEK submit preflight",
                        command="ssh <supek> <check repo/env/scratch/staged run state>",
                        key_prefix=f"supek_preflight_{run['run_slug']}",
                        button_text="Run remote check",
                        what="Verify the remote repo, conda init, scratch root, and staged run directory before you submit a SUPEK workflow.",
                        when="Use this right before remote workflow submission, especially after changing the branch, scratch path, or staged run state.",
                        produces="A read-only readiness result that tells you whether the current SUPEK submit path is sane.",
                        next_step="If it passes, you can submit the chosen SUPEK workflow with more confidence.",
                        action_kind="supek-preflight",
                        after_submit=preserve_location(
                            "Remote SUPEK",
                            panel_key=f"model_supek_panel_{run['run_slug']}",
                            panel_value=remote_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])), remote_sync_status=remote_sync_status: queue_supek_preflight_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            profile=supek_profile,
                            require_staged_run=remote_sync_status != "not_synced",
                        ),
                    )
                    _render_latest_preflight_summary(
                        st,
                        title="Latest SUPEK submit preflight",
                        actions=list(run.get("dashboard_actions", [])),
                        kind="supek-submit-preflight",
                    )

        if _show_panel(remote_panel, "Remote model workflows"):
            with _dropdown_section(st, "Remote model workflows", expanded=True):
                if remote_sync_status != "not_synced":
                    for command_name in ("run-replay", "propose-round", "run-discovery", "evaluate-final"):
                        info = RUN_ACTION_INFO[command_name]
                        _render_draft_action(
                            st,
                            label=f"Submit {info['label']} on SUPEK",
                            command=f"Submit {command_name} via qsub on SUPEK",
                            key_prefix=f"supek_submit_{run['run_slug']}_{command_name}",
                            what=info["what"],
                            when="Use this when you want the selected model workflow to run on SUPEK instead of your local machine.",
                            produces="A PBS wrapper plus a tracked remote job id if the submission succeeds.",
                            next_step="Queue polling and remote artifact pull become the next monitoring actions.",
                            action_kind="supek-submit-workflow",
                            after_submit=preserve_location(
                                "Remote SUPEK",
                                panel_key=f"model_supek_panel_{run['run_slug']}",
                                panel_value=remote_panel,
                            ),
                            on_submit=lambda command_name=command_name, run_dir=Path(str(run["run_dir"])): draft_supek_submit_action(
                                run_root=Path(str(state["run_root"])),
                                run_dir=run_dir,
                                profile=supek_profile,
                                command_name=command_name,
                            ),
                        )
                else:
                    st.info("Remote workflow submission stays blocked until the run state has been uploaded to SUPEK.")

        if _show_panel(remote_panel, "Monitor and pull results back"):
            with _dropdown_section(st, "Monitor and pull results back", expanded=True):
                if run.get("remote_job_id") or remote_sync_status in {"submitted", "running"}:
                    _render_launch_action(
                        st,
                        label="Poll SUPEK queue",
                        command=f"ssh {supek_profile['username']}@{supek_profile['host']} qstat -u {supek_profile['username']}",
                        key_prefix=f"supek_poll_{run['run_slug']}",
                        button_text="Poll remote queue",
                        what="Refresh the tracked PBS queue state for this run's active SUPEK job.",
                        when="Use this after a remote SUPEK workflow has been submitted.",
                        produces="An updated remote job snapshot and sync status in the dashboard.",
                        next_step="Once the job finishes, you can pull the artifacts back into the dashboard staging area.",
                        contract_id="supek-poll-qstat",
                        readiness=build_button_readiness(state, "supek-poll-qstat", run=run),
                        after_submit=preserve_location(
                            "Remote SUPEK",
                            panel_key=f"model_supek_panel_{run['run_slug']}",
                            panel_value=remote_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])): queue_supek_poll_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            profile=supek_profile,
                            remote_job_id=str(run.get("remote_job_id", "")),
                        ),
                    )
                else:
                    st.info("Queue polling becomes available after a tracked remote SUPEK job is submitted.")
                if remote_sync_status in {"submitted", "running", "outputs_staged", "outputs_returned"}:
                    _render_draft_action(
                        st,
                        label="Pull SUPEK artifacts back",
                        command="scp -r <remote run dir> <dashboard staging dir>",
                        key_prefix=f"supek_pull_{run['run_slug']}",
                        what="Copy the remote SUPEK outputs back into the dashboard's safe staging area on your local machine.",
                        when="Use this after the remote job has produced outputs you want to inspect or merge locally.",
                        produces="A staged local download under `_dashboard_remote_state/downloads/...`.",
                        next_step="You can inspect the staged files and merge only the artifacts you actually want to keep.",
                        action_kind="supek-pull-artifacts",
                        after_submit=preserve_location(
                            "Remote SUPEK",
                            panel_key=f"model_supek_panel_{run['run_slug']}",
                            panel_value=remote_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])): draft_supek_pull_artifacts_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            profile=supek_profile,
                        ),
                    )
                else:
                    st.info("Artifact pull becomes available after a remote SUPEK workflow has been submitted.")

        if _show_panel(remote_panel, "Logs and job control"):
            with _dropdown_section(st, "Logs and job control", expanded=False):
                remote_reference_rows = [
                    {"artifact": "Tracked job", "path": str(run.get("remote_job_id", "")) or "-", "why_it_matters": "Use this job id when polling or cancelling the remote SUPEK workflow."},
                    {"artifact": "PBS wrapper", "path": str(run.get("remote_wrapper", "")) or "-", "why_it_matters": "This is the exact wrapper script SUPEK submitted to PBS."},
                    {"artifact": "stdout", "path": str(run.get("remote_stdout", "")) or "-", "why_it_matters": "Primary runtime output for the latest tracked SUPEK workflow."},
                    {"artifact": "stderr", "path": str(run.get("remote_stderr", "")) or "-", "why_it_matters": "Primary failure/debug output for the latest tracked SUPEK workflow."},
                ]
                st.dataframe(remote_reference_rows)
                if run.get("remote_job_id") or any(str(run.get(key, "")).strip() for key in ("remote_stdout", "remote_stderr", "remote_wrapper")):
                    _render_launch_action(
                        st,
                        label="Fetch latest SUPEK logs",
                        command="ssh <supek> 'tail remote stdout/stderr/wrapper'",
                        key_prefix=f"supek_logs_{run['run_slug']}",
                        button_text="Fetch remote logs",
                        what="Read the latest remote wrapper/stdout/stderr directly from SUPEK without downloading the full run directory.",
                        when="Use this when you want to inspect a live or recently finished remote job before pulling all artifacts back.",
                        produces="A read-only dashboard action whose stdout contains the latest remote log excerpts.",
                        next_step="Use the excerpts to decide whether to poll again, cancel the job, or pull artifacts back.",
                        action_kind="supek-fetch-logs",
                        after_submit=preserve_location(
                            "Remote SUPEK",
                            panel_key=f"model_supek_panel_{run['run_slug']}",
                            panel_value=remote_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])): queue_supek_fetch_logs_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            profile=supek_profile,
                            remote_stdout=str(run.get("remote_stdout", "")),
                            remote_stderr=str(run.get("remote_stderr", "")),
                            remote_wrapper=str(run.get("remote_wrapper", "")),
                        ),
                    )
                else:
                    st.info("Remote log fetch becomes useful once the latest SUPEK submission has a wrapper or tracked stdout/stderr paths.")
                if run.get("remote_job_id") and remote_sync_status in {"submitted", "running"}:
                    _render_draft_action(
                        st,
                        label="Cancel tracked SUPEK job",
                        command=f"ssh {supek_profile['username']}@{supek_profile['host']} qdel {run.get('remote_job_id', '')}",
                        key_prefix=f"supek_cancel_{run['run_slug']}",
                        what="Request scheduler-side cancellation of the currently tracked SUPEK PBS job.",
                        when="Use this only when the job is clearly stuck, misconfigured, or no longer the thesis action you want to run.",
                        produces="A scheduler cancellation request plus a reset back to a staged remote run state if the cancellation succeeds.",
                        next_step="You can inspect logs, adjust the run state, then submit a fresh remote workflow if needed.",
                        action_kind="supek-cancel-job",
                        after_submit=preserve_location(
                            "Remote SUPEK",
                            panel_key=f"model_supek_panel_{run['run_slug']}",
                            panel_value=remote_panel,
                        ),
                        on_submit=lambda run_dir=Path(str(run["run_dir"])), remote_job_id=str(run.get("remote_job_id", "")): draft_supek_cancel_action(
                            run_root=Path(str(state["run_root"])),
                            run_dir=run_dir,
                            profile=supek_profile,
                            remote_job_id=remote_job_id,
                        ),
                    )
                else:
                    st.info("Remote cancel becomes available only while a tracked SUPEK job is still queued or running.")

        if _show_panel(remote_panel, "Latest SUPEK console snapshot"):
            with _dropdown_section(st, "Latest SUPEK console snapshot", expanded=False):
                _render_remote_console(
                    st,
                    title="Latest SUPEK console snapshot",
                    actions=list(run.get("dashboard_actions", [])),
                    kinds={
                        "supek-verify-env",
                        "supek-sync-repo",
                        "supek-sync-run",
                        "supek-submit-preflight",
                        "supek-poll-qstat",
                        "supek-fetch-logs",
                        "supek-submit-workflow",
                        "supek-cancel-job",
                        "supek-pull-artifacts",
                    },
                    key_prefix=f"supek_console_{run['run_slug']}",
                )
        return

    if selected_section == "Decision log":
        _render_decision_workspace(st, state, scope="run", run=run)
        return

    if selected_section == "Recent actions":
        st.subheader("Recent actions")
        _render_action_history(
            st,
            actions=list(run.get("dashboard_actions", [])),
            run_root=str(state["run_root"]),
            key_prefix=f"run_{run['run_slug']}",
        )
        return

    st.subheader("Technical details and reference commands")
    st.markdown("#### Run config")
    st.json(run["config"])
    for command in run["local_commands"]:
        st.write(command["label"])
        st.code(command["command"], language="bash")
