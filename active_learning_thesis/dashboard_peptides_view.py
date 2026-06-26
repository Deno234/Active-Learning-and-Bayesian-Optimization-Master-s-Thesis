from __future__ import annotations

from types import ModuleType


PEPTIDE_BUCKET_SPECS = [
    ("Lifecycle ledger", "ledger", "One row per peptide / run pair, showing where that peptide currently sits in the thesis workflow."),
    ("Candidate selection", "candidate_selection", "One queue for deciding which proposed or discovery peptides should enter MD next, which ones should wait, and which ones should stay out."),
    ("Bulk review / ingest", "review_pipeline", "A centralized workspace for the fastest human-review and ingest handoff back into the model."),
    ("Review & ingest queue", "review_pipeline", "The shortest path back into the model: either assign the human label or create / ingest `cgmd_ingest.csv`."),
    ("Suggested by model", "suggested_by_model", "Peptides currently proposed by the model for validation."),
    ("Sent for MD", "sent_for_md", "Peptides that already have a guided MD ladder entry."),
    ("MD in progress", "md_in_progress", "Peptides currently staged, submitted, or otherwise active in the MD workflow."),
    ("Needs review / label", "needs_review", "Full-analysis peptides waiting for human review or `cgmd_label` assignment."),
    ("Ready for ingest", "ready_for_ingest", "Reviewed peptides that are ready to become `cgmd_ingest.csv` rows."),
    ("Already ingested", "already_ingested", "Peptides that have already been fed back into the model via `ingest-round`."),
]
GUIDED_REVIEW_BUCKETS = {"Bulk review / ingest", "Review & ingest queue"}


def render_peptides_view(st, state: dict[str, object], *, ns: ModuleType) -> None:
    _persisted_choice = ns._persisted_choice
    _queue_query_param_update = ns._queue_query_param_update
    _render_bulk_review_ingest_workspace = ns._render_bulk_review_ingest_workspace
    _render_candidate_selection_workspace = ns._render_candidate_selection_workspace
    _render_metric_cards = ns._render_metric_cards
    _trigger_dashboard_rerun = ns._trigger_dashboard_rerun

    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    counts = inventory.get("counts", {}) if isinstance(inventory.get("counts", {}), dict) else {}
    st.header("Peptides")
    st.caption("This view tracks peptides in thesis language: suggested, sent to MD, waiting on review, ready for ingest, and already fed back into the model.")
    _render_metric_cards(st, list(counts.items()))

    guided_mode = str(state.get("workflow_mode", "Expert mode")) == "Guided thesis mode"
    bucket_titles = [spec[0] for spec in PEPTIDE_BUCKET_SPECS]
    default_bucket = "Candidate selection" if guided_mode else bucket_titles[0]
    selected_bucket = _persisted_choice(
        st,
        st.radio,
        label="Peptide inventory section",
        options=bucket_titles,
        key="dashboard_peptides_section",
        query_key="peptides_section",
        default=default_bucket,
        write_query=False,
    )
    title, key, description = next(spec for spec in PEPTIDE_BUCKET_SPECS if spec[0] == selected_bucket)
    st.subheader(title)
    st.caption(description)
    if guided_mode and selected_bucket in GUIDED_REVIEW_BUCKETS:
        st.info(
            "Review and ingest execution now has one canonical home: MD Validation -> Review & ingest. "
            "This Peptides page stays focused on lifecycle tracking and candidate choice."
        )
        ready_count = int(counts.get("ready_for_ingest", 0) or 0)
        review_count = int(counts.get("needs_review", 0) or 0)
        st.caption(f"Mirror status: {review_count} need review, {ready_count} are ready for ingest.")
        if st.button("Open MD Validation -> Review & ingest", key="peptides_open_md_review_ingest"):
            session_state = getattr(st, "session_state", {})
            try:
                session_state["dashboard_view"] = "MD Validation"
                session_state["dashboard_md_section"] = "Review & ingest"
            except Exception:
                pass
            _queue_query_param_update(st, "view", "MD Validation")
            _queue_query_param_update(st, "md_section", "Review & ingest")
            _trigger_dashboard_rerun(st)
        return
    if selected_bucket == "Candidate selection":
        _render_candidate_selection_workspace(st, state)
        return
    if selected_bucket == "Bulk review / ingest":
        _render_bulk_review_ingest_workspace(st, state)
        return
    rows = list(inventory.get(key, [])) if isinstance(inventory.get(key, []), list) else []
    if rows:
        if key == "ledger":
            st.info("Use this ledger when you want one answer per peptide: who suggested it, where it is in MD, whether it already has a review label, and what should happen next.")
        elif key == "review_pipeline":
            st.info("Use this queue when you want the fastest path from finished MD analysis back into the model. If the label is missing, review first. If the label exists, create or ingest the CSV next.")
        st.dataframe(rows)
    else:
        st.info(f"No peptides are currently in '{title}'.")
