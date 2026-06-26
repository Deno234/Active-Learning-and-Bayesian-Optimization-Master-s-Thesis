from __future__ import annotations

from active_learning_thesis.dashboard_thesis_checklist import (
    build_thesis_phase_rows,
    build_thesis_phase_summary,
)


GUIDED_PANEL_VIEWS = {"Today", "Model Workflow", "MD Validation", "Operations", "Results"}
GUIDED_TASK_SECTIONS = (
    "Replay / Study Benchmark",
    "Real AL",
    "MD Simulation",
    "Label & Ingest",
    "Discovery / Bayesian Optimization Search",
    "Results / Thesis Export",
    "Advanced / Debug / Remote Ops",
)


def _task_for_queue_row(row: dict[str, object]) -> str:
    category = str(row.get("category", "")).strip()
    action = str(row.get("action_now", "")).strip().lower()
    if "replay" in action or category == "Study":
        return "Replay / Study Benchmark"
    if "propose" in action or category in {"Model workflow", "Candidate selection"}:
        return "Real AL"
    if category in {"MD preparation", "MD slate"}:
        return "MD Simulation"
    if category in {"Review", "Ingest", "AL promotion"}:
        return "Label & Ingest"
    if "discovery" in action or "bayesian" in action:
        return "Discovery / Bayesian Optimization Search"
    if category == "Reporting":
        return "Results / Thesis Export"
    if category in {"Remote monitoring", "Remote reconciliation", "Slate recovery", "Artifact verification"}:
        return "Advanced / Debug / Remote Ops"
    return category or "Replay / Study Benchmark"


def _canonical_for_queue_row(row: dict[str, object]) -> str:
    category = str(row.get("category", "")).strip()
    open_view = str(row.get("open_view", "")).strip()
    if category == "Candidate selection":
        return "Peptides -> Candidate selection"
    if category in {"Review", "Ingest", "AL promotion"}:
        return "MD Validation -> Review & ingest"
    if category in {"MD preparation", "MD slate", "Slate recovery"}:
        return "MD Validation -> Ladder overview"
    if category in {"Remote monitoring", "Remote reconciliation", "Artifact verification"}:
        return "Operations -> Remote jobs"
    if open_view == "Model Workflow":
        return "Model Workflow -> Local model actions"
    if open_view == "Results":
        return "Results -> Thesis output builder"
    if open_view:
        return open_view
    return ""


def build_guided_state_panel(state: dict[str, object]) -> dict[str, str]:
    rows = build_thesis_phase_rows(state)
    summary = build_thesis_phase_summary(rows)
    overview = state.get("overview", {}) if isinstance(state.get("overview", {}), dict) else {}
    today_queue = overview.get("today_queue", []) if isinstance(overview.get("today_queue", []), list) else []
    if today_queue and isinstance(today_queue[0], dict):
        row = today_queue[0]
        blocking_items = overview.get("blocked_items", []) if isinstance(overview.get("blocked_items", []), list) else []
        blocker = ""
        if blocking_items and isinstance(blocking_items[0], dict):
            blocker = str(blocking_items[0].get("blocker", "") or "").strip()
        return {
            "current_phase": _task_for_queue_row(row),
            "exact_next_action": str(row.get("action_now", "") or "").strip(),
            "canonical_page_section": _canonical_for_queue_row(row),
            "blocking_issue": blocker or "None visible",
            "why": str(row.get("why", "") or "This is the highest-priority visible task in the current thesis cockpit.").strip(),
        }
    current_phase = str(summary.get("next_phase", "Setup") or "Setup")
    row = next((item for item in rows if str(item.get("phase", "")) == current_phase), rows[0] if rows else {})
    blocking_issue = str(row.get("blocker", "") or "").strip()
    if blocking_issue == "-":
        blocking_issue = ""
    next_click = str(row.get("next_click", summary.get("next_click", "")) or "").strip()
    return {
        "current_phase": current_phase,
        "exact_next_action": str(row.get("safe_next_move", summary.get("safe_next_move", "")) or "").strip(),
        "canonical_page_section": next_click,
        "blocking_issue": blocking_issue or "None visible",
        "why": str(row.get("why_it_matters", "") or "This keeps the normal thesis workflow moving without exposing debug controls first.").strip(),
    }


def render_guided_state_panel(st, state: dict[str, object], *, view: str) -> None:
    if str(state.get("workflow_mode", "Guided thesis mode")) != "Guided thesis mode":
        return
    if view not in GUIDED_PANEL_VIEWS:
        return
    panel = build_guided_state_panel(state)
    st.subheader("Current state / Next action")
    st.markdown(f"**Current phase:** {panel['current_phase']}")
    st.markdown(f"**Exact next action:** {panel['exact_next_action']}")
    st.markdown(f"**Canonical page/section:** {panel['canonical_page_section']}")
    st.caption("Task-centered cockpit: " + " | ".join(GUIDED_TASK_SECTIONS))
    blocker = panel["blocking_issue"]
    if blocker and blocker != "None visible":
        st.warning(f"Blocking issue: {blocker}")
    else:
        st.info("Blocking issue: None visible")
    st.caption(f"Why: {panel['why']}")
