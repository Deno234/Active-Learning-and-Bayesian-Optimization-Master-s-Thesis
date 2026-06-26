from __future__ import annotations

import html

from active_learning_thesis.dashboard_metadata import PAGE_GUIDES
from active_learning_thesis.dashboard_preferences import DASHBOARD_UI_MODES, DEFAULT_DASHBOARD_UI_MODE


def _dashboard_ui_mode(st, *, default: str = DEFAULT_DASHBOARD_UI_MODE) -> str:
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        return default
    selected = str(session_state.get("dashboard_ui_mode", default))
    if selected not in DASHBOARD_UI_MODES:
        return default
    return selected


def render_recommended_card(
    st,
    *,
    eyebrow: str,
    title: str,
    summary: str,
    why: str = "",
    do_now: str = "",
    next_after: str = "",
) -> None:
    if _dashboard_ui_mode(st) == "Stable mode":
        st.caption(str(eyebrow).upper())
        st.subheader(title)
        st.write(summary)
        if why:
            st.caption(f"Why this is next: {why}")
        if do_now:
            st.info(f"Do this now: {do_now}")
        if next_after:
            st.caption(f"After that: {next_after}")
        return
    st.markdown(
        (
            "<div style='background: linear-gradient(135deg, rgba(29,78,216,0.10) 0%, rgba(34,197,94,0.08) 100%); "
            "border: 1px solid rgba(29,78,216,0.16); border-radius: 20px; padding: 1rem 1.1rem; margin: 0.25rem 0 1rem;'>"
            f"<div style='font-size:0.78rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#1d4ed8;'>{eyebrow}</div>"
            f"<div style='font-size:1.3rem; font-weight:700; color:#0f172a; margin-top:0.2rem;'>{title}</div>"
            f"<div style='font-size:0.98rem; color:#334155; margin-top:0.35rem;'>{summary}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if why:
        st.caption(f"Why this is next: {why}")
    if do_now:
        st.info(f"Do this now: {do_now}")
    if next_after:
        st.caption(f"After that: {next_after}")


def render_plan_checkpoint_table(st, *, title: str, rows: list[dict[str, str]], summary: str) -> None:
    st.write(title)
    st.caption(summary)
    if rows:
        st.dataframe(rows)
        current = next((row for row in rows if row.get("status") == "current"), None)
        if current:
            st.info(f"Current checkpoint: {current.get('checkpoint', '')}")


def render_runner_memory_panel(
    st,
    *,
    current_checkpoint: str,
    events: list[dict[str, object]],
    blocker: str = "",
    blocker_level: str = "warning",
) -> None:
    st.write("Runner memory")
    latest = events[0] if events else None
    if latest is not None:
        render_metric_cards(
            st,
            [
                ("Last advanced checkpoint", str(latest.get("checkpoint", "")) or "-"),
                ("Last advanced at", str(latest.get("created_at", "")) or "-"),
                ("Recorded steps", len(events)),
            ],
        )
        action_label = str(latest.get("action_label", "")).strip()
        if action_label:
            st.caption(f"Last chosen path: {action_label}")
    else:
        st.caption("No checkpoint advance has been recorded from the cockpit for this path yet.")

    if blocker.strip():
        message = f"Currently blocked because: {blocker.strip()}"
        if blocker_level == "error":
            st.error(message)
        elif blocker_level == "info":
            st.info(message)
        else:
            st.warning(message)
    elif current_checkpoint:
        st.info(f"Current gate: {current_checkpoint} is the next safe checkpoint from the dashboard's point of view.")

    if events:
        st.caption("Recent checkpoint trail")
        st.dataframe(
            [
                {
                    "when": str(item.get("created_at", "")),
                    "checkpoint": str(item.get("checkpoint", "")),
                    "action": str(item.get("action_label", "")),
                    "status": str(item.get("action_status", "")),
                    "note": str(item.get("note", "")),
                }
                for item in events[:6]
            ]
        )


def render_metric_cards(st, items: list[tuple[str, object]]) -> None:
    if not items:
        return
    columns = st.columns(len(items))
    if _dashboard_ui_mode(st) == "Stable mode":
        for column, (label, value) in zip(columns, items):
            column.caption(str(label))
            column.write(str(value))
        return
    for column, (label, value) in zip(columns, items):
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        column.markdown(
            (
                "<div class='dashboard-summary-card'>"
                f"<div class='dashboard-summary-label'>{safe_label}</div>"
                f"<div class='dashboard-summary-value'>{safe_value}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_page_guide(st, *, view: str, state: dict[str, object]) -> None:
    guide = PAGE_GUIDES.get(view)
    if not guide:
        return
    overview = state.get("overview", {}) if isinstance(state.get("overview", {}), dict) else {}
    contextual_note = {
        "Today": f"Current workspace snapshot: {overview.get('run_count', 0)} visible runs, {overview.get('peptide_count', 0)} peptide ladders, {len(overview.get('approval_queue', []))} approvals, and {len(state.get('notifications', []))} active notifications.",
        "Model Workflow": "Choose one thesis run in-page, then use the checkpoint runner and section chooser instead of trying to interpret the whole run directory at once.",
        "Results": "This page is read-heavy by design. Use it when you want evidence and comparisons, not when you want to trigger the next operational step.",
        "Peptides": "This page answers peptide questions directly: which candidates to send, which peptides are in MD, and which ones are ready for review or ingest.",
        "MD Validation": "This page is intentionally peptide-centric. Pick one peptide, follow the ladder, and let the checkpoint copy tell you the next safe MD action.",
        "Operations": "This page is the global control room. It is the right place for cluster readiness, approvals, logs, transfers, and curation rather than experiment interpretation.",
    }.get(view, "")
    st.info(
        f"Page guide: {guide['purpose']}\n\n"
        f"Best for: {guide['best_for']}\n\n"
        f"On this page: {guide['contains']}"
    )
    if contextual_note:
        st.caption(contextual_note)


def render_action_guidance(
    st,
    *,
    what: str = "",
    when: str = "",
    produces: str = "",
    next_step: str = "",
) -> None:
    items = [
        ("What this does", what),
        ("When to use it", when),
        ("What it produces", produces),
        ("What becomes available next", next_step),
    ]
    lines = [f"{label}: {str(value).strip()}" for label, value in items if str(value).strip()]
    if not lines:
        return
    tooltip = html.escape("\n".join(lines), quote=True).replace("\n", "&#10;")
    st.markdown(
        (
            "<span "
            f"title=\"{tooltip}\" "
            "style=\"display:inline-flex;align-items:center;justify-content:center;width:1.1rem;height:1.1rem;"
            "border-radius:999px;background:#e0f2fe;color:#075985;font-size:0.78rem;font-weight:800;"
            "border:1px solid #bae6fd;cursor:help;margin-right:0.35rem;\">i</span>"
            "<span style=\"font-size:0.82rem;color:#475569;\">Action info</span>"
        ),
        unsafe_allow_html=True,
    )


def render_badges(st, badges: list[str]) -> None:
    if not badges:
        return
    st.write("Badges: " + " ".join(f"`{badge}`" for badge in badges))


def inject_dashboard_theme(st, *, ui_mode: str) -> None:
    base_css = """
        <style>
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.25rem;
            max-width: 1500px;
        }
        [data-testid="stElementContainer"][data-stale="true"] {
            display: none !important;
        }
        [data-testid="stElementContainer"][data-stale="true"] * {
            display: none !important;
        }
        [data-testid="stElementContainer"][data-stale="true"] details,
        [data-testid="stElementContainer"][data-stale="true"] [data-testid="stExpander"],
        [data-testid="stElementContainer"][data-stale="true"] [data-testid="stExpanderDetails"] {
            display: none !important;
        }
        [data-testid="stElementContainer"][data-stale="true"] {
            pointer-events: none !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            transform: none !important;
        }
        </style>
    """
    if ui_mode != "Rich mode":
        st.markdown(base_css, unsafe_allow_html=True)
        return
    st.markdown(
        base_css
        + """
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f8ff 0%, #eef3fb 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        [data-testid="stMetric"] {
            background: #f8fbff;
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .dashboard-summary-card {
            background: #f8fbff;
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
            min-height: 118px;
        }
        .dashboard-summary-label {
            font-size: 0.92rem;
            color: #334155;
            margin-bottom: 0.35rem;
        }
        .dashboard-summary-value {
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 700;
            color: #0f172a;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        div[data-baseweb="tab"] {
            border-radius: 999px;
            padding-left: 0.95rem;
            padding-right: 0.95rem;
            background: rgba(226, 232, 240, 0.55);
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #1d4ed8;
            color: white;
        }
        [data-testid="stCodeBlock"] {
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            overflow: hidden;
        }
        .dashboard-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #22c55e 100%);
            color: white;
            padding: 1.2rem 1.35rem;
            border-radius: 22px;
            margin-bottom: 1rem;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
        }
        .dashboard-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }
        .dashboard-hero p {
            margin: 0.45rem 0 0;
            opacity: 0.92;
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_hero(st, *, ui_mode: str) -> None:
    if ui_mode == "Stable mode":
        st.title("Active Learning Thesis Admin Cockpit")
        st.write("Thesis-first run curation, peptide tracking, guided MD validation, and safe SUPEK/BURA operations from one place.")
        return
    st.markdown(
        (
            "<div class='dashboard-hero'>"
            "<h1>Active Learning Thesis Admin Cockpit</h1>"
            "<p>Thesis-first run curation, peptide tracking, guided MD validation, and safe SUPEK/BURA operations from one place.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
