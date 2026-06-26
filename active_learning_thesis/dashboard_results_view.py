from __future__ import annotations

from pathlib import Path
from types import ModuleType


ANALYTICS_SECTIONS = [
    "Thesis packet",
    "Thesis output builder",
    "Study comparison hub",
    "Appendix packet",
    "Thesis narrative",
    "Compare selected runs",
    "Thesis milestones",
    "Thesis scorecards",
    "Figure-ready comparisons",
    "Metric history",
    "Replay evidence",
    "Discovery evidence",
    "Review / feedback audit",
    "Promotion audit",
    "Peptide provenance audit",
    "Thesis decision log",
    "Export-ready tables",
    "MD status",
    "Action timeline",
]


def render_results_view(st, state: dict[str, object], *, ns: ModuleType) -> None:
    _bar_chart_rows = ns._bar_chart_rows
    _appendix_packet_markdown = ns._appendix_packet_markdown
    _best_final_run = ns._best_final_run
    _concat_frames = ns._concat_frames
    _decision_log_rows = ns._decision_log_rows
    _discovery_highlight_rows = ns._discovery_highlight_rows
    _figure_caption_rows = ns._figure_caption_rows
    _format_float = ns._format_float
    _frame_empty = ns._frame_empty
    _frame_records = ns._frame_records
    _focused_comparison_rows = ns._focused_comparison_rows
    _multi_metric_chart = ns._multi_metric_chart
    _persisted_choice = ns._persisted_choice
    _render_export_pack = ns._render_export_pack
    _render_metric_cards = ns._render_metric_cards
    _reporting_readiness_counts = ns._reporting_readiness_counts
    _reporting_readiness_for_run = ns._reporting_readiness_for_run
    _promotion_audit_rows = ns._promotion_audit_rows
    _peptide_provenance_audit_rows = ns._peptide_provenance_audit_rows
    _replay_best_strategy_rows = ns._replay_best_strategy_rows
    _result_scorecard_rows = ns._result_scorecard_rows
    _result_summary_rows = ns._result_summary_rows
    _review_feedback_audit_rows = ns._review_feedback_audit_rows
    _safe_float = ns._safe_float
    _selected_results_runs = ns._selected_results_runs
    _selected_review_audit_rows = ns._selected_review_audit_rows
    _thesis_milestone_rows = ns._thesis_milestone_rows
    _thesis_narrative_callout_rows = ns._thesis_narrative_callout_rows
    _thesis_narrative_markdown = ns._thesis_narrative_markdown
    _thesis_packet_markdown = ns._thesis_packet_markdown
    _wide_chart = ns._wide_chart
    _with_constant = ns._with_constant
    action_timeline_frame = ns.action_timeline_frame
    discovery_frame = ns.discovery_frame
    md_ladder_summary_frame = ns.md_ladder_summary_frame
    METRIC_FIELDS = ns.METRIC_FIELDS
    replay_curve_frame = ns.replay_curve_frame
    run_metric_history = ns.run_metric_history
    _quote_path = ns._quote_path
    _base_render_launch_action = ns._render_launch_action
    submit_thesis_canary_action = ns.submit_thesis_canary_action
    submit_thesis_figures_action = ns.submit_thesis_figures_action
    submit_thesis_packet_action = ns.submit_thesis_packet_action
    discover_study_comparisons = ns.discover_study_comparisons
    discover_study_manifests = ns.discover_study_manifests
    discover_study_summaries = ns.discover_study_summaries
    read_csv_rows = ns.read_csv_rows
    read_json_file = ns.read_json_file

    st.header("Results")
    st.caption("Use this page when you want thesis-ready comparisons, final metrics, replay evidence, discovery summaries, and no-CLI reporting exports.")
    runs = list(state.get("runs", []))
    if not runs:
        st.info("No runs found for the current filters.")
        return

    final_ready_runs = [run for run in runs if run.get("final_metrics")]
    discovery_ready_runs = [run for run in runs if _frame_records(discovery_frame(run))]
    replay_ready_runs = [run for run in runs if _frame_records(replay_curve_frame(run))]
    _render_metric_cards(
        st,
        [
            ("Visible runs", len(runs)),
            ("Runs with final metrics", len(final_ready_runs)),
            ("Runs with replay data", len(replay_ready_runs)),
            ("Runs with discovery summaries", len(discovery_ready_runs)),
        ],
    )
    st.info("This page is for reporting and comparison. Use Model Workflow for acting on one run, and use Results when you want evidence, packets, figures, tables, and thesis discussion.")
    if str(state.get("workflow_mode", "Expert mode")) == "Guided thesis mode":
        st.caption(
            "Guided thesis tasks here: Replay / Study Benchmark | Discovery / Bayesian Optimization Search | Results / Thesis Export."
        )

    metric_name = st.selectbox("Metric for charts", METRIC_FIELDS, index=0, key="results_metric")
    best_final_run = _best_final_run(runs, metric_name=metric_name)
    highlights = [
        ("Chosen metric", metric_name),
        ("Best frozen final run", str(best_final_run.get("run_display_name", best_final_run.get("run_name", ""))) if best_final_run else "No frozen final metrics yet"),
        (
            f"Best final {metric_name}",
            _format_float(_safe_float(best_final_run.get("final_metrics", {}).get(metric_name))) if best_final_run else "",
        ),
        ("Replay evidence runs", len(replay_ready_runs)),
        ("Discovery evidence runs", len(discovery_ready_runs)),
    ]
    _render_metric_cards(st, highlights)
    selected_section = _persisted_choice(
        st,
        st.radio,
        label="Results section",
        options=ANALYTICS_SECTIONS,
        key="dashboard_results_section",
        query_key="results_section",
        default=ANALYTICS_SECTIONS[0],
        write_query=False,
    )

    def _render_launch_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "Results")
        kwargs.setdefault("section", selected_section)
        return _base_render_launch_action(*args, **kwargs)

    if selected_section == "Thesis output builder":
        run_root = Path(str(state["run_root"]))
        st.subheader("Thesis output builder")
        st.caption("This is the no-CLI handoff for final writing artifacts: export the packet, build figure/table/caption bundles from a packet, and optionally run the seeded canary that proves the end-to-end thesis path still works.")
        packet_title = st.text_input(
            "Packet title",
            value="thesis_packet",
            key="results_output_packet_title",
        )
        packet_command = (
            "python -m active_learning_thesis export-thesis-packet "
            f"--run-root {_quote_path(run_root)} "
            f'--title "{str(packet_title).replace(chr(34), "").strip() or "thesis_packet"}" '
            f"--metric {metric_name}"
        )
        _render_launch_action(
            st,
            label="Export thesis packet",
            command=packet_command,
            key_prefix="results_export_thesis_packet",
            button_text="Export packet",
            what="Collect run summaries, metrics, MD review evidence, dashboard-derived readiness tables, canary reports, final freezes, and reproducibility metadata into one timestamped packet.",
            when="Use this after at least one run has evidence you want to report, especially after freezing your final run.",
            produces="A thesis packet directory under `_thesis_packets/` with README, manifest, tables, figure data, and metadata.",
            next_step="Build thesis figures from the newest packet below.",
            contract_id="export-thesis-packet",
            on_submit=lambda packet_title=packet_title, metric_name=metric_name: submit_thesis_packet_action(
                run_root=run_root,
                title=str(packet_title).strip() or "thesis_packet",
                metric=str(metric_name),
            ),
        )

        packet_dirs: list[Path] = []
        packet_root = run_root / "_thesis_packets"
        if packet_root.exists():
            packet_dirs = sorted(
                [path.parent for path in packet_root.glob("*/packet_manifest.json")],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        st.markdown("#### Build figures from a packet")
        if packet_dirs:
            packet_options = [str(path) for path in packet_dirs]
            selected_packet = st.selectbox(
                "Packet directory",
                packet_options,
                index=0,
                key="results_output_packet_dir",
            )
            figure_command = (
                "python -m active_learning_thesis build-thesis-figures "
                f"--packet-dir {_quote_path(selected_packet)} "
                f"--metric {metric_name}"
            )
            _render_launch_action(
                st,
                label="Build thesis figures",
                command=figure_command,
                key_prefix="results_build_thesis_figures",
                button_text="Build figures",
                what="Generate output-only SVG figures, clean CSV tables, and copy-ready caption markdown from the selected packet.",
                when="Use this after exporting a packet, or when you want to rebuild figures from a newer packet.",
                produces="A `thesis_figures/` bundle under the selected packet.",
                next_step="Open the generated README/captions and choose the figures that belong in the thesis.",
                contract_id="build-thesis-figures",
                on_submit=lambda selected_packet=selected_packet, metric_name=metric_name: submit_thesis_figures_action(
                    run_root=run_root,
                    packet_dir=Path(str(selected_packet)),
                    metric=str(metric_name),
                ),
            )
        else:
            st.info("No exported thesis packet is available yet. Export a packet above, refresh the dashboard, then build figures from it.")

        st.markdown("#### Confidence check")
        st.caption("The canary is optional, but useful before a thesis milestone because it proves the synthetic end-to-end loop still works after code changes.")
        canary_name = st.text_input(
            "Canary name",
            value="seeded_thesis_canary",
            key="results_canary_name",
        )
        canary_seed_text = st.text_input(
            "Canary seed",
            value="20260425",
            key="results_canary_seed",
        )
        canary_peptides_text = st.text_input(
            "Canary peptides",
            value="2",
            key="results_canary_peptides",
        )
        force_canary = st.checkbox(
            "Replace existing canary with the same name and seed",
            value=False,
            key="results_canary_force",
        )
        try:
            canary_seed = int(str(canary_seed_text).strip())
        except ValueError:
            canary_seed = 20260425
            st.warning("Canary seed must be an integer. The action will use 20260425 unless you correct it.")
        try:
            canary_peptides = max(1, int(str(canary_peptides_text).strip()))
        except ValueError:
            canary_peptides = 2
            st.warning("Canary peptides must be an integer. The action will use 2 unless you correct it.")
        canary_command = (
            "python -m active_learning_thesis thesis-canary "
            f"--run-root {_quote_path(run_root)} "
            f'--name "{str(canary_name).replace(chr(34), "").strip() or "seeded_thesis_canary"}" '
            f"--seed {canary_seed} "
            f"--peptides {canary_peptides}"
        )
        if force_canary:
            canary_command += " --force"
        _render_launch_action(
            st,
            label="Run seeded thesis canary",
            command=canary_command,
            key_prefix="results_thesis_canary",
            button_text="Run canary",
            what="Create a deterministic synthetic thesis loop with generated MD outputs, evidence-backed review rows, ingest validation, and a canary report.",
            when="Use this before an important thesis export or after changing workflow code.",
            produces="A canary report under `_thesis_canaries/` that says whether the end-to-end thesis path still passes.",
            next_step="If it passes, export the thesis packet; if it fails, inspect the canary report before trusting new workflow changes.",
            contract_id="thesis-canary",
            on_submit=lambda canary_name=canary_name, canary_seed=canary_seed, canary_peptides=canary_peptides, force_canary=force_canary: submit_thesis_canary_action(
                run_root=run_root,
                name=str(canary_name).strip() or "seeded_thesis_canary",
                seed=int(canary_seed),
                peptides=int(canary_peptides),
                force=bool(force_canary),
            ),
        )
        return

    if selected_section == "Study comparison hub":
        run_root = Path(str(state["run_root"]))
        study_rows = discover_study_manifests(run_root)
        summary_rows = discover_study_summaries(run_root)
        comparison_rows = discover_study_comparisons(run_root)
        st.subheader("Study comparison hub")
        st.caption("Use this section when you need thesis-strength evidence across seeds, acquisition strategies, or ablation studies. The tables shown here are generated by Operations -> Study designer.")
        _render_metric_cards(
            st,
            [
                ("Study manifests", len(study_rows)),
                ("Study summaries", len(summary_rows)),
                ("Study comparisons", len(comparison_rows)),
                ("Completed studies", sum(1 for row in study_rows if str(row.get("status", "")) == "completed")),
            ],
        )
        replay_best = _replay_best_strategy_rows(runs, metric_name=metric_name)
        if replay_best:
            st.markdown("#### Single-run replay benchmarks")
            st.caption(
                "These are not formal multi-seed studies yet, but they are useful first-pass SUPEK replay results. "
                "Use them to choose which strategies deserve a real multi-seed study."
            )
            st.dataframe(replay_best)
            _render_export_pack(
                st,
                title="Single-run replay benchmark summary",
                description=f"Best replay-benchmark strategy per visible run by `{metric_name}`.",
                rows=replay_best,
                key_prefix="study_hub_single_replay",
            )
        else:
            st.info("No single-run replay benchmark rows found for the visible runs.")

        if study_rows:
            st.markdown("#### Study manifests")
            st.dataframe(study_rows)
            selected_manifest = st.selectbox(
                "Inspect study manifest",
                [str(row["manifest"]) for row in study_rows],
                index=0,
                key="results_study_manifest",
            )
            manifest = read_json_file(Path(selected_manifest))
            config = manifest.get("config", {}) if isinstance(manifest.get("config", {}), dict) else {}
            runs_in_manifest = list(manifest.get("runs", [])) if isinstance(manifest.get("runs", []), list) else []
            _render_metric_cards(
                st,
                [
                    ("Status", manifest.get("status", "")),
                    ("Runs", manifest.get("run_count", len(runs_in_manifest))),
                    ("Completed", manifest.get("completed_run_count", "")),
                    ("Failures", manifest.get("failure_count", 0)),
                    ("Metric", config.get("metric", "")),
                ],
            )
            if runs_in_manifest:
                st.dataframe(runs_in_manifest)
        else:
            st.info("No study manifests found yet. Create one from Operations -> Study designer.")

        if summary_rows:
            st.markdown("#### Strategy summary evidence")
            st.dataframe(summary_rows)
            selected_summary = st.selectbox(
                "Inspect study summary",
                [str(row["summary"]) for row in summary_rows],
                index=0,
                key="results_study_summary",
            )
            summary = read_json_file(Path(selected_summary))
            outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs", {}), dict) else {}
            best_strategy = str(summary.get("best_strategy_by_aulc", ""))
            if best_strategy:
                st.success(f"Best strategy by AULC({summary.get('metric', '')}): {best_strategy}")
            strategy_rows = read_csv_rows(Path(str(outputs.get("strategy_summary", ""))), limit=100)
            paired_rows = read_csv_rows(Path(str(outputs.get("paired_vs_random", ""))), limit=100)
            run_strategy_rows = read_csv_rows(Path(str(outputs.get("run_strategy_summary", ""))), limit=100)
            if strategy_rows:
                st.markdown("#### Strategy-level table")
                st.dataframe(strategy_rows)
            if paired_rows:
                st.markdown("#### Paired vs random table")
                st.dataframe(paired_rows)
            if run_strategy_rows:
                st.markdown("#### Run/strategy table")
                st.dataframe(run_strategy_rows)
        else:
            st.info("No study summary files found yet. Summarize evidence from Operations -> Study designer.")

        if comparison_rows:
            st.markdown("#### Matched study comparisons")
            st.dataframe(comparison_rows)
            selected_comparison = st.selectbox(
                "Inspect study comparison",
                [str(row["summary"]) for row in comparison_rows],
                index=0,
                key="results_study_comparison",
            )
            comparison = read_json_file(Path(selected_comparison))
            outputs = comparison.get("outputs", {}) if isinstance(comparison.get("outputs", {}), dict) else {}
            narrative_value = str(outputs.get("thesis_narrative", "")).strip()
            narrative_path = Path(narrative_value) if narrative_value else None
            if narrative_path is not None and narrative_path.is_file():
                st.markdown("#### Thesis narrative")
                try:
                    narrative_text = narrative_path.read_text(encoding="utf-8")
                except OSError as exc:
                    st.warning(f"Could not read thesis narrative: {exc}")
                else:
                    st.code(narrative_text, language="markdown")
            strategy_delta_rows = read_csv_rows(Path(str(outputs.get("strategy_summary", ""))), limit=100)
            paired_comparison_rows = read_csv_rows(Path(str(outputs.get("paired_comparison", ""))), limit=100)
            if strategy_delta_rows:
                st.markdown("#### Strategy deltas")
                st.dataframe(strategy_delta_rows)
            if paired_comparison_rows:
                st.markdown("#### Paired study rows")
                st.dataframe(paired_comparison_rows)
        else:
            st.info("No matched study comparisons found yet. Compare two study manifests from Operations -> Study designer.")
        return

    if selected_section == "Thesis packet":
        scorecard_rows = _result_scorecard_rows(runs, metric_name=metric_name)
        milestone_rows = _thesis_milestone_rows(runs)
        review_audit = _review_feedback_audit_rows(state)
        promotion_audit = _promotion_audit_rows(state)
        provenance_audit = _peptide_provenance_audit_rows(state)
        decision_rows = _decision_log_rows(state)
        replay_best = _replay_best_strategy_rows(runs, metric_name=metric_name)
        discovery_best = _discovery_highlight_rows(runs)
        caption_rows = _figure_caption_rows(runs, metric_name=metric_name)
        readiness_counts = _reporting_readiness_counts(runs)
        report_ready_count = sum(int(row.get("count", 0) or 0) for row in readiness_counts if str(row.get("readiness", "")) == "Report-ready")
        needs_ingest_count = sum(int(row.get("count", 0) or 0) for row in readiness_counts if str(row.get("readiness", "")) == "Needs ingest")
        review_pending = sum(1 for row in review_audit if str(row.get("feedback_state", "")) == "Needs human review")
        model_ingest_ready = sum(1 for row in review_audit if str(row.get("feedback_state", "")) == "Ready for model ingest")
        promoted_count = sum(
            1 for row in promotion_audit if str(row.get("promotion_state", "")).startswith("Promoted into real batch")
        )
        st.subheader("Thesis packet")
        st.caption("Use this section as the thesis-writing handoff: a compact packet of scorecards, figure captions, review/feedback traceability, and copy-ready tables.")
        _render_metric_cards(
            st,
            [
                ("Report-ready runs", report_ready_count),
                ("Runs needing ingest", needs_ingest_count),
                ("Peptides needing review", review_pending),
                ("Peptides ready for model ingest", model_ingest_ready),
                ("Promotion bridges recorded", promoted_count),
                ("Recorded thesis decisions", len(decision_rows)),
            ],
        )
        st.markdown("#### Copy-ready thesis summary")
        st.code(_thesis_packet_markdown(runs, metric_name=metric_name, state=state), language="markdown")
        st.markdown("#### Comparison snapshot")
        if scorecard_rows:
            st.dataframe(scorecard_rows)
        else:
            st.info("No scorecard rows are available for the visible runs.")
        if caption_rows:
            st.markdown("#### Suggested figure captions")
            st.dataframe(caption_rows)
        if review_audit:
            st.markdown("#### Review / feedback audit snapshot")
            st.dataframe(review_audit)
        if promotion_audit:
            st.markdown("#### Promotion audit snapshot")
            st.dataframe(promotion_audit)
        if provenance_audit:
            st.markdown("#### Peptide provenance snapshot")
            st.dataframe(provenance_audit[:12])
        if decision_rows:
            st.markdown("#### Thesis decision snapshot")
            st.dataframe(decision_rows[:10])
        if milestone_rows:
            st.markdown("#### Thesis milestones snapshot")
            st.dataframe(milestone_rows)
        _render_export_pack(
            st,
            title="Thesis packet comparison table",
            description="Primary comparison table for visible thesis runs.",
            rows=scorecard_rows,
            key_prefix="results_packet_scorecards",
        )
        _render_export_pack(
            st,
            title="Thesis packet review / feedback audit",
            description="Traceable human-review and model-feedback status for full-analysis peptides.",
            rows=review_audit,
            key_prefix="results_packet_review_audit",
        )
        _render_export_pack(
            st,
            title="Thesis packet promotion audit",
            description="Traceable promotion history for MD results that started as reporting-only and later entered the real AL loop.",
            rows=promotion_audit,
            key_prefix="results_packet_promotion_audit",
        )
        _render_export_pack(
            st,
            title="Thesis packet peptide provenance audit",
            description="One row per peptide showing source, MD path, review state, promotion bridge, ingest state, and any consistency flags.",
            rows=provenance_audit,
            key_prefix="results_packet_provenance_audit",
        )
        if replay_best:
            _render_export_pack(
                st,
                title="Thesis packet replay summary",
                description="Best replay-benchmark row per visible run.",
                rows=replay_best,
                key_prefix="results_packet_replay",
            )
        if discovery_best:
            _render_export_pack(
                st,
                title="Thesis packet discovery summary",
                description="Best discovery highlight row per visible run.",
                rows=discovery_best,
                key_prefix="results_packet_discovery",
            )
        _render_export_pack(
            st,
            title="Thesis packet decision log",
            description="Local-only thesis decisions recorded while choosing runs, discovery actions, or MD feedback handoffs.",
            rows=decision_rows,
            key_prefix="results_packet_decisions",
        )
        return

    if selected_section == "Thesis narrative":
        selected_runs, selected_names = _selected_results_runs(st, runs)
        narrative_rows = _focused_comparison_rows(selected_runs, metric_name=metric_name)
        narrative_callouts = _thesis_narrative_callout_rows(selected_runs, metric_name=metric_name, state=state)
        review_rows = _selected_review_audit_rows(state, selected_names)
        st.subheader("Thesis narrative")
        st.caption("Use this section when you are turning dashboard evidence into write-up language. It gives you a draft storyline, selected-run evidence, and figure/table callouts for the same runs.")
        _render_metric_cards(
            st,
            [
                ("Selected runs", len(selected_runs)),
                ("Selected with frozen final", sum(1 for run in selected_runs if run.get("final_metrics"))),
                ("Selected with discovery evidence", sum(1 for run in selected_runs if _frame_records(discovery_frame(run)))),
                ("Selected review blockers", sum(1 for row in review_rows if str(row.get("feedback_state", "")) == "Needs human review")),
            ],
        )
        st.markdown("#### Draft-ready narrative")
        st.code(_thesis_narrative_markdown(selected_runs, metric_name=metric_name, state=state), language="markdown")
        if narrative_rows:
            st.markdown("#### Narrative evidence snapshot")
            st.dataframe(narrative_rows)
        if narrative_callouts:
            st.markdown("#### Suggested figure and table callouts")
            st.dataframe(narrative_callouts)
            _render_export_pack(
                st,
                title="Narrative callout pack",
                description="Copy-ready figure/table callouts for the selected runs.",
                rows=narrative_callouts,
                key_prefix="results_narrative_callouts",
            )
        else:
            st.info("Select at least one run with reporting evidence to build figure and table callouts.")
        return

    if selected_section == "Appendix packet":
        comparison_runs, comparison_names = _selected_results_runs(st, runs)
        selected_runs_only = st.checkbox(
            "Use only the selected comparison runs in this appendix packet",
            value=False,
            key="results_appendix_selected_only",
        )
        report_ready_only = st.checkbox(
            "Keep only report-ready runs in this appendix packet",
            value=False,
            key="results_appendix_report_ready_only",
        )

        appendix_runs = list(comparison_runs) if selected_runs_only else list(runs)
        if report_ready_only:
            appendix_runs = [
                run
                for run in appendix_runs
                if _reporting_readiness_for_run(run)[0] == "Report-ready"
            ]
        appendix_run_names = [
            str(run.get("run_display_name", run.get("run_name", "")))
            for run in appendix_runs
        ]
        appendix_run_name_set = set(appendix_run_names)

        filtered_comparison_runs = [
            run
            for run in comparison_runs
            if not report_ready_only or _reporting_readiness_for_run(run)[0] == "Report-ready"
        ]
        scorecard_rows = _result_scorecard_rows(appendix_runs, metric_name=metric_name)
        summary_rows = _result_summary_rows(appendix_runs)
        milestone_rows = _thesis_milestone_rows(appendix_runs)
        comparison_rows = _focused_comparison_rows(filtered_comparison_runs, metric_name=metric_name)
        review_audit = [
            row for row in _review_feedback_audit_rows(state)
            if str(row.get("run", "")) in appendix_run_name_set
        ]
        promotion_audit = [
            row for row in _promotion_audit_rows(state)
            if str(row.get("run", "")) in appendix_run_name_set
        ]
        provenance_audit = [
            row for row in _peptide_provenance_audit_rows(state)
            if str(row.get("run", "")) in appendix_run_name_set
        ]
        decision_rows = _decision_log_rows(
            state,
            run_names=appendix_run_name_set if appendix_run_name_set else set(),
        )
        caption_rows = _figure_caption_rows(appendix_runs, metric_name=metric_name)
        replay_best = _replay_best_strategy_rows(appendix_runs, metric_name=metric_name)
        discovery_best = _discovery_highlight_rows(appendix_runs)
        appendix_section_rows = [
            {
                "section": "Run comparison scorecards",
                "rows": len(scorecard_rows),
                "why_it_belongs": "Core baseline/final comparison table for the appendix.",
            },
            {
                "section": "Selected-run comparison",
                "rows": len(comparison_rows),
                "why_it_belongs": "Focused side-by-side comparison for the thesis discussion set.",
            },
            {
                "section": "Thesis milestones",
                "rows": len(milestone_rows),
                "why_it_belongs": "Run lifecycle summary for benchmark, feedback, discovery, and frozen final status.",
            },
            {
                "section": "Review / feedback audit",
                "rows": len(review_audit),
                "why_it_belongs": "Human-review handoff and ingest readiness trace.",
            },
            {
                "section": "Promotion audit",
                "rows": len(promotion_audit),
                "why_it_belongs": "Bridge from reporting-only MD evidence into the real AL loop.",
            },
            {
                "section": "Peptide provenance audit",
                "rows": len(provenance_audit),
                "why_it_belongs": "One-row peptide history from suggestion through MD, review, promotion, and ingest.",
            },
            {
                "section": "Thesis decision log",
                "rows": len(decision_rows),
                "why_it_belongs": "Local reasoning trail for important thesis choices.",
            },
            {
                "section": "Figure captions",
                "rows": len(caption_rows),
                "why_it_belongs": "Draft-ready captions for figures that match the appendix evidence.",
            },
        ]

        st.subheader("Thesis appendix packet")
        st.caption("Use this workspace when you want one bundled thesis appendix export instead of hopping across scorecards, audits, provenance, and caption sections individually.")
        _render_metric_cards(
            st,
            [
                ("Runs in packet", len(appendix_runs)),
                ("Comparison focus runs", len(filtered_comparison_runs)),
                ("Report-ready runs", sum(1 for run in appendix_runs if _reporting_readiness_for_run(run)[0] == "Report-ready")),
                ("Audit rows", len(review_audit) + len(promotion_audit) + len(provenance_audit)),
                ("Decision rows", len(decision_rows)),
                ("Caption rows", len(caption_rows)),
            ],
        )
        if not appendix_runs:
            st.info("No runs match the current appendix packet filters.")
            return

        st.markdown("#### Appendix packet guide")
        st.dataframe(appendix_section_rows)
        st.markdown("#### Copy-ready appendix overview")
        st.code(
            _appendix_packet_markdown(
                appendix_runs,
                metric_name=metric_name,
                selected_runs_only=selected_runs_only,
                report_ready_only=report_ready_only,
                comparison_runs=filtered_comparison_runs,
                review_rows=review_audit,
                promotion_rows=promotion_audit,
                provenance_rows=provenance_audit,
                decision_rows=decision_rows,
                caption_rows=caption_rows,
            ),
            language="markdown",
        )
        st.markdown("#### Appendix comparison snapshot")
        st.dataframe(scorecard_rows)
        if comparison_rows:
            st.markdown("#### Appendix selected-run comparison")
            st.dataframe(comparison_rows)
        if caption_rows:
            st.markdown("#### Appendix figure captions")
            st.dataframe(caption_rows)

        _render_export_pack(
            st,
            title="Appendix packet section guide",
            description="Checklist of the appendix sections bundled below and why each one belongs in the packet.",
            rows=appendix_section_rows,
            key_prefix="results_appendix_sections",
        )
        _render_export_pack(
            st,
            title="Appendix packet run comparison",
            description="Master appendix comparison table for the currently included runs.",
            rows=summary_rows,
            key_prefix="results_appendix_summary",
        )
        _render_export_pack(
            st,
            title="Appendix packet scorecards",
            description="Selected metric scorecards for the currently included runs.",
            rows=scorecard_rows,
            key_prefix="results_appendix_scorecards",
        )
        _render_export_pack(
            st,
            title="Appendix packet selected-run comparison",
            description="Focused side-by-side comparison for the chosen discussion runs.",
            rows=comparison_rows,
            key_prefix="results_appendix_focused_comparison",
        )
        _render_export_pack(
            st,
            title="Appendix packet milestone table",
            description="Lifecycle status for benchmark, MD feedback, discovery, and frozen final readiness.",
            rows=milestone_rows,
            key_prefix="results_appendix_milestones",
        )
        _render_export_pack(
            st,
            title="Appendix packet review / feedback audit",
            description="Human-review and model-feedback traceability for the currently included runs.",
            rows=review_audit,
            key_prefix="results_appendix_review_audit",
        )
        _render_export_pack(
            st,
            title="Appendix packet promotion audit",
            description="Promotion bridge status for reporting-only MD results that may later enter the real AL loop.",
            rows=promotion_audit,
            key_prefix="results_appendix_promotion_audit",
        )
        _render_export_pack(
            st,
            title="Appendix packet peptide provenance audit",
            description="One-row peptide history from source through MD, review, promotion, ingest, and consistency checks.",
            rows=provenance_audit,
            key_prefix="results_appendix_provenance_audit",
        )
        _render_export_pack(
            st,
            title="Appendix packet thesis decision log",
            description="Local-only reasoning notes that explain why important run and peptide decisions were made.",
            rows=decision_rows,
            key_prefix="results_appendix_decisions",
        )
        _render_export_pack(
            st,
            title="Appendix packet figure captions",
            description="Draft-ready figure captions that match the currently included runs and metric focus.",
            rows=caption_rows,
            key_prefix="results_appendix_captions",
        )
        if replay_best:
            _render_export_pack(
                st,
                title="Appendix packet replay evidence",
                description="Best replay-benchmark row per run for the appendix packet.",
                rows=replay_best,
                key_prefix="results_appendix_replay",
            )
        if discovery_best:
            _render_export_pack(
                st,
                title="Appendix packet discovery evidence",
                description="Best discovery highlight row per run for the appendix packet.",
                rows=discovery_best,
                key_prefix="results_appendix_discovery",
            )
        return

    if selected_section == "Compare selected runs":
        selected_runs, _ = _selected_results_runs(st, runs)
        comparison_rows = _focused_comparison_rows(selected_runs, metric_name=metric_name)
        st.subheader("Compare selected runs")
        st.caption("Use this workspace when you want a side-by-side comparison of the exact thesis runs you are discussing right now, instead of all visible runs at once.")
        _render_metric_cards(
            st,
            [
                ("Selected runs", len(selected_runs)),
                ("Report-ready selected", sum(1 for run in selected_runs if _reporting_readiness_for_run(run)[0] == "Report-ready")),
                ("With discovery evidence", sum(1 for run in selected_runs if _frame_records(discovery_frame(run)))),
                ("With frozen final", sum(1 for run in selected_runs if run.get("final_metrics"))),
            ],
        )
        if comparison_rows:
            st.dataframe(comparison_rows)
            final_chart = _bar_chart_rows(comparison_rows, label_key="run", value_key=f"final_{metric_name}")
            delta_chart = _bar_chart_rows(comparison_rows, label_key="run", value_key=f"delta_{metric_name}")
            if final_chart:
                st.markdown(f"#### Selected-run final {metric_name}")
                st.bar_chart(final_chart)
            if delta_chart:
                st.markdown(f"#### Selected-run improvement in {metric_name}")
                st.bar_chart(delta_chart)
            _render_export_pack(
                st,
                title="Focused comparison table",
                description="Copy-ready side-by-side comparison for the runs selected above.",
                rows=comparison_rows,
                key_prefix="results_focused_comparison",
            )
        else:
            st.info("Pick at least one visible run to build a focused comparison.")
        return

    if selected_section == "Thesis milestones":
        milestone_rows = _thesis_milestone_rows(runs)
        st.subheader("Thesis milestones")
        st.caption("Use this tracker when you want one row per run showing where it sits in the thesis lifecycle: benchmark, proposal, MD feedback, discovery, and frozen final evaluation.")
        _render_metric_cards(
            st,
            [
                ("Runs with baseline", sum(1 for row in milestone_rows if row.get("baseline_ready") == "yes")),
                ("Runs with proposals", sum(1 for row in milestone_rows if row.get("batch_proposed") == "yes")),
                ("Runs with MD feedback ready", sum(1 for row in milestone_rows if row.get("md_feedback_ready") == "yes")),
                ("Runs with frozen final", sum(1 for row in milestone_rows if row.get("final_evaluated") == "yes")),
            ],
        )
        if milestone_rows:
            st.dataframe(milestone_rows)
        else:
            st.info("No milestone rows are available for the visible runs.")
        return

    if selected_section == "Thesis scorecards":
        scorecard_rows = _result_scorecard_rows(runs, metric_name=metric_name)
        st.subheader("Thesis scorecards")
        st.caption("These rows answer the core reporting question for each visible run: how mature is it, what is the current reporting status, and how does the selected metric compare between baseline and frozen final evaluation.")
        if scorecard_rows:
            st.dataframe(scorecard_rows)
        else:
            st.info("No scorecard rows are available for the visible runs.")
        report_ready = [
            {
                "run": str(run.get("run_display_name", run.get("run_name", ""))),
                "model_state": str(run.get("ml_status_label", run.get("ml_status", ""))),
                "reporting_readiness": _reporting_readiness_for_run(run)[0],
                "recommended_reporting_step": (
                    "Use this run in thesis reporting now."
                    if run.get("final_metrics")
                    else "Finish a frozen final evaluation before using this run as a final result."
                ),
            }
            for run in runs
        ]
        st.markdown("#### Reporting readiness")
        st.dataframe(report_ready)
        return

    if selected_section == "Figure-ready comparisons":
        st.subheader("Figure-ready comparisons")
        st.caption("Use this section when you want the fastest path from dashboard evidence to thesis figures. Each chart is framed around a concrete comparison question.")
        scorecard_rows = _result_scorecard_rows(runs, metric_name=metric_name)
        final_chart = _bar_chart_rows(scorecard_rows, label_key="run", value_key=f"final_{metric_name}")
        delta_chart = _bar_chart_rows(scorecard_rows, label_key="run", value_key=f"delta_{metric_name}")
        if final_chart:
            st.markdown(f"#### Frozen final {metric_name} by run")
            st.caption(f"Use this chart when you want to compare the final frozen {metric_name} across visible runs.")
            st.bar_chart(final_chart)
        if delta_chart:
            st.markdown(f"#### Improvement from baseline to final ({metric_name})")
            st.caption(f"Use this chart when you want to show which runs actually improved the chosen metric over the baseline starting point.")
            st.bar_chart(delta_chart)

        replay_best = _replay_best_strategy_rows(runs, metric_name=metric_name)
        replay_chart = _bar_chart_rows(replay_best, label_key="run", value_key=metric_name)
        if replay_chart:
            st.markdown("#### Best replay strategy score by run")
            st.caption("Use this to justify which acquisition strategy looked strongest on the initial dataset-only replay benchmark.")
            st.bar_chart(replay_chart)

        discovery_best = _discovery_highlight_rows(runs)
        discovery_chart = _bar_chart_rows(discovery_best, label_key="run", value_key="utility_score")
        if discovery_chart:
            st.markdown("#### Discovery shortlist utility by run")
            st.caption("Use this to compare how strong the best discovery shortlist looked for each run.")
            st.bar_chart(discovery_chart)

        readiness_counts = _reporting_readiness_counts(runs)
        readiness_chart = _bar_chart_rows(readiness_counts, label_key="readiness", value_key="count")
        if readiness_chart:
            st.markdown("#### Reporting readiness counts")
            st.caption("Use this to summarize how many visible runs are already report-ready versus still in progress.")
            st.bar_chart(readiness_chart)

        caption_rows = _figure_caption_rows(runs, metric_name=metric_name)
        if caption_rows:
            st.markdown("#### Suggested figure captions")
            st.dataframe(caption_rows)
        return

    if selected_section == "Metric history":
        metric_history = _concat_frames([_with_constant(run_metric_history(run), run_name=run.get("run_display_name", run.get("run_name", ""))) for run in runs])
        metric_rows = _frame_records(metric_history)
        if metric_rows:
            st.subheader("Baseline and final metric history")
            round_chart = _wide_chart(metric_rows, index_key="round_id", series_key="run_name", value_key=metric_name)
            if round_chart:
                st.line_chart(round_chart)
            labeled_chart = _wide_chart(metric_rows, index_key="labeled_count", series_key="run_name", value_key=metric_name)
            if labeled_chart:
                st.line_chart(labeled_chart)
            st.dataframe(metric_history)
        else:
            st.info("No baseline/final metric history found for the current runs.")
        return

    if selected_section == "Replay evidence":
        replay_frame = _concat_frames([_with_constant(replay_curve_frame(run), run_name=run.get("run_display_name", run.get("run_name", ""))) for run in runs])
        replay_rows = _frame_records(replay_frame)
        st.subheader("Replay benchmark evidence")
        st.caption("Use this section to justify why one acquisition strategy looked stronger than another on the initial dataset only.")
        best_rows = _replay_best_strategy_rows(runs, metric_name=metric_name)
        if best_rows:
            st.markdown("#### Best replay strategy per run")
            st.dataframe(best_rows)
        if replay_rows:
            replay_chart = _wide_chart(replay_rows, index_key="round_id", series_key="strategy", value_key=metric_name)
            if replay_chart:
                st.line_chart(replay_chart)
            st.dataframe(replay_frame)
        else:
            st.info("No replay summaries found for the current runs.")
        return

    if selected_section == "Discovery evidence":
        discovery_frames = _concat_frames([_with_constant(discovery_frame(run), run_name=run.get("run_display_name", run.get("run_name", ""))) for run in runs])
        discovery_rows = _frame_records(discovery_frames)
        st.subheader("Discovery / Bayesian Optimization Search")
        st.caption("Use this section to explain which discovery strategy produced the most promising candidate shortlist and which peptides it surfaced.")
        discovery_highlights = _discovery_highlight_rows(runs)
        if discovery_highlights:
            st.markdown("#### Best discovery highlight per run")
            st.dataframe(discovery_highlights)
        if discovery_rows:
            discovery_chart = _multi_metric_chart(
                discovery_rows,
                index_key="strategy",
                value_keys=["exported_count", "top_batch_mean_utility_score", "top_batch_mean_pred_std"],
                label_key="run_name",
            )
            if discovery_chart:
                st.bar_chart(discovery_chart)
            st.dataframe(discovery_frames)
        else:
            st.info("No discovery summaries found for the current runs.")
        return

    if selected_section == "Review / feedback audit":
        audit_rows = _review_feedback_audit_rows(state)
        st.subheader("Review / feedback audit")
        st.caption("Use this section to track the human side of the thesis loop: which full-analysis peptides were reviewed, which labels were saved, and which items have already been fed back into the model.")
        audit_counts: dict[str, int] = {}
        for row in audit_rows:
            key = str(row.get("feedback_state", ""))
            audit_counts[key] = audit_counts.get(key, 0) + 1
        if audit_rows:
            _render_metric_cards(
                st,
                [
                    ("Needs review", audit_counts.get("Needs human review", 0)),
                    ("Needs ingest CSV", audit_counts.get("Needs ingest CSV", 0)),
                    ("Ready for model ingest", audit_counts.get("Ready for model ingest", 0)),
                    ("Already ingested", audit_counts.get("Already ingested", 0)),
                ],
            )
            st.dataframe(audit_rows)
        else:
            st.info("No full-analysis review rows are visible for the current workspace / filters.")
        return

    if selected_section == "Promotion audit":
        promotion_rows = _promotion_audit_rows(state)
        st.subheader("Promotion audit")
        st.caption("Use this to trace the bridge between reporting-only MD campaigns and the real AL loop: what can be promoted now, what is still waiting, and what has already been rebound into a real proposed batch.")
        if promotion_rows:
            _render_metric_cards(
                st,
                [
                    ("Can promote now", sum(1 for row in promotion_rows if str(row.get("promotion_state", "")).startswith("Can promote now"))),
                    ("Waiting for real batch", sum(1 for row in promotion_rows if str(row.get("promotion_state", "")) == "Waiting for real proposed batch")),
                    ("Already promoted", sum(1 for row in promotion_rows if str(row.get("promotion_state", "")).startswith("Promoted into real batch"))),
                    ("AL-ingestable now", sum(1 for row in promotion_rows if str(row.get("ingest_support", "")) == "AL-ingestable")),
                ],
            )
            st.dataframe(promotion_rows)
            _render_export_pack(
                st,
                title="Promotion audit table",
                description="One row per promotion-relevant full-analysis peptide, including reporting-only blockers and already promoted campaigns.",
                rows=promotion_rows,
                key_prefix="results_promotion_audit",
            )
        else:
            st.info("No promotion-relevant full-analysis peptides are visible for the current workspace / filters.")
        return

    if selected_section == "Peptide provenance audit":
        provenance_rows = _peptide_provenance_audit_rows(state)
        st.subheader("Peptide provenance audit")
        st.caption("Use this as the full peptide-history ledger: where the peptide came from, how it moved through MD, what the human review decided, whether it was promoted into the AL loop, and whether any consistency checks need attention.")
        if provenance_rows:
            _render_metric_cards(
                st,
                [
                    ("Peptides in ledger", len(provenance_rows)),
                    ("Attention needed", sum(1 for row in provenance_rows if str(row.get("integrity_state", "")) == "Attention needed")),
                    ("MD in progress", sum(1 for row in provenance_rows if str(row.get("lifecycle_state", "")) == "MD in progress")),
                    ("Reviewed for reporting", sum(1 for row in provenance_rows if str(row.get("lifecycle_state", "")) == "Reviewed for reporting")),
                    ("Ready for ingest", sum(1 for row in provenance_rows if str(row.get("lifecycle_state", "")) == "Ready for ingest")),
                    ("Already ingested", sum(1 for row in provenance_rows if str(row.get("lifecycle_state", "")) == "Already ingested")),
                ],
            )
            st.dataframe(provenance_rows)
            _render_export_pack(
                st,
                title="Peptide provenance audit table",
                description="One row per run/sequence showing source, MD slate path, review/promotion/ingest status, and consistency checks.",
                rows=provenance_rows,
                key_prefix="results_provenance_audit",
            )
        else:
            st.info("No peptide provenance rows are visible for the current workspace / filters.")
        return

    if selected_section == "Thesis decision log":
        decision_rows = _decision_log_rows(state)
        st.subheader("Thesis decision log")
        st.caption("Use this when you want a durable trail of why you chose a run, froze a final result, validated a discovery shortlist, or overrode an MD review label.")
        if decision_rows:
            _render_metric_cards(
                st,
                [
                    ("Recorded decisions", len(decision_rows)),
                    ("Run-level decisions", sum(1 for row in decision_rows if str(row.get("scope", "")) == "run")),
                    ("Peptide-level decisions", sum(1 for row in decision_rows if str(row.get("scope", "")) == "peptide")),
                    ("Unique targets", len({str(row.get("target", "")) for row in decision_rows})),
                ],
            )
            st.dataframe(decision_rows)
            _render_export_pack(
                st,
                title="Thesis decision log table",
                description="Use this when you want a copy-friendly audit of the reasoning decisions made across runs and peptides.",
                rows=decision_rows,
                key_prefix="results_decision_log",
            )
        else:
            st.info("No thesis decisions have been recorded yet. Save one from Model Workflow or MD Validation to start the log.")
        return

    if selected_section == "Export-ready tables":
        rows = _result_summary_rows(runs)
        st.subheader("Export-ready tables")
        st.caption("These are the cleanest copy/paste tables for thesis writing: one overall comparison table, one replay summary table, and one discovery shortlist table.")
        replay_best = _replay_best_strategy_rows(runs, metric_name=metric_name)
        discovery_best = _discovery_highlight_rows(runs)
        milestone_rows = _thesis_milestone_rows(runs)
        review_audit = _review_feedback_audit_rows(state)
        promotion_audit = _promotion_audit_rows(state)
        provenance_audit = _peptide_provenance_audit_rows(state)
        decision_rows = _decision_log_rows(state)
        _render_export_pack(
            st,
            title="Run comparison table",
            description="Use this as the master comparison table across visible runs.",
            rows=rows,
            key_prefix="results_run_comparison",
        )
        _render_export_pack(
            st,
            title="Replay strategy summary table",
            description="Use this when you need one replay-benchmark row per run.",
            rows=replay_best,
            key_prefix="results_replay_summary",
        )
        _render_export_pack(
            st,
            title="Discovery shortlist summary table",
            description="Use this when you want one discovery highlight row per run, including the surfaced candidate sequences.",
            rows=discovery_best,
            key_prefix="results_discovery_summary",
        )
        _render_export_pack(
            st,
            title="Thesis milestone tracker table",
            description="Use this when you want one row per run showing exactly which thesis milestones are already complete.",
            rows=milestone_rows,
            key_prefix="results_milestone_tracker",
        )
        _render_export_pack(
            st,
            title="Review / feedback audit table",
            description="Use this when you want one row per full-analysis peptide showing the current human-review and model-feedback status.",
            rows=review_audit,
            key_prefix="results_review_audit",
        )
        _render_export_pack(
            st,
            title="Promotion audit table",
            description="Use this when you want one row per reporting-only/promotion-relevant peptide showing whether the result can enter the real AL loop yet.",
            rows=promotion_audit,
            key_prefix="results_promotion_audit_export",
        )
        _render_export_pack(
            st,
            title="Peptide provenance audit table",
            description="Use this when you want one row per peptide that traces origin, MD path, review outcome, promotion status, ingest state, and consistency flags.",
            rows=provenance_audit,
            key_prefix="results_provenance_audit_export",
        )
        _render_export_pack(
            st,
            title="Thesis decision log table",
            description="Use this when you want one row per recorded thesis decision across runs and peptides.",
            rows=decision_rows,
            key_prefix="results_decision_export",
        )
        return

    if selected_section == "MD status":
        ladder_frame = md_ladder_summary_frame(list(state.get("peptides", [])))
        ladder_rows = _frame_records(ladder_frame)
        if ladder_rows:
            ladder_chart = _wide_chart(ladder_rows, index_key="md_profile", series_key="job_root_status", value_key="count")
            if ladder_chart:
                st.bar_chart(ladder_chart)
            st.dataframe(ladder_frame)
        else:
            st.info("No MD campaigns found for the current filters.")
        return

    action_frame = action_timeline_frame(list(state.get("actions", [])))
    if not _frame_empty(action_frame):
        st.dataframe(action_frame)
    else:
        st.info("No dashboard actions recorded yet.")
