from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from active_learning_thesis.dashboard import collect_dashboard_state, render_dashboard
from active_learning_thesis.dashboard_md_view import MD_SECTIONS
from active_learning_thesis.dashboard_model_view import MODEL_SECTIONS
from active_learning_thesis.dashboard_operations_view import OPERATIONS_SECTIONS
from active_learning_thesis.dashboard_peptides_view import PEPTIDE_BUCKET_SPECS
from active_learning_thesis.dashboard_preferences import DASHBOARD_UI_MODES
from active_learning_thesis.dashboard_results_view import ANALYTICS_SECTIONS


class _FakeStreamlit:
    class _QueryParams(dict):
        def __init__(self, initial: dict[str, str] | None = None):
            super().__init__(initial or {})
            self.update_calls: list[dict[str, str]] = []

        def update(self, other=None, **kwargs):
            payload: dict[str, str] = {}
            if other:
                payload.update({str(key): str(value) for key, value in dict(other).items()})
            if kwargs:
                payload.update({str(key): str(value) for key, value in kwargs.items()})
            self.update_calls.append(payload)
            return super().update(payload)

    def __init__(
        self,
        view: str,
        *,
        radio_values: dict[str, str] | None = None,
        select_values: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
    ):
        self.sidebar = self
        self._view = view
        self._radio_values = radio_values or {}
        self._select_values = select_values or {}
        self.calls: list[tuple[str, object]] = []
        self.session_state: dict[str, object] = {}
        self.query_params = self._QueryParams(query_params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, value):
        self.calls.append(("title", value))

    def caption(self, value):
        self.calls.append(("caption", value))

    def header(self, value):
        self.calls.append(("header", value))

    def subheader(self, value):
        self.calls.append(("subheader", value))

    def write(self, value):
        self.calls.append(("write", value))

    def markdown(self, value, unsafe_allow_html=False):
        self.calls.append(("markdown", (value, unsafe_allow_html)))

    def info(self, value):
        self.calls.append(("info", value))

    def success(self, value):
        self.calls.append(("success", value))

    def error(self, value):
        self.calls.append(("error", value))

    def warning(self, value):
        self.calls.append(("warning", value))

    def metric(self, label, value):
        self.calls.append(("metric", (label, value)))

    def dataframe(self, value):
        self.calls.append(("dataframe", value))

    def json(self, value):
        self.calls.append(("json", value))

    def code(self, value, language=None):
        self.calls.append(("code", (value, language)))

    def line_chart(self, value):
        self.calls.append(("line_chart", value))

    def bar_chart(self, value):
        self.calls.append(("bar_chart", value))

    def divider(self):
        self.calls.append(("divider", None))

    def rerun(self):
        self.calls.append(("rerun", None))

    def columns(self, spec):
        self.calls.append(("columns", spec))
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    def tabs(self, names):
        self.calls.append(("tabs", tuple(names)))
        return [self for _ in names]

    def expander(self, label, expanded=False):
        self.calls.append(("expander", (label, expanded)))
        return self

    def container(self):
        self.calls.append(("container", None))
        return self

    def empty(self):
        self.calls.append(("empty", None))
        return self

    def radio(self, label, options, index=0, key=None, **kwargs):
        self.calls.append(("radio", (label, options, key)))
        chosen = self._radio_values.get(key or label)
        if chosen in options:
            if key is not None:
                self.session_state[key] = chosen
            return chosen
        if key == "dashboard_view" and self._view in options:
            self.session_state[key] = self._view
            return self._view
        if key is not None:
            state_value = self.session_state.get(key)
            if state_value in options:
                return state_value
        return self._view if self._view in options else options[index]

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        self.calls.append(("selectbox", (label, options, key)))
        chosen = self._select_values.get(key or label)
        if chosen in options:
            if key is not None:
                self.session_state[key] = chosen
            return chosen
        if key is not None:
            state_value = self.session_state.get(key)
            if state_value in options:
                return state_value
        return options[index] if options else ""

    def checkbox(self, label, value=False, key=None, **kwargs):
        self.calls.append(("checkbox", (label, key)))
        return value

    def button(self, label, key=None, **kwargs):
        self.calls.append(("button", (label, key)))
        return False

    def text_input(self, label, value="", key=None, **kwargs):
        self.calls.append(("text_input", (label, value, key)))
        return value

    def text_area(self, label, value="", key=None, **kwargs):
        self.calls.append(("text_area", (label, value, key)))
        return value


@dataclass(frozen=True)
class IntegrityScenario:
    name: str
    view: str
    ui_mode: str
    radio_values: dict[str, str]
    query_params: dict[str, str]


def _first_run_name(state: dict[str, object]) -> str:
    runs = state.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return ""
    run = runs[0]
    if not isinstance(run, dict):
        return ""
    return str(run.get("run_display_name", run.get("run_name", "")))


def _first_peptide_sequence(state: dict[str, object]) -> str:
    peptides = state.get("peptides", [])
    if not isinstance(peptides, list) or not peptides:
        return ""
    ladder = peptides[0]
    if not isinstance(ladder, dict):
        return ""
    return str(ladder.get("sequence", ""))


def build_dashboard_integrity_scenarios(state: dict[str, object]) -> list[IntegrityScenario]:
    run_name = _first_run_name(state)
    peptide = _first_peptide_sequence(state)
    peptide_sections = [spec[0] for spec in PEPTIDE_BUCKET_SPECS]
    scenarios: list[IntegrityScenario] = []
    for ui_mode in DASHBOARD_UI_MODES:
        scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Today", view="Today", ui_mode=ui_mode, radio_values={}, query_params={}))
        scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Results", view="Results", ui_mode=ui_mode, radio_values={}, query_params={}))
        scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Peptides", view="Peptides", ui_mode=ui_mode, radio_values={}, query_params={}))
        scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Operations", view="Operations", ui_mode=ui_mode, radio_values={}, query_params={}))
        for section in ANALYTICS_SECTIONS:
            scenarios.append(
                IntegrityScenario(
                    name=f"{ui_mode} :: Results / {section}",
                    view="Results",
                    ui_mode=ui_mode,
                    radio_values={"dashboard_results_section": section},
                    query_params={},
                )
            )
        for section in peptide_sections:
            scenarios.append(
                IntegrityScenario(
                    name=f"{ui_mode} :: Peptides / {section}",
                    view="Peptides",
                    ui_mode=ui_mode,
                    radio_values={"dashboard_peptides_section": section},
                    query_params={},
                )
            )
        for section in OPERATIONS_SECTIONS:
            scenarios.append(
                IntegrityScenario(
                    name=f"{ui_mode} :: Operations / {section}",
                    view="Operations",
                    ui_mode=ui_mode,
                    radio_values={"dashboard_operations_section": section},
                    query_params={},
                )
            )
        model_query = {"run_detail": run_name} if run_name else {}
        md_query = {"peptide": peptide} if peptide else {}
        if run_name:
            scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Model Workflow", view="Model Workflow", ui_mode=ui_mode, radio_values={}, query_params=model_query))
            for section in MODEL_SECTIONS:
                scenarios.append(
                    IntegrityScenario(
                        name=f"{ui_mode} :: Model Workflow / {section}",
                        view="Model Workflow",
                        ui_mode=ui_mode,
                        radio_values={"dashboard_model_section": section},
                        query_params=model_query,
                    )
                )
        else:
            scenarios.append(IntegrityScenario(name=f"{ui_mode} :: Model Workflow", view="Model Workflow", ui_mode=ui_mode, radio_values={}, query_params={}))
        if peptide:
            scenarios.append(IntegrityScenario(name=f"{ui_mode} :: MD Validation", view="MD Validation", ui_mode=ui_mode, radio_values={}, query_params=md_query))
            for section in MD_SECTIONS:
                scenarios.append(
                    IntegrityScenario(
                        name=f"{ui_mode} :: MD Validation / {section}",
                        view="MD Validation",
                        ui_mode=ui_mode,
                        radio_values={"dashboard_md_section": section},
                        query_params=md_query,
                    )
                )
        else:
            scenarios.append(IntegrityScenario(name=f"{ui_mode} :: MD Validation", view="MD Validation", ui_mode=ui_mode, radio_values={}, query_params={}))
    return scenarios


def run_dashboard_integrity_check(run_root: Path, *, refresh_seconds: int = 0) -> dict[str, object]:
    state = collect_dashboard_state(run_root)
    scenarios = build_dashboard_integrity_scenarios(state)
    failures: list[dict[str, str]] = []
    passed: list[str] = []
    for scenario in scenarios:
        st = _FakeStreamlit(
            scenario.view,
            radio_values=scenario.radio_values,
            query_params=scenario.query_params,
        )
        st.session_state["dashboard_ui_mode"] = scenario.ui_mode
        st.session_state["dashboard_skip_preference_save"] = True
        try:
            render_dashboard(st, state, refresh_seconds=refresh_seconds)
            passed.append(scenario.name)
        except Exception as exc:  # pragma: no cover - exercised in failure mode
            failures.append(
                {
                    "scenario": scenario.name,
                    "view": scenario.view,
                    "ui_mode": scenario.ui_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "run_root": str(run_root),
        "generated_at": str(state.get("generated_at", "")),
        "scenario_count": len(scenarios),
        "passed_count": len(passed),
        "failure_count": len(failures),
        "passed": passed,
        "failures": failures,
        "counts": {
            "runs": len(state.get("runs", [])),
            "peptides": len(state.get("peptides", [])),
            "actions": len(state.get("actions", [])),
            "notifications": len(state.get("notifications", [])),
        },
    }


def format_dashboard_integrity_report(report: dict[str, object]) -> str:
    lines = [
        f"Dashboard integrity check for: {report.get('run_root', '')}",
        f"Scenarios passed: {report.get('passed_count', 0)}/{report.get('scenario_count', 0)}",
    ]
    counts = report.get("counts", {})
    if isinstance(counts, dict):
        lines.append(
            "Visible state: "
            f"runs={counts.get('runs', 0)} "
            f"peptides={counts.get('peptides', 0)} "
            f"actions={counts.get('actions', 0)} "
            f"notifications={counts.get('notifications', 0)}"
        )
    failures = report.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.append("Failures:")
        for failure in failures:
            if isinstance(failure, dict):
                lines.append(f"- {failure.get('scenario', 'scenario')}: {failure.get('error', 'unknown error')}")
    else:
        lines.append("No render failures detected across the covered dashboard scenarios.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local dashboard integrity smoke check.")
    parser.add_argument("--run-root", default="active_learning_runs")
    parser.add_argument("--refresh-seconds", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print the full integrity report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_dashboard_integrity_check(Path(args.run_root), refresh_seconds=args.refresh_seconds)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_dashboard_integrity_report(report))
    return 0 if int(report.get("failure_count", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
