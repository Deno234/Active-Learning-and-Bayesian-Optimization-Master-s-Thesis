from __future__ import annotations

import argparse
import io
import json
import unittest
from pathlib import Path
from typing import Sequence

from active_learning_thesis.dashboard_actions import shutdown_dashboard_action_workers
from active_learning_thesis.dashboard_integrity import (
    format_dashboard_integrity_report,
    run_dashboard_integrity_check,
)

DEFAULT_DASHBOARD_SMOKE_TESTS: tuple[str, ...] = (
    "tests.test_dashboard_health",
    "tests.test_dashboard_remote",
    "tests.test_dashboard_profiles",
    "tests.test_dashboard_actions",
    "tests.test_dashboard_feedback",
    "tests.test_dashboard_md_recovery",
    "tests.test_dashboard_md_slate",
    "tests.test_dashboard_golden_workflows",
    "tests.test_dashboard_integrity",
    "tests.test_dashboard",
)


def run_dashboard_smoke(
    run_root: Path,
    *,
    refresh_seconds: int = 0,
    test_names: Sequence[str] | None = None,
) -> dict[str, object]:
    integrity_report = run_dashboard_integrity_check(run_root, refresh_seconds=refresh_seconds)
    selected_tests = tuple(str(name) for name in (test_names or DEFAULT_DASHBOARD_SMOKE_TESTS))
    stream = io.StringIO()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(list(selected_tests))
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    try:
        result = runner.run(suite)
    finally:
        shutdown_dashboard_action_workers()

    unit_report = {
        "test_names": list(selected_tests),
        "tests_run": int(result.testsRun),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(getattr(result, "skipped", [])),
        "successful": bool(result.wasSuccessful()),
        "output": stream.getvalue(),
    }
    overall_success = int(integrity_report.get("failure_count", 0)) == 0 and bool(unit_report["successful"])
    return {
        "run_root": str(run_root),
        "refresh_seconds": refresh_seconds,
        "integrity": integrity_report,
        "unit_tests": unit_report,
        "overall_success": overall_success,
    }


def format_dashboard_smoke_report(report: dict[str, object]) -> str:
    lines = [
        f"Dashboard smoke check for: {report.get('run_root', '')}",
        "",
        format_dashboard_integrity_report(report.get("integrity", {})),
        "",
    ]
    unit_tests = report.get("unit_tests", {})
    if isinstance(unit_tests, dict):
        test_names = unit_tests.get("test_names", [])
        tests_run = int(unit_tests.get("tests_run", 0))
        failures = int(unit_tests.get("failures", 0))
        errors = int(unit_tests.get("errors", 0))
        skipped = int(unit_tests.get("skipped", 0))
        lines.extend(
            [
                "Dashboard-targeted test suites:",
                f"- Suites: {', '.join(str(name) for name in test_names)}",
                f"- Tests run: {tests_run}",
                f"- Failures: {failures}",
                f"- Errors: {errors}",
                f"- Skipped: {skipped}",
            ]
        )
        if failures or errors:
            output = str(unit_tests.get("output", "")).strip()
            if output:
                lines.extend(["", "Test runner output:", output])
    lines.extend(
        [
            "",
            "Overall result: " + ("OK" if bool(report.get("overall_success")) else "FAILED"),
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the dashboard integrity matrix plus the key dashboard-focused test suites.",
    )
    parser.add_argument("--run-root", default="active_learning_runs")
    parser.add_argument("--refresh-seconds", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print the full smoke report as JSON.")
    parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Optional unittest module names to run instead of the default dashboard smoke suites.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_dashboard_smoke(
        Path(args.run_root),
        refresh_seconds=args.refresh_seconds,
        test_names=args.tests,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_dashboard_smoke_report(report))
    return 0 if bool(report.get("overall_success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
