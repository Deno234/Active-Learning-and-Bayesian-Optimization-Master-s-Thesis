from __future__ import annotations

import argparse
from pathlib import Path

from active_learning_thesis.dashboard import collect_dashboard_state, render_dashboard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local Streamlit dashboard.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--refresh-seconds", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    import streamlit as st

    state = collect_dashboard_state(Path(args.run_root))
    render_dashboard(st, state, refresh_seconds=args.refresh_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
