from __future__ import annotations

import argparse
from pathlib import Path

from active_learning_thesis.dashboard_actions import execute_action_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one queued dashboard action.")
    parser.add_argument("--action-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = execute_action_file(Path(args.action_file))
    status = str(result.get("status", "failed"))
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
