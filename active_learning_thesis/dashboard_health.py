from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from active_learning_thesis.dashboard_profiles import SUPPORTED_CLUSTERS, get_cluster_profile
from active_learning_thesis.dashboard_remote_state import load_cluster_health, save_cluster_health

HealthResult = dict[str, object]
Runner = Callable[..., subprocess.CompletedProcess[str]]

NETWORK_FAILURE_HINTS = (
    "connection timed out",
    "operation timed out",
    "timed out",
    "no route to host",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "name or service not known",
    "temporary failure in name resolution",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_details(stdout: str, stderr: str) -> list[str]:
    details: list[str] = []
    if stdout.strip():
        details.append(f"stdout: {stdout.strip()}")
    if stderr.strip():
        details.append(f"stderr: {stderr.strip()}")
    return details


def _result(status: str, summary: str, *, hint: str = "", stdout: str = "", stderr: str = "") -> HealthResult:
    return {
        "status": status,
        "summary": summary,
        "hint": hint,
        "details": _normalize_details(stdout, stderr),
    }


def interpret_ssh_add_probe(returncode: int, stdout: str, stderr: str) -> HealthResult:
    combined = f"{stdout}\n{stderr}".strip().lower()
    if returncode == 0 and stdout.strip():
        return _result(
            "ok",
            "OpenSSH agent has at least one loaded identity.",
            hint="OpenSSH agent readiness looks good for remote dashboard actions.",
            stdout=stdout,
            stderr=stderr,
        )
    if "the agent has no identities" in combined:
        return _result(
            "warning",
            "OpenSSH agent has no loaded identity.",
            hint="Run `ssh-add <private-key>` before remote SUPEK actions.",
            stdout=stdout,
            stderr=stderr,
        )
    if (
        "could not open a connection to your authentication agent" in combined
        or "error connecting to agent" in combined
        or "communication with agent failed" in combined
        or "service has not been started" in combined
        or "cannot find the file specified" in combined
    ):
        return _result(
            "warning",
            "OpenSSH agent is unavailable.",
            hint="Start the Windows `ssh-agent` service, then run `ssh-add` and re-check SUPEK health.",
            stdout=stdout,
            stderr=stderr,
        )
    if "not recognized as an internal or external command" in combined:
        return _result(
            "error",
            "`ssh-add` is not available on this machine.",
            hint="Install or enable Windows OpenSSH client support before using dashboard remote actions.",
            stdout=stdout,
            stderr=stderr,
        )
    return _result(
        "warning",
        "OpenSSH agent readiness is unclear.",
        hint="Run `ssh-add -l` in a terminal and check whether your key is loaded.",
        stdout=stdout,
        stderr=stderr,
    )


def interpret_remote_probe(cluster: str, returncode: int, stdout: str, stderr: str) -> HealthResult:
    combined = f"{stdout}\n{stderr}".strip()
    lowered = combined.lower()
    cluster_label = cluster.upper()
    if returncode == 0:
        host_value = stdout.strip().splitlines()[0] if stdout.strip() else "probe succeeded"
        return _result(
            "ok",
            f"{cluster_label} is reachable non-interactively over SSH.",
            hint=f"Remote {cluster_label} actions look ready.",
            stdout=host_value,
            stderr=stderr,
        )
    if "permission denied" in lowered:
        return _result(
            "warning",
            f"{cluster_label} is reachable, but SSH authentication failed.",
            hint=f"Check your loaded key and `~/.ssh/config`, then re-check {cluster_label} health.",
            stdout=stdout,
            stderr=stderr,
        )
    if "host key verification failed" in lowered:
        return _result(
            "warning",
            f"{cluster_label} host key is not trusted yet.",
            hint=f"Connect once with plain `ssh {cluster}` to accept the host key, then re-check.",
            stdout=stdout,
            stderr=stderr,
        )
    if "could not resolve hostname" in lowered:
        return _result(
            "error",
            f"{cluster_label} SSH alias or host could not be resolved.",
            hint=f"Check the `{cluster}` host entry in `~/.ssh/config` and your cluster profile, then re-check.",
            stdout=stdout,
            stderr=stderr,
        )
    if any(token in lowered for token in NETWORK_FAILURE_HINTS):
        hint = (
            "BURA is unreachable non-interactively. If FortiClient is required, connect VPN and re-check."
            if cluster == "bura"
            else "SUPEK is unreachable non-interactively. Check network access and your SSH alias, then re-check."
        )
        return _result(
            "warning",
            f"{cluster_label} is unreachable non-interactively.",
            hint=hint,
            stdout=stdout,
            stderr=stderr,
        )
    return _result(
        "warning",
        f"{cluster_label} health probe failed.",
        hint=f"Review the SSH output and re-check {cluster_label} health.",
        stdout=stdout,
        stderr=stderr,
    )


def _run_probe(command: list[str], *, timeout_seconds: int, runner: Runner) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _probe_local_auth(*, timeout_seconds: int, runner: Runner) -> HealthResult:
    try:
        completed = _run_probe(["ssh-add", "-l"], timeout_seconds=timeout_seconds, runner=runner)
    except FileNotFoundError as exc:
        return _result(
            "error",
            "`ssh-add` is not available on this machine.",
            hint="Install or enable Windows OpenSSH client support before using dashboard remote actions.",
            stderr=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            "warning",
            "OpenSSH agent probe timed out.",
            hint="Try `ssh-add -l` in a terminal, then re-check SUPEK health.",
            stderr=str(exc),
        )
    return interpret_ssh_add_probe(completed.returncode, completed.stdout, completed.stderr)


def _probe_remote(cluster: str, profile: dict[str, str], *, timeout_seconds: int, runner: Runner) -> HealthResult:
    ssh_target = f"{profile['username']}@{profile['host']}"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout_seconds}",
        ssh_target,
        "hostname",
    ]
    try:
        completed = _run_probe(command, timeout_seconds=timeout_seconds, runner=runner)
    except FileNotFoundError as exc:
        return _result(
            "error",
            "`ssh` is not available on this machine.",
            hint="Install or enable Windows OpenSSH client support before using dashboard remote actions.",
            stderr=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return interpret_remote_probe(cluster, 255, "", str(exc))
    return interpret_remote_probe(cluster, completed.returncode, completed.stdout, completed.stderr)


def _status_rank(status: str) -> int:
    order = {"unknown": 0, "ok": 1, "warning": 2, "error": 3}
    return order.get(status, 0)


def _combine_cluster_health(cluster: str, local_auth: HealthResult | None, remote_status: HealthResult) -> dict[str, object]:
    pieces = [item for item in [local_auth, remote_status] if isinstance(item, dict)]
    overall = "ok"
    for piece in pieces:
        piece_status = str(piece.get("status", "unknown"))
        if _status_rank(piece_status) > _status_rank(overall):
            overall = piece_status
    summary_source = next((piece for piece in pieces if str(piece.get("status", "unknown")) != "ok"), remote_status)
    details: list[str] = []
    for piece in pieces:
        details.extend(str(item) for item in piece.get("details", []) if str(item).strip())
    return {
        "cluster": cluster,
        "checked_at": _now_iso(),
        "overall_status": overall,
        "local_auth_status": local_auth or {},
        "remote_status": remote_status,
        "summary": str(summary_source.get("summary", f"{cluster.upper()} health is {overall}.")),
        "hint": str(summary_source.get("hint", "")),
        "details": details,
    }


def check_cluster_health(
    run_root: Path,
    cluster: str,
    profile: dict[str, str],
    *,
    timeout_seconds: int = 5,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    local_auth = _probe_local_auth(timeout_seconds=timeout_seconds, runner=runner) if cluster == "supek" else {}
    remote_status = _probe_remote(cluster, profile, timeout_seconds=timeout_seconds, runner=runner)
    payload = _combine_cluster_health(cluster, local_auth if cluster == "supek" else None, remote_status)
    save_cluster_health(run_root, cluster, payload)
    return payload


def check_all_cluster_health(
    run_root: Path,
    profiles_payload: dict[str, object],
    *,
    timeout_seconds: int = 5,
    runner: Runner = subprocess.run,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for cluster in SUPPORTED_CLUSTERS:
        profile = get_cluster_profile(profiles_payload, cluster)
        if profile is None:
            cached = load_cluster_health(run_root, cluster)
            if str(cached.get("cluster", "")) == cluster and str(cached.get("checked_at", "")):
                results.append(cached)
            continue
        results.append(check_cluster_health(run_root, cluster, profile, timeout_seconds=timeout_seconds, runner=runner))
    return results
