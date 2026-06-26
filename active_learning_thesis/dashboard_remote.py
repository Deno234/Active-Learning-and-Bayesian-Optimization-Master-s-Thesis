from __future__ import annotations

import json
import re
from dataclasses import dataclass
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

from active_learning_thesis.dashboard_actions import (
    draft_dashboard_action,
    serialize_action_command,
    submit_dashboard_action,
)
from active_learning_thesis.dashboard_preferences import (
    DEFAULT_DASHBOARD_APPROVAL_MODE,
    load_dashboard_preferences,
)
from active_learning_thesis.dashboard_remote_state import (
    downloads_root,
    uploads_root,
    save_cluster_snapshot,
    update_sync_status,
)

REMOTE_MUTATING_KINDS = {
    "supek-sync-repo",
    "supek-sync-run",
    "supek-submit-workflow",
    "supek-submit-study",
    "supek-submit-study-array",
    "supek-cancel-job",
    "supek-pull-artifacts",
    "bura-upload-campaign",
    "bura-normalize-scripts",
    "bura-preflight",
    "bura-submit-chain",
    "bura-full-autopilot",
    "bura-md-workflow",
    "bura-cancel-job",
    "bura-pull-package",
}
TRUSTED_REMOTE_ACTION_KINDS = {
    "supek-sync-run",
    "supek-submit-workflow",
    "supek-submit-study",
    "supek-submit-study-array",
    "supek-pull-artifacts",
    "bura-upload-campaign",
    "bura-normalize-scripts",
    "bura-preflight",
    "bura-submit-chain",
    "bura-full-autopilot",
    "bura-md-workflow",
    "bura-pull-package",
}
DESTRUCTIVE_REMOTE_ACTION_KINDS = {
    "supek-cancel-job",
    "bura-cancel-job",
}
REMOTE_READONLY_KINDS = {
    "supek-verify-env",
    "supek-submit-preflight",
    "supek-poll-qstat",
    "supek-fetch-logs",
    "bura-submit-readiness",
    "bura-reconcile-campaign",
    "bura-poll-squeue",
    "bura-inspect-logs",
}
POLL_KINDS = {"supek-poll-qstat", "bura-poll-squeue"}
SUPEK_WORKFLOW_COMMANDS = {"init-run", "run-replay", "propose-round", "run-discovery", "evaluate-final", "ingest-round"}
BURA_SYNC_STATUS_ORDER = [
    "not_synced",
    "staged_remote",
    "submitted",
    "running",
    "outputs_staged",
    "outputs_returned",
    "finalized_local",
]
_QSUB_JOB_RE = re.compile(r"(?P<job_id>\d+(?:\.[A-Za-z0-9_.-]+)?)")
_SBATCH_JOB_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")
_CHAIN_JOB_RE = re.compile(r"->\s*(?P<job_id>\d+)")


@dataclass(frozen=True)
class SupekActionPayload:
    """Shared SUPEK action payload used for both preview and submission."""

    run_root: Path
    title: str
    kind: str
    cluster: str
    command: list[str]
    cwd: Path
    related_run: str = ""
    related_sequence: str = ""
    related_campaign: str = ""
    output_path: str | Path | None = None
    metadata: dict[str, object] | None = None
    requires_approval: bool = True

    @property
    def display_command(self) -> str:
        return serialize_action_command(self.command[0], self.command[1:])


def draft_supek_payload_action(payload: SupekActionPayload) -> dict[str, object]:
    return _queue_or_draft_remote_action(
        run_root=payload.run_root,
        title=payload.title,
        kind=payload.kind,
        cluster=payload.cluster,
        command=payload.command,
        cwd=payload.cwd,
        related_run=payload.related_run,
        related_sequence=payload.related_sequence,
        related_campaign=payload.related_campaign,
        output_path=payload.output_path,
        metadata=payload.metadata,
        requires_approval=payload.requires_approval,
    )


def should_require_remote_approval(
    *,
    kind: str,
    requested_requires_approval: bool,
    approval_mode: str = DEFAULT_DASHBOARD_APPROVAL_MODE,
) -> bool:
    if not requested_requires_approval:
        return False
    if kind in DESTRUCTIVE_REMOTE_ACTION_KINDS:
        return True
    if approval_mode == "Trusted actions" and kind in TRUSTED_REMOTE_ACTION_KINDS:
        return False
    return True


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")



def _ssh_target(profile: dict[str, str]) -> str:
    return f"{profile['username']}@{profile['host']}"



def _posix_join(*parts: str) -> str:
    cleaned = [part for part in parts if str(part).strip()]
    if not cleaned:
        return ""
    current = PurePosixPath(cleaned[0])
    for part in cleaned[1:]:
        current = current / part
    return str(current)



def _posix_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _expand_remote_profile_tokens(profile: dict[str, str], value: str | Path) -> str:
    raw = str(value).replace('\\', '/')
    username = str(profile.get('username', '')).strip()
    if username:
        raw = raw.replace('${USER}', username).replace('$USER', username)
    return raw


def _remote_shell_path(value: str | Path, profile: dict[str, str] | None = None) -> str:
    raw = _expand_remote_profile_tokens(profile, value) if profile is not None else str(value).replace('\\', '/')
    if raw == '~':
        return '$HOME'
    if raw.startswith('~/'):
        rest = raw[2:].replace('\\', '/')
        escaped = rest.replace('"', '\"').replace('$', '\$').replace('`', '\`')
        return f'"$HOME/{escaped}"' if escaped else '$HOME'
    return _posix_quote(raw)


def _build_ssh_command(profile: dict[str, str], remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        _ssh_target(profile),
        f"bash -lc {shlex.quote(remote_command)}",
    ]


def _emit_remote_check(test_command: str, ok_token: str, missing_token: str) -> str:
    return f"if {test_command}; then echo {ok_token}; else echo {missing_token}; fi"


def _build_scp_upload_command(profile: dict[str, str], local_path: Path, remote_path: str) -> list[str]:
    return [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-r",
        str(local_path.resolve()),
        f"{_ssh_target(profile)}:{remote_path}",
    ]



def _build_scp_download_command(profile: dict[str, str], remote_path: str, local_path: Path) -> list[str]:
    return [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-r",
        f"{_ssh_target(profile)}:{remote_path}",
        str(local_path.resolve()),
    ]



def _queue_or_draft_remote_action(
    *,
    run_root: Path,
    title: str,
    kind: str,
    cluster: str,
    command: list[str],
    cwd: Path,
    related_run: str = "",
    related_sequence: str = "",
    related_campaign: str = "",
    output_path: str | Path | None = None,
    metadata: dict[str, object] | None = None,
    requires_approval: bool,
    launch_worker: bool = True,
) -> dict[str, object]:
    preferences = load_dashboard_preferences(run_root)
    approval_mode = (
        str(preferences.get("approval_mode", DEFAULT_DASHBOARD_APPROVAL_MODE))
        if bool(preferences.get("exists", False))
        else "Strict approvals"
    )
    effective_requires_approval = should_require_remote_approval(
        kind=kind,
        requested_requires_approval=requires_approval,
        approval_mode=approval_mode,
    )
    action_metadata = dict(metadata or {})
    if requires_approval and not effective_requires_approval:
        action_metadata["approval_mode"] = approval_mode
        action_metadata["trusted_inline_approval"] = True
    kwargs = dict(
        run_root=run_root,
        title=title,
        kind=kind,
        command=command[0],
        args=command[1:],
        cwd=cwd,
        scope=cluster,
        cluster=cluster,
        related_run=related_run,
        related_sequence=related_sequence,
        related_campaign=related_campaign,
        output_path=output_path,
        display_command=serialize_action_command(command[0], command[1:]),
        launch_worker=launch_worker,
        metadata=action_metadata,
        exclusive=kind not in REMOTE_READONLY_KINDS and kind not in POLL_KINDS,
    )
    if effective_requires_approval:
        return draft_dashboard_action(**kwargs)
    return submit_dashboard_action(**kwargs)



def extract_qsub_job_id(text: str) -> str:
    for line in text.splitlines():
        match = _QSUB_JOB_RE.fullmatch(line.strip())
        if match:
            return match.group("job_id")
    for line in text.splitlines():
        match = _QSUB_JOB_RE.search(line.strip())
        if match:
            return match.group("job_id")
    return ""


def extract_qsub_job_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        label_value = line.split("=", 1)[-1].strip() if "=" in line else line.strip()
        for candidate in [label_value, line.strip()]:
            match = _QSUB_JOB_RE.fullmatch(candidate)
            if match:
                job_id = match.group("job_id")
                if job_id not in seen:
                    ids.append(job_id)
                    seen.add(job_id)
                break
    if ids:
        return ids
    for match in _QSUB_JOB_RE.finditer(text):
        job_id = match.group("job_id")
        if job_id not in seen:
            ids.append(job_id)
            seen.add(job_id)
    return ids



def extract_sbatch_job_ids(text: str) -> list[str]:
    ids = [match.group("job_id") for match in _SBATCH_JOB_RE.finditer(text)]
    ids.extend(match.group("job_id") for match in _CHAIN_JOB_RE.finditer(text))
    deduped: list[str] = []
    seen: set[str] = set()
    for job_id in ids:
        if job_id not in seen:
            deduped.append(job_id)
            seen.add(job_id)
    return deduped



def parse_qstat_output(text: str) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("job") or line.startswith("---"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        jobs.append(
            {
                "job_id": parts[0],
                "name": parts[1],
                "user": parts[2] if len(parts) > 2 else "",
                "state": parts[-2] if len(parts) >= 2 else "",
                "queue": parts[-1] if len(parts) >= 1 else "",
                "raw": raw_line,
            }
        )
    return jobs



def parse_squeue_output(text: str) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("JOBID"):
            continue
        parts = line.split(maxsplit=7)
        if len(parts) < 7:
            continue
        jobs.append(
            {
                "job_id": parts[0],
                "partition": parts[1] if len(parts) > 1 else "",
                "name": parts[2] if len(parts) > 2 else "",
                "user": parts[3] if len(parts) > 3 else "",
                "state": parts[4] if len(parts) > 4 else "",
                "time": parts[5] if len(parts) > 5 else "",
                "nodes": parts[6] if len(parts) > 6 else "",
                "reason": parts[7] if len(parts) > 7 else "",
                "raw": raw_line,
            }
        )
    return jobs



def summarize_cluster_jobs(jobs: Iterable[dict[str, str]]) -> dict[str, int]:
    summary = {"pending": 0, "running": 0, "failed": 0, "held": 0}
    for job in jobs:
        state = str(job.get("state", "")).upper()
        reason = str(job.get("reason", "")).lower()
        if state == "R":
            summary["running"] += 1
        elif state in {"F", "CA", "NF", "TO"}:
            summary["failed"] += 1
        elif state == "PD":
            summary["pending"] += 1
            if "held" in reason or "dependencynever" in reason:
                summary["held"] += 1
        else:
            summary["pending"] += 1
    return summary



def _remote_run_dir(profile: dict[str, str], run_dir: Path) -> str:
    return _posix_join(_expand_remote_profile_tokens(profile, profile.get("scratch_run_root", "")), run_dir.name)



def _remote_log_dir(profile: dict[str, str], run_dir: Path) -> str:
    return _posix_join(_expand_remote_profile_tokens(profile, profile.get("log_root", "")), run_dir.name)



def _remote_campaign_dir(profile: dict[str, str], campaign_dir: Path) -> str:
    return _posix_join(_expand_remote_profile_tokens(profile, profile.get("campaign_root", "~")), campaign_dir.name)



def _safe_download_leaf(target_key: str | Path, fallback: str) -> str:
    raw = Path(str(target_key)).name or fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned or fallback



def _supek_stage_parent(run_root: Path) -> Path:
    return downloads_root(run_root, "supek")



def _supek_upload_stage_parent(run_root: Path) -> Path:
    return uploads_root(run_root, "supek")



def _bura_stage_parent(run_root: Path, campaign_dir: Path) -> Path:
    return downloads_root(run_root, "bura") / campaign_dir.name / "packages"



def _should_skip_supek_upload_name(name: str) -> bool:
    return (
        name in {"md_campaigns", "_dashboard_actions", "_dashboard_remote_state", "__pycache__"}
        or name.startswith('.tmp')
        or name.startswith('tmp')
    )



def _stage_supek_run_upload(run_root: Path, run_dir: Path) -> Path:
    stage_parent = _supek_upload_stage_parent(run_root)
    stage_dir = stage_parent / run_dir.name
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)

    def _ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if _should_skip_supek_upload_name(name)}

    shutil.copytree(run_dir, stage_dir, ignore=_ignore)
    return stage_dir



def _pbs_wrapper_lines(
    profile: dict[str, str],
    *,
    job_name: str,
    stdout_log: str,
    stderr_log: str,
    command_name: str,
    remote_run_dir: str,
) -> list[str]:
    queue = str(profile.get("default_queue", "")).strip()
    walltime = str(profile.get("default_walltime", "")).strip()
    pbs_select = str(profile.get("pbs_select", "")).strip()
    lines = ["#!/bin/bash", f"#PBS -N {job_name}"]
    if queue:
        lines.append(f"#PBS -q {queue}")
    if walltime:
        lines.append(f"#PBS -l walltime={walltime}")
    if pbs_select:
        lines.append(f"#PBS -l {pbs_select}")
    lines.extend(
        [
            f"#PBS -o {stdout_log}",
            f"#PBS -e {stderr_log}",
            "set -euo pipefail",
            f"source {_remote_shell_path(profile['conda_init_path'], profile)}",
            f"conda activate {_posix_quote(profile['env_name'])}",
        ]
    )
    module_load = str(profile.get("module_load", "")).strip()
    if module_load:
        lines.append(module_load)
    lines.extend(
        [
            f"cd {_remote_shell_path(profile['repo_path'], profile)}",
            f"python -m active_learning_thesis {command_name} --run-dir {_remote_shell_path(remote_run_dir, profile)}",
        ]
    )
    return lines



def draft_supek_verify_action(*, run_root: Path, run_dir: Path, profile: dict[str, str]) -> dict[str, object]:
    remote_command = (
        f"test -d {_remote_shell_path(profile['repo_path'], profile)} && "
        f"test -f {_remote_shell_path(profile['conda_init_path'], profile)} && "
        f"echo repo_ok && echo env={_posix_quote(profile['env_name'])}"
    )
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Verify Supek profile for {run_dir.name}",
        kind="supek-verify-env",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={"remote_path": profile.get("repo_path", "")},
        requires_approval=False,
    )



def draft_supek_sync_repo_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    branch: str,
) -> dict[str, object]:
    remote_command = (
        f"cd {_remote_shell_path(profile['repo_path'], profile)} && "
        "origin_url=$(git remote get-url origin) && "
        "case \"$origin_url\" in https://github.com/*|http://github.com/*) "
        "echo 'Supek repo origin uses HTTPS; switch origin to SSH before using dashboard repo sync.' >&2; exit 12 ;; esac && "
        f"GIT_TERMINAL_PROMPT=0 git fetch origin && git checkout {_posix_quote(branch)} && GIT_TERMINAL_PROMPT=0 git pull origin {_posix_quote(branch)}"
    )
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Sync repo on Supek for {run_dir.name}",
        kind="supek-sync-repo",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={"branch": branch, "remote_path": profile['repo_path']},
        requires_approval=True,
    )



def draft_supek_sync_run_action(*, run_root: Path, run_dir: Path, profile: dict[str, str]) -> dict[str, object]:
    local_run_dir = run_dir.resolve()
    staged_run_dir = _stage_supek_run_upload(run_root, local_run_dir)
    remote_root = _expand_remote_profile_tokens(profile, profile.get("scratch_run_root", ""))
    remote_path = _remote_run_dir(profile, local_run_dir)
    command = _build_scp_upload_command(profile, staged_run_dir, remote_root)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Upload run directory to Supek for {local_run_dir.name}",
        kind="supek-sync-run",
        cluster="supek",
        command=command,
        cwd=staged_run_dir.parent,
        related_run=str(local_run_dir),
        output_path=staged_run_dir,
        metadata={
            "remote_path": remote_path,
            "target_key": str(local_run_dir),
            "local_stage_path": str(staged_run_dir),
        },
        requires_approval=True,
    )



def build_supek_workflow_payload(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    command_name: str,
    command_args: list[str] | None = None,
    job_config: dict[str, object] | None = None,
) -> SupekActionPayload:
    if command_name not in SUPEK_WORKFLOW_COMMANDS:
        raise ValueError(f"Unsupported Supek workflow command: {command_name}")
    remote_run_dir = _remote_run_dir(profile, run_dir)
    remote_log_dir = _remote_log_dir(profile, run_dir)
    job_name = f"{run_dir.name}_{command_name}".replace("-", "_")[:15]
    stdout_log = _posix_join(remote_log_dir, f"{command_name}.out")
    stderr_log = _posix_join(remote_log_dir, f"{command_name}.err")
    wrapper_path = _posix_join(remote_log_dir, f"{command_name}.pbs")

    queue = str(profile.get("default_queue", "")).strip()
    walltime = str(profile.get("default_walltime", "")).strip()
    pbs_select = str(profile.get("pbs_select", "")).strip()
    module_load = str(profile.get("module_load", "")).strip()
    wrapper_parts = [
        _posix_quote("#!/bin/bash"),
        _posix_quote(f"#PBS -N {job_name}"),
    ]
    if queue:
        wrapper_parts.append(_posix_quote(f"#PBS -q {queue}"))
    if walltime:
        wrapper_parts.append(_posix_quote(f"#PBS -l walltime={walltime}"))
    if pbs_select:
        wrapper_parts.append(_posix_quote(f"#PBS -l {pbs_select}"))
    wrapper_parts.extend(
        [
            '"#PBS -o ${stdout_log}"',
            '"#PBS -e ${stderr_log}"',
            _posix_quote("set -euo pipefail"),
        ]
    )
    wrapper_parts.extend(
        [
            '"source ${conda_init}"',
            _posix_quote(f"conda activate {profile['env_name']}"),
        ]
    )
    if module_load:
        wrapper_parts.append(_posix_quote(module_load))
    extra_args = " ".join(_posix_quote(str(item)) for item in (command_args or []))
    workflow_command = f"python -m active_learning_thesis {command_name} --run-dir ${{remote_run_dir}}"
    if extra_args:
        workflow_command = f"{workflow_command} {extra_args}"
    wrapper_parts.extend(
        [
            '"cd ${repo_path}"',
            f'"{workflow_command}"',
        ]
    )
    printf_lines = " ".join(wrapper_parts)
    remote_command = (
        f"log_dir={_remote_shell_path(remote_log_dir, profile)} && "
        'mkdir -p "$log_dir" && '
        f'wrapper_path="$log_dir/{command_name}.pbs" && '
        f'stdout_log="$log_dir/{command_name}.out" && '
        f'stderr_log="$log_dir/{command_name}.err" && '
        f"conda_init={_remote_shell_path(profile['conda_init_path'], profile)} && "
        f"repo_path={_remote_shell_path(profile['repo_path'], profile)} && "
        f"remote_run_dir={_remote_shell_path(remote_run_dir, profile)} && "
        f"printf '%s\n' {printf_lines} > \"$wrapper_path\" && "
        'qsub "$wrapper_path"'
    )
    command = _build_ssh_command(profile, remote_command)
    return SupekActionPayload(
        run_root=run_root,
        title=f"Submit {command_name} on Supek for {run_dir.name}",
        kind="supek-submit-workflow",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={
            "remote_path": remote_run_dir,
            "remote_stdout": stdout_log,
            "remote_stderr": stderr_log,
            "remote_wrapper": wrapper_path,
            "workflow_command": command_name,
            "workflow_command_args": list(command_args or []),
            "job_config": dict(job_config or {}),
            "target_key": str(run_dir),
        },
        requires_approval=True,
    )


def build_supek_study_payload(
    *,
    run_root: Path,
    study_name: str,
    profile: dict[str, str],
    seeds: int,
    seed_start: int,
    seed_step: int,
    epochs: int,
    max_rounds: int,
    batch_size: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    strategies: list[str] | None = None,
    metric: str = "f1",
    target: float | None = None,
    ensemble_size: int = 5,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    dry_run: bool = False,
    force_replay: bool = False,
    summarize: bool = True,
    allow_config_mismatch: bool = False,
    walltime: str = "",
    aggregate_walltime: str = "",
) -> SupekActionPayload:
    normalized_study_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(study_name or "").strip()).strip("._-")
    if not normalized_study_name:
        raise ValueError("Study name is empty.")
    remote_run_root = _expand_remote_profile_tokens(profile, profile.get("scratch_run_root", ""))
    remote_log_dir = _posix_join(_expand_remote_profile_tokens(profile, profile.get("log_root", "")), normalized_study_name)
    job_name = f"{normalized_study_name}_study".replace("-", "_")[:15]
    stdout_log = _posix_join(remote_log_dir, "run-study.out")
    stderr_log = _posix_join(remote_log_dir, "run-study.err")
    wrapper_path = _posix_join(remote_log_dir, "run-study.pbs")

    queue = str(profile.get("default_queue", "")).strip()
    requested_walltime = str(walltime or "").strip() or str(profile.get("default_walltime", "")).strip()
    pbs_select = str(profile.get("pbs_select", "")).strip()
    module_load = str(profile.get("module_load", "")).strip()
    strategy_parts = " ".join(_posix_quote(str(item)) for item in (strategies or []))
    study_command = (
        "python -m active_learning_thesis run-study "
        f"--study-name {_posix_quote(normalized_study_name)} "
        f"--run-root {_remote_shell_path(remote_run_root, profile)} "
        f"--seeds {int(seeds)} "
        f"--seed-start {int(seed_start)} "
        f"--seed-step {int(seed_step)} "
        f"--epochs {int(epochs)} "
        f"--max-rounds {int(max_rounds)} "
        f"--batch-size {int(batch_size)} "
        f"--candidate-pool-min {int(candidate_pool_min)} "
        f"--replay-seed-size {int(replay_seed_size)} "
        f"--real-strategy {_posix_quote(str(real_strategy or 'ensemble_mi'))} "
        f"--ensemble-size {int(ensemble_size)} "
        f"--metric {_posix_quote(str(metric or 'f1'))} "
        f"--generator-objective-mode {_posix_quote(str(generator_objective_mode))} "
        f"--binary-threshold-strategy {_posix_quote(str(binary_threshold_strategy))}"
    )
    if strategy_parts:
        study_command += f" --strategies {strategy_parts}"
    if target is not None:
        study_command += f" --target {target}"
    if train_family_for_init:
        study_command += " --train-family-for-init"
    if not use_calibrated_acquisition:
        study_command += " --raw-acquisition"
    if use_similarity_penalty:
        study_command += " --use-similarity-penalty"
    if not use_length_penalty:
        study_command += " --no-length-penalty"
    if dry_run:
        study_command += " --dry-run"
    if force_replay:
        study_command += " --force-replay"
    if not summarize:
        study_command += " --no-summarize"
    if allow_config_mismatch:
        study_command += " --allow-config-mismatch"

    wrapper_parts = [
        _posix_quote("#!/bin/bash"),
        _posix_quote(f"#PBS -N {job_name}"),
    ]
    if queue:
        wrapper_parts.append(_posix_quote(f"#PBS -q {queue}"))
    if requested_walltime:
        wrapper_parts.append(_posix_quote(f"#PBS -l walltime={requested_walltime}"))
    if pbs_select:
        wrapper_parts.append(_posix_quote(f"#PBS -l {pbs_select}"))
    wrapper_parts.extend(
        [
            '"#PBS -o ${stdout_log}"',
            '"#PBS -e ${stderr_log}"',
            _posix_quote("set -euo pipefail"),
            '"source ${conda_init}"',
            _posix_quote(f"conda activate {profile['env_name']}"),
        ]
    )
    if module_load:
        wrapper_parts.append(_posix_quote(module_load))
    wrapper_parts.extend(['"cd ${repo_path}"', _posix_quote(study_command)])
    printf_lines = " ".join(wrapper_parts)
    remote_command = (
        f"log_dir={_remote_shell_path(remote_log_dir, profile)} && "
        'mkdir -p "$log_dir" && '
        'wrapper_path="$log_dir/run-study.pbs" && '
        'stdout_log="$log_dir/run-study.out" && '
        'stderr_log="$log_dir/run-study.err" && '
        f"conda_init={_remote_shell_path(profile['conda_init_path'], profile)} && "
        f"repo_path={_remote_shell_path(profile['repo_path'], profile)} && "
        f"printf '%s\n' {printf_lines} > \"$wrapper_path\" && "
        'qsub "$wrapper_path"'
    )
    command = _build_ssh_command(profile, remote_command)
    local_study_dir = run_root / "_studies" / normalized_study_name
    return SupekActionPayload(
        run_root=run_root,
        title=f"Submit study on Supek for {normalized_study_name}",
        kind="supek-submit-study",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(local_study_dir),
        output_path=local_study_dir,
        metadata={
            "study_name": normalized_study_name,
            "remote_path": _posix_join(remote_run_root, "_studies", normalized_study_name),
            "remote_run_root": remote_run_root,
            "remote_stdout": stdout_log,
            "remote_stderr": stderr_log,
            "remote_wrapper": wrapper_path,
            "walltime": requested_walltime,
            "job_config": {
                "study_name": normalized_study_name,
                "seeds": int(seeds),
                "seed_start": int(seed_start),
                "seed_step": int(seed_step),
                "epochs": int(epochs),
                "max_rounds": int(max_rounds),
                "batch_size": int(batch_size),
                "candidate_pool_min": int(candidate_pool_min),
                "replay_seed_size": int(replay_seed_size),
                "real_strategy": str(real_strategy or "ensemble_mi"),
                "strategies": list(strategies or []),
                "metric": str(metric or "f1"),
                "target": target,
                "ensemble_size": int(ensemble_size),
                "train_family_for_init": bool(train_family_for_init),
                "use_calibrated_acquisition": bool(use_calibrated_acquisition),
                "generator_objective_mode": str(generator_objective_mode),
                "use_similarity_penalty": bool(use_similarity_penalty),
                "use_length_penalty": bool(use_length_penalty),
                "binary_threshold_strategy": str(binary_threshold_strategy),
                "dry_run": bool(dry_run),
                "force_replay": bool(force_replay),
                "summarize": bool(summarize),
                "allow_config_mismatch": bool(allow_config_mismatch),
                "split_by_seed": False,
                "queue": queue,
                "walltime": requested_walltime,
                "aggregate_walltime": str(aggregate_walltime or ""),
                "pbs_select": pbs_select,
            },
            "target_key": str(local_study_dir),
        },
        requires_approval=True,
    )


def build_supek_study_array_payload(
    *,
    run_root: Path,
    study_name: str,
    profile: dict[str, str],
    seeds: int,
    seed_start: int,
    seed_step: int,
    epochs: int,
    max_rounds: int,
    batch_size: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    strategies: list[str] | None = None,
    metric: str = "f1",
    target: float | None = None,
    ensemble_size: int = 5,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    dry_run: bool = False,
    force_replay: bool = False,
    summarize: bool = True,
    allow_config_mismatch: bool = False,
    walltime: str = "",
    aggregate_walltime: str = "02:00:00",
) -> SupekActionPayload:
    normalized_study_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(study_name or "").strip()).strip("._-")
    if not normalized_study_name:
        raise ValueError("Study name is empty.")
    if seeds <= 0:
        raise ValueError("Seed count must be positive.")
    remote_run_root = _expand_remote_profile_tokens(profile, profile.get("scratch_run_root", ""))
    remote_log_dir = _posix_join(_expand_remote_profile_tokens(profile, profile.get("log_root", "")), normalized_study_name)
    queue = str(profile.get("default_queue", "")).strip()
    requested_walltime = str(walltime or "").strip() or str(profile.get("default_walltime", "")).strip()
    requested_aggregate_walltime = str(aggregate_walltime or "").strip() or "02:00:00"
    pbs_select = str(profile.get("pbs_select", "")).strip()
    module_load = str(profile.get("module_load", "")).strip()
    strategy_parts = " ".join(_posix_quote(str(item)) for item in (strategies or []))

    def study_command_for(*, seed_count: int, current_seed_start: int, current_seed_index_start: int, include_summary: bool) -> str:
        command = (
            "python -m active_learning_thesis run-study "
            f"--study-name {_posix_quote(normalized_study_name)} "
            f"--run-root {_remote_shell_path(remote_run_root, profile)} "
            f"--seeds {int(seed_count)} "
            f"--seed-start {int(current_seed_start)} "
            f"--seed-step {int(seed_step)} "
            f"--seed-index-start {int(current_seed_index_start)} "
            f"--epochs {int(epochs)} "
            f"--max-rounds {int(max_rounds)} "
            f"--batch-size {int(batch_size)} "
            f"--candidate-pool-min {int(candidate_pool_min)} "
            f"--replay-seed-size {int(replay_seed_size)} "
            f"--real-strategy {_posix_quote(str(real_strategy or 'ensemble_mi'))} "
            f"--ensemble-size {int(ensemble_size)} "
            f"--metric {_posix_quote(str(metric or 'f1'))} "
            f"--generator-objective-mode {_posix_quote(str(generator_objective_mode))} "
            f"--binary-threshold-strategy {_posix_quote(str(binary_threshold_strategy))}"
        )
        if strategy_parts:
            command += f" --strategies {strategy_parts}"
        if target is not None:
            command += f" --target {target}"
        if train_family_for_init:
            command += " --train-family-for-init"
        if not use_calibrated_acquisition:
            command += " --raw-acquisition"
        if use_similarity_penalty:
            command += " --use-similarity-penalty"
        if not use_length_penalty:
            command += " --no-length-penalty"
        if dry_run:
            command += " --dry-run"
        if force_replay:
            command += " --force-replay"
        if not include_summary:
            command += " --no-summarize"
        if allow_config_mismatch:
            command += " --allow-config-mismatch"
        return command

    def wrapper_parts_for(*, job_name: str, stdout_name: str, stderr_name: str, command: str, wrapper_walltime: str) -> list[str]:
        parts = [_posix_quote("#!/bin/bash"), _posix_quote(f"#PBS -N {job_name}")]
        if queue:
            parts.append(_posix_quote(f"#PBS -q {queue}"))
        if wrapper_walltime:
            parts.append(_posix_quote(f"#PBS -l walltime={wrapper_walltime}"))
        if pbs_select:
            parts.append(_posix_quote(f"#PBS -l {pbs_select}"))
        parts.extend(
            [
                f'"#PBS -o $log_dir/{stdout_name}"',
                f'"#PBS -e $log_dir/{stderr_name}"',
                _posix_quote("set -euo pipefail"),
                '"source ${conda_init}"',
                _posix_quote(f"conda activate {profile['env_name']}"),
            ]
        )
        if module_load:
            parts.append(_posix_quote(module_load))
        parts.extend(['"cd ${repo_path}"', _posix_quote(command)])
        return parts

    remote_commands = [
        f"log_dir={_remote_shell_path(remote_log_dir, profile)}",
        'mkdir -p "$log_dir"',
        f"conda_init={_remote_shell_path(profile['conda_init_path'], profile)}",
        f"repo_path={_remote_shell_path(profile['repo_path'], profile)}",
        'cd "$repo_path"',
        'source "$conda_init"',
        f"conda activate {_posix_quote(profile['env_name'])}",
        "python -m active_learning_thesis run-study --help | grep -q -- --seed-index-start",
        'seed_job_ids=""',
    ]
    seed_values = [seed_start + index * seed_step for index in range(seeds)]
    for index, seed in enumerate(seed_values, start=1):
        stem = f"seed_{index:02d}_{seed}"
        wrapper_path = f"$log_dir/{stem}.pbs"
        seed_command = study_command_for(
            seed_count=1,
            current_seed_start=seed,
            current_seed_index_start=index,
            include_summary=False,
        )
        seed_parts = wrapper_parts_for(
            job_name=f"{normalized_study_name[:8]}_s{index:02d}"[:15],
            stdout_name=f"{stem}.out",
            stderr_name=f"{stem}.err",
            command=seed_command,
            wrapper_walltime=requested_walltime,
        )
        remote_commands.extend(
            [
                f"printf '%s\n' {' '.join(seed_parts)} > {wrapper_path}",
                f"jid=$(qsub {wrapper_path})",
                f"echo seed_{index:02d}_job=$jid",
                'seed_job_ids="${seed_job_ids:+$seed_job_ids:}$jid"',
            ]
        )

    aggregate_command = study_command_for(
        seed_count=seeds,
        current_seed_start=seed_start,
        current_seed_index_start=1,
        include_summary=summarize,
    )
    aggregate_parts = wrapper_parts_for(
        job_name=f"{normalized_study_name[:8]}_agg"[:15],
        stdout_name="aggregate.out",
        stderr_name="aggregate.err",
        command=aggregate_command,
        wrapper_walltime=requested_aggregate_walltime,
    )
    remote_commands.extend(
        [
            'printf \'%s\n\' ' + " ".join(aggregate_parts) + ' > "$log_dir/aggregate.pbs"',
            'agg_jid=$(qsub -W depend=afterok:$seed_job_ids "$log_dir/aggregate.pbs")',
            'echo aggregate_job=$agg_jid',
        ]
    )
    remote_command = " && ".join(remote_commands)
    command = _build_ssh_command(profile, remote_command)
    local_study_dir = run_root / "_studies" / normalized_study_name
    return SupekActionPayload(
        run_root=run_root,
        title=f"Submit split study on Supek for {normalized_study_name}",
        kind="supek-submit-study-array",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(local_study_dir),
        output_path=local_study_dir,
        metadata={
            "study_name": normalized_study_name,
            "remote_path": _posix_join(remote_run_root, "_studies", normalized_study_name),
            "remote_run_root": remote_run_root,
            "remote_stdout": _posix_join(remote_log_dir, "aggregate.out"),
            "remote_stderr": _posix_join(remote_log_dir, "aggregate.err"),
            "remote_wrapper": _posix_join(remote_log_dir, "aggregate.pbs"),
            "seed_job_count": seeds,
            "walltime": requested_walltime,
            "aggregate_walltime": requested_aggregate_walltime,
            "job_config": {
                "study_name": normalized_study_name,
                "seeds": int(seeds),
                "seed_start": int(seed_start),
                "seed_step": int(seed_step),
                "epochs": int(epochs),
                "max_rounds": int(max_rounds),
                "batch_size": int(batch_size),
                "candidate_pool_min": int(candidate_pool_min),
                "replay_seed_size": int(replay_seed_size),
                "real_strategy": str(real_strategy or "ensemble_mi"),
                "strategies": list(strategies or []),
                "metric": str(metric or "f1"),
                "target": target,
                "ensemble_size": int(ensemble_size),
                "train_family_for_init": bool(train_family_for_init),
                "use_calibrated_acquisition": bool(use_calibrated_acquisition),
                "generator_objective_mode": str(generator_objective_mode),
                "use_similarity_penalty": bool(use_similarity_penalty),
                "use_length_penalty": bool(use_length_penalty),
                "binary_threshold_strategy": str(binary_threshold_strategy),
                "dry_run": bool(dry_run),
                "force_replay": bool(force_replay),
                "summarize": bool(summarize),
                "allow_config_mismatch": bool(allow_config_mismatch),
                "split_by_seed": True,
                "queue": queue,
                "walltime": requested_walltime,
                "aggregate_walltime": requested_aggregate_walltime,
                "pbs_select": pbs_select,
            },
            "target_key": str(local_study_dir),
        },
        requires_approval=True,
    )



def draft_supek_submit_action(**kwargs) -> dict[str, object]:
    return draft_supek_payload_action(build_supek_workflow_payload(**kwargs))


def draft_supek_submit_study_action(**kwargs) -> dict[str, object]:
    return draft_supek_payload_action(build_supek_study_payload(**kwargs))


def draft_supek_submit_study_array_action(**kwargs) -> dict[str, object]:
    return draft_supek_payload_action(build_supek_study_array_payload(**kwargs))


def queue_supek_preflight_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    require_staged_run: bool = True,
) -> dict[str, object]:
    remote_repo = _remote_shell_path(profile["repo_path"], profile)
    remote_repo_git = _remote_shell_path(_posix_join(_expand_remote_profile_tokens(profile, profile["repo_path"]), ".git"), profile)
    remote_conda = _remote_shell_path(profile["conda_init_path"], profile)
    remote_scratch_root = _remote_shell_path(_expand_remote_profile_tokens(profile, profile.get("scratch_run_root", "")), profile)
    remote_run_dir = _remote_shell_path(_remote_run_dir(profile, run_dir), profile)
    remote_log_root = _remote_shell_path(_expand_remote_profile_tokens(profile, profile.get("log_root", "")), profile)
    env_name = str(profile.get("env_name", "")).strip()
    checks = [
        _emit_remote_check(f"test -d {remote_repo}", "repo_ok", "repo_missing"),
        _emit_remote_check(f"test -d {remote_repo_git}", "repo_git_ok", "repo_git_missing"),
        _emit_remote_check(f"test -f {remote_conda}", "conda_init_ok", "conda_init_missing"),
        _emit_remote_check("command -v qsub >/dev/null 2>&1", "scheduler_cmd_ok", "scheduler_cmd_missing"),
        _emit_remote_check(f"test -d {remote_scratch_root}", "scratch_root_ok", "scratch_root_missing"),
        _emit_remote_check(
            f"test -d {remote_run_dir}",
            "run_state_staged",
            "run_state_missing",
        ),
        _emit_remote_check(f"test -d {remote_log_root}", "log_root_ok", "log_root_missing"),
    ]
    module_load = str(profile.get("module_load", "")).strip()
    env_setup = ""
    if env_name:
        env_setup = f"source {remote_conda} >/dev/null 2>&1 && conda activate {_posix_quote(env_name)} >/dev/null 2>&1"
    if env_setup and module_load:
        checks.append(
            _emit_remote_check(
                f"{env_setup} && {module_load} >/dev/null 2>&1",
                "module_load_ok",
                "module_load_missing",
            )
        )
    elif module_load:
        checks.append(_emit_remote_check(f"{module_load} >/dev/null 2>&1", "module_load_ok", "module_load_missing"))
    if env_setup:
        import_setup = env_setup
        if module_load:
            import_setup = f"{import_setup} && {module_load} >/dev/null 2>&1"
        checks.append(
            _emit_remote_check(
                f"{import_setup} && python -c {_posix_quote('import active_learning_thesis')} >/dev/null 2>&1",
                "python_import_ok",
                "python_import_missing",
            )
        )
    remote_command = " ; ".join(checks)
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Run SUPEK submit preflight for {run_dir.name}",
        kind="supek-submit-preflight",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={
            "target_key": str(run_dir),
            "require_staged_run": require_staged_run,
            "remote_path": _remote_run_dir(profile, run_dir),
        },
        requires_approval=False,
    )


def queue_supek_poll_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    remote_job_id: str = "",
) -> dict[str, object]:
    command = _build_ssh_command(profile, f"qstat -u {_posix_quote(profile['username'])}")
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Poll Supek queue for {run_dir.name}",
        kind="supek-poll-qstat",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={"target_key": str(run_dir), "remote_job_id": remote_job_id},
        requires_approval=False,
    )



def queue_supek_fetch_logs_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    remote_stdout: str = "",
    remote_stderr: str = "",
    remote_wrapper: str = "",
) -> dict[str, object]:
    commands: list[str] = []
    for label, path in (
        ("PBS wrapper", remote_wrapper),
        ("stdout", remote_stdout),
        ("stderr", remote_stderr),
    ):
        if not str(path).strip():
            continue
        remote_file = _remote_shell_path(path, profile)
        commands.append(
            f"if test -f {remote_file}; then "
            f"echo '===== {label}: {path} ====='; "
            f"tail -n 120 {remote_file}; "
            "else "
            f"echo '===== {label}: missing ({path}) ====='; "
            "fi"
        )
    if not commands:
        remote_log_dir = _remote_log_dir(profile, run_dir)
        remote_dir = _remote_shell_path(remote_log_dir, profile)
        commands.append(
            f"if test -d {remote_dir}; then "
            f"cd {remote_dir} && "
            "found=0; "
            "for file in *.pbs *.out *.err; do "
            "if test -f \"$file\"; then found=1; echo \"===== $file =====\"; tail -n 120 \"$file\"; fi; "
            "done; "
            "if test \"$found\" -eq 0; then echo 'No SUPEK log files found yet.'; fi; "
            "else "
            "echo 'No SUPEK log directory found yet.'; "
            "fi"
        )
    command = _build_ssh_command(profile, " && ".join(commands))
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Fetch latest SUPEK logs for {run_dir.name}",
        kind="supek-fetch-logs",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={
            "target_key": str(run_dir),
            "remote_stdout": remote_stdout,
            "remote_stderr": remote_stderr,
            "remote_wrapper": remote_wrapper,
        },
        requires_approval=False,
    )


def draft_supek_cancel_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
    remote_job_id: str,
) -> dict[str, object]:
    job_id = str(remote_job_id).strip()
    if not job_id:
        raise ValueError("A tracked SUPEK job id is required before the job can be cancelled.")
    command = _build_ssh_command(profile, f"qdel {_posix_quote(job_id)}")
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Cancel SUPEK job for {run_dir.name}",
        kind="supek-cancel-job",
        cluster="supek",
        command=command,
        cwd=Path.cwd(),
        related_run=str(run_dir),
        output_path=run_dir,
        metadata={"target_key": str(run_dir), "remote_job_id": job_id},
        requires_approval=True,
    )


def draft_supek_pull_artifacts_action(
    *,
    run_root: Path,
    run_dir: Path,
    profile: dict[str, str],
) -> dict[str, object]:
    remote_run_dir = _remote_run_dir(profile, run_dir)
    local_stage_parent = _supek_stage_parent(run_root)
    local_stage_parent.mkdir(parents=True, exist_ok=True)
    local_stage_path = local_stage_parent / _safe_download_leaf(run_dir, run_dir.name)
    command = _build_scp_download_command(profile, remote_run_dir, local_stage_parent)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Pull run artifacts from Supek for {run_dir.name}",
        kind="supek-pull-artifacts",
        cluster="supek",
        command=command,
        cwd=local_stage_parent,
        related_run=str(run_dir),
        output_path=local_stage_path,
        metadata={"remote_path": remote_run_dir, "target_key": str(run_dir), "local_stage_path": str(local_stage_path)},
        requires_approval=True,
    )



def draft_bura_upload_campaign_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    requires_approval: bool = True,
) -> dict[str, object]:
    local_campaign_dir = campaign_dir.resolve()
    remote_root = _expand_remote_profile_tokens(profile, profile.get("campaign_root", "~"))
    remote_path = _remote_campaign_dir(profile, local_campaign_dir)
    command = _build_scp_upload_command(profile, local_campaign_dir, remote_root)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Upload BURA campaign {local_campaign_dir.name}",
        kind="bura-upload-campaign",
        cluster="bura",
        command=command,
        cwd=local_campaign_dir.parent,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(local_campaign_dir),
        output_path=local_campaign_dir,
        metadata={"remote_path": remote_path, "target_key": str(local_campaign_dir)},
        requires_approval=requires_approval,
    )



def draft_bura_normalize_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    requires_approval: bool = True,
) -> dict[str, object]:
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_command = (
        f"cd {_remote_shell_path(remote_campaign_dir, profile)} && "
        'find . -type f -name "*.sh" -exec dos2unix {} + && '
        'find . -type f -name "*.sh" -exec chmod u+x {} +'
    )
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Normalize uploaded BURA scripts for {campaign_dir.name}",
        kind="bura-normalize-scripts",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"remote_path": remote_campaign_dir, "target_key": str(campaign_dir)},
        requires_approval=requires_approval,
    )



def draft_bura_preflight_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    requires_approval: bool = True,
) -> dict[str, object]:
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_command = f"cd {_remote_shell_path(remote_campaign_dir, profile)} && {profile['module_load']} && bash ./preflight_bura.sh"
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Run BURA preflight for {campaign_dir.name}",
        kind="bura-preflight",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"remote_path": remote_campaign_dir, "target_key": str(campaign_dir)},
        requires_approval=requires_approval,
    )



def draft_bura_submit_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    exclude_nodes: str = "",
    requires_approval: bool = True,
) -> dict[str, object]:
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    exclude = exclude_nodes.strip() or profile.get("default_exclude_nodes", "").strip()
    submit_cmd = "bash ./submit_chain.sh "
    if exclude:
        submit_cmd += f"--exclude {_posix_quote(exclude)} "
    submit_cmd += _posix_quote(sequence)
    remote_command = f"cd {_remote_shell_path(remote_campaign_dir, profile)} && {submit_cmd}"
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Submit BURA chain for {sequence}",
        kind="bura-submit-chain",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"remote_path": remote_campaign_dir, "exclude_nodes": exclude, "target_key": str(campaign_dir)},
        requires_approval=requires_approval,
    )



def queue_bura_readiness_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
) -> dict[str, object]:
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_campaign = _remote_shell_path(remote_campaign_dir, profile)
    package_dir = _remote_shell_path(_posix_join(remote_campaign_dir, "packages", sequence), profile)
    checks = [
        _emit_remote_check(f"test -d {remote_campaign}", "campaign_dir_ok", "campaign_dir_missing"),
        _emit_remote_check(f"cd {remote_campaign} && test -f ./preflight_bura.sh", "preflight_script_ok", "preflight_script_missing"),
        _emit_remote_check(f"cd {remote_campaign} && test -f ./submit_chain.sh", "submit_script_ok", "submit_script_missing"),
        _emit_remote_check(f"cd {remote_campaign} && bash -n ./preflight_bura.sh >/dev/null 2>&1", "preflight_syntax_ok", "preflight_syntax_missing"),
        _emit_remote_check(f"cd {remote_campaign} && bash -n ./submit_chain.sh >/dev/null 2>&1", "submit_syntax_ok", "submit_syntax_missing"),
        _emit_remote_check(f"test -d {package_dir}", "package_dir_ok", "package_dir_missing"),
        _emit_remote_check("command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1", "scheduler_cmd_ok", "scheduler_cmd_missing"),
        _emit_remote_check("command -v dos2unix >/dev/null 2>&1", "dos2unix_ok", "dos2unix_missing"),
        _emit_remote_check(f"{profile['module_load']} >/dev/null 2>&1", "module_load_ok", "module_load_missing"),
    ]
    remote_command = " ; ".join(checks)
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Run BURA submit readiness check for {sequence}",
        kind="bura-submit-readiness",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"target_key": str(campaign_dir), "remote_path": remote_campaign_dir},
        requires_approval=False,
    )


def queue_bura_reconcile_campaign_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
) -> dict[str, object]:
    """Verify a manually uploaded BURA campaign and bind it to dashboard state."""
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_campaign = _remote_shell_path(remote_campaign_dir, profile)
    package_dir = _remote_shell_path(_posix_join(remote_campaign_dir, "packages", sequence), profile)
    checks = [
        f"test -d {remote_campaign}",
        f"cd {remote_campaign} && test -f ./preflight_bura.sh",
        f"cd {remote_campaign} && test -f ./submit_chain.sh",
        f"test -d {package_dir}",
    ]
    remote_command = (
        " && ".join(checks)
        + f" && echo bura_campaign_reconciled:{_posix_quote(campaign_dir.name)}"
    )
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Reconcile existing BURA campaign {campaign_dir.name}",
        kind="bura-reconcile-campaign",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"target_key": str(campaign_dir), "remote_path": remote_campaign_dir},
        requires_approval=False,
    )


def queue_bura_poll_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    remote_job_id: str = "",
) -> dict[str, object]:
    command = _build_ssh_command(profile, f"squeue -u {_posix_quote(profile['username'])}")
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Poll BURA queue for {sequence}",
        kind="bura-poll-squeue",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"target_key": str(campaign_dir), "remote_job_id": remote_job_id},
        requires_approval=False,
    )



def queue_bura_fetch_logs_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
) -> dict[str, object]:
    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_command = (
        f"cd {_remote_shell_path(remote_campaign_dir, profile)} && "
        "found=0; "
        "for file in ./slurm-*.out ./*.out ./*.err ./*.log "
        f"./packages/{sequence}/*.out ./packages/{sequence}/*.err ./packages/{sequence}/*.log; do "
        "if test -f \"$file\"; then found=1; echo \"===== $file =====\"; tail -n 120 \"$file\"; fi; "
        "done; "
        "if test \"$found\" -eq 0; then echo 'No BURA log files found yet.'; fi"
    )
    command = _build_ssh_command(profile, remote_command)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Fetch latest BURA logs for {sequence}",
        kind="bura-inspect-logs",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"target_key": str(campaign_dir), "remote_path": remote_campaign_dir},
        requires_approval=False,
    )


def draft_bura_cancel_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    remote_job_id: str,
    requires_approval: bool = True,
) -> dict[str, object]:
    job_id = str(remote_job_id).strip()
    if not job_id:
        raise ValueError("A tracked BURA job id is required before the job can be cancelled.")
    command = _build_ssh_command(profile, f"scancel {_posix_quote(job_id)}")
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Cancel BURA job for {sequence}",
        kind="bura-cancel-job",
        cluster="bura",
        command=command,
        cwd=Path.cwd(),
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        metadata={"target_key": str(campaign_dir), "remote_job_id": job_id},
        requires_approval=requires_approval,
    )


def draft_bura_pull_package_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    profile: dict[str, str],
    related_run: str,
    requires_approval: bool = True,
) -> dict[str, object]:
    remote_package_dir = _posix_join(_remote_campaign_dir(profile, campaign_dir), "packages", sequence)
    local_stage_parent = _bura_stage_parent(run_root, campaign_dir)
    local_stage_parent.mkdir(parents=True, exist_ok=True)
    local_stage_path = local_stage_parent / _safe_download_leaf(sequence, sequence)
    command = _build_scp_download_command(profile, remote_package_dir, local_stage_parent)
    return _queue_or_draft_remote_action(
        run_root=run_root,
        title=f"Copy BURA outputs back for {sequence}",
        kind="bura-pull-package",
        cluster="bura",
        command=command,
        cwd=local_stage_parent,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=local_stage_path,
        metadata={"remote_path": remote_package_dir, "target_key": str(campaign_dir), "local_stage_path": str(local_stage_path)},
        requires_approval=requires_approval,
    )


def submit_bura_md_workflow_action(
    *,
    run_root: Path,
    payload: dict[str, object],
    launch_worker: bool = True,
) -> dict[str, object]:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    sequence = str(payload.get("sequence", ""))
    campaign_dir = Path(str(payload.get("campaign_dir", "")))
    related_run = str(metadata.get("run_dir", ""))
    payload_json = json.dumps(payload, sort_keys=True)
    args = [
        "-m",
        "active_learning_thesis",
        "dashboard-run-bura-md-workflow",
        "--payload-json",
        payload_json,
    ]
    action_metadata = {
        "macro_kind": "bura-md-workflow",
        "backend_kind": "bura-full-autopilot",
        "payload_version": str(payload.get("payload_version", metadata.get("payload_version", ""))),
        "sequence": sequence,
        "source_type": str(payload.get("source_type", "")),
        "source_row": payload.get("source_row", {}),
        "source_path": str(metadata.get("source_path", "")),
        "md_profile": str(payload.get("md_profile", "")),
        "campaign_name": str(payload.get("campaign_name", "")),
        "campaign_dir": str(campaign_dir),
        "expected_artifacts": payload.get("expected_artifacts", []),
        "stage_commands": payload.get("stage_commands", []),
        "preview_command": str(payload.get("macro_command", "")),
        "payload": payload,
        "target_key": str(campaign_dir),
    }
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Run BURA MD workflow for {sequence}",
        kind="bura-md-workflow",
        command=sys.executable,
        args=args,
        cwd=Path.cwd(),
        scope="bura",
        cluster="bura",
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        display_command=serialize_action_command(sys.executable, args),
        metadata=action_metadata,
        exclusive=True,
        launch_worker=launch_worker,
    )


def submit_bura_full_autopilot_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    related_run: str,
    exclude_nodes: str = "",
    poll_seconds: int = 60,
    max_wait_seconds: int = 60 * 60 * 24 * 14,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "bura-full-autopilot",
        "--run-root",
        str(run_root),
        "--campaign-dir",
        str(campaign_dir),
        "--sequence",
        sequence,
        "--poll-seconds",
        str(poll_seconds),
        "--max-wait-seconds",
        str(max_wait_seconds),
    ]
    if exclude_nodes.strip():
        args.extend(["--exclude-nodes", exclude_nodes.strip()])
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Full BURA autopilot for {sequence}",
        kind="bura-full-autopilot",
        command=sys.executable,
        args=args,
        cwd=Path.cwd(),
        scope="bura",
        cluster="bura",
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        display_command=serialize_action_command(sys.executable, args),
        metadata={"target_key": str(campaign_dir), "poll_seconds": poll_seconds, "exclude_nodes": exclude_nodes},
        exclusive=True,
    )


def submit_bura_recover_outputs_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    related_run: str,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "bura-recover-md-outputs",
        "--run-root",
        str(run_root),
        "--campaign-dir",
        str(campaign_dir),
        "--sequence",
        sequence,
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Recover BURA outputs for {sequence}",
        kind="bura-recover-md-outputs",
        command=sys.executable,
        args=args,
        cwd=Path.cwd(),
        scope="bura",
        cluster="bura",
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=campaign_dir,
        display_command=serialize_action_command(sys.executable, args),
        metadata={
            "target_key": str(campaign_dir),
            "recovery_mode": "copy_back_finalize_only",
            "submits_new_simulation": False,
        },
        exclusive=True,
        launch_worker=launch_worker,
    )



def _matching_sync_status(jobs: list[dict[str, str]], remote_job_id: str) -> str | None:
    if not remote_job_id:
        return None
    matching = [job for job in jobs if str(job.get("job_id", "")) == remote_job_id]
    if not matching:
        return None
    held = False
    pending = False
    for job in matching:
        state = str(job.get("state", "")).upper()
        reason = str(job.get("reason", "")).lower()
        if state == "R":
            return "running"
        if state == "PD":
            pending = True
            if "held" in reason or "dependencynever" in reason:
                held = True
    if held or pending:
        return "submitted"
    return None



def post_process_action(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    kind = str(payload.get("kind", ""))
    cluster = str(payload.get("cluster", ""))
    status = str(payload.get("status", ""))
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    related_run = str(payload.get("related_run", ""))
    related_campaign = str(payload.get("related_campaign", ""))
    related_sequence = str(payload.get("related_sequence", ""))
    target_key = str(metadata.get("target_key", "") or related_campaign or related_run)
    remote_path = str(metadata.get("remote_path", ""))
    updates: dict[str, object] = {}

    if kind == "finalize-md-stage" and status == "succeeded" and related_campaign:
        update_sync_status(
            run_root,
            cluster="bura",
            target_key=related_campaign,
            status="finalized_local",
            related_run=related_run,
            related_campaign=related_campaign,
            related_sequence=related_sequence,
            metadata={"source": "finalize-md-stage"},
        )
        updates["sync_status"] = "finalized_local"
        return updates

    if cluster not in {"supek", "bura"}:
        return updates

    stdout_text = ""
    stdout_log = Path(str(payload.get("stdout_log", "")))
    if stdout_log.exists():
        stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")

    if kind == "supek-poll-qstat" and status in {"succeeded", "failed"}:
        jobs = parse_qstat_output(stdout_text)
        snapshot = {
            "cluster": "supek",
            "collected_at": _now_iso(),
            "jobs": jobs,
            "summary": summarize_cluster_jobs(jobs),
            "raw_excerpt": "\n".join(stdout_text.splitlines()[-40:]),
        }
        save_cluster_snapshot(run_root, "supek", snapshot)
        remote_job_id = str(metadata.get("remote_job_id", "") or payload.get("remote_job_id", ""))
        sync_status = _matching_sync_status(jobs, remote_job_id)
        if target_key and sync_status:
            update_sync_status(
                run_root,
                cluster="supek",
                target_key=target_key,
                status=sync_status,
                related_run=related_run,
                related_campaign=related_campaign,
                related_sequence=related_sequence,
                remote_path=remote_path,
                remote_job_id=remote_job_id,
                metadata={"polled_at": snapshot["collected_at"]},
            )
            updates["sync_status"] = sync_status
        return updates

    if kind == "bura-poll-squeue" and status in {"succeeded", "failed"}:
        jobs = parse_squeue_output(stdout_text)
        snapshot = {
            "cluster": "bura",
            "collected_at": _now_iso(),
            "jobs": jobs,
            "summary": summarize_cluster_jobs(jobs),
            "raw_excerpt": "\n".join(stdout_text.splitlines()[-40:]),
        }
        save_cluster_snapshot(run_root, "bura", snapshot)
        remote_job_id = str(metadata.get("remote_job_id", "") or payload.get("remote_job_id", ""))
        sync_status = _matching_sync_status(jobs, remote_job_id)
        if target_key and sync_status:
            update_sync_status(
                run_root,
                cluster="bura",
                target_key=target_key,
                status=sync_status,
                related_run=related_run,
                related_campaign=related_campaign,
                related_sequence=related_sequence,
                remote_path=remote_path,
                remote_job_id=remote_job_id,
                metadata={"polled_at": snapshot["collected_at"]},
            )
            updates["sync_status"] = sync_status
        return updates

    if status != "succeeded":
        return updates

    if kind == "supek-sync-repo":
        updates["repo_sync_status"] = "succeeded"
    elif kind == "supek-sync-run":
        updates["sync_status"] = "staged_remote"
    elif kind in {"supek-submit-workflow", "supek-submit-study"}:
        remote_job_id = extract_qsub_job_id(stdout_text)
        if remote_job_id:
            updates["remote_job_id"] = remote_job_id
        updates["sync_status"] = "submitted"
    elif kind == "supek-submit-study-array":
        remote_job_ids = extract_qsub_job_ids(stdout_text)
        if remote_job_ids:
            updates["remote_job_id"] = remote_job_ids[-1]
            metadata = dict(metadata)
            metadata["remote_job_ids"] = remote_job_ids
            metadata["seed_remote_job_ids"] = remote_job_ids[:-1]
            metadata["aggregate_remote_job_id"] = remote_job_ids[-1]
            payload["metadata"] = metadata
        updates["sync_status"] = "submitted"
    elif kind == "supek-cancel-job":
        updates["sync_status"] = "staged_remote"
        updates["remote_job_id"] = ""
    elif kind == "supek-pull-artifacts":
        updates["sync_status"] = "outputs_staged"
        updates["local_stage_path"] = str(metadata.get("local_stage_path", ""))
    elif kind == "bura-upload-campaign":
        updates["sync_status"] = "staged_remote"
    elif kind in {"bura-reconcile-campaign", "bura-normalize-scripts", "bura-preflight"}:
        updates["sync_status"] = "staged_remote"
    elif kind == "bura-submit-chain":
        job_ids = extract_sbatch_job_ids(stdout_text)
        if job_ids:
            updates["remote_job_id"] = job_ids[0]
            metadata = dict(metadata)
            metadata["remote_job_ids"] = job_ids
            payload["metadata"] = metadata
        updates["sync_status"] = "submitted"
    elif kind == "bura-cancel-job":
        updates["sync_status"] = "staged_remote"
        updates["remote_job_id"] = ""
    elif kind == "bura-pull-package":
        updates["sync_status"] = "outputs_staged"
        updates["local_stage_path"] = str(metadata.get("local_stage_path", ""))
    elif kind == "bura-md-workflow":
        updates["sync_status"] = "finalized_local"

    if updates.get("sync_status") and target_key:
        sync_remote_job_id: str | None
        if kind in {"supek-cancel-job", "bura-cancel-job"}:
            sync_remote_job_id = None
        else:
            sync_remote_job_id = str(updates.get("remote_job_id", payload.get("remote_job_id", "")))
        update_sync_status(
            run_root,
            cluster=cluster,
            target_key=target_key,
            status=str(updates["sync_status"]),
            related_run=related_run,
            related_campaign=related_campaign,
            related_sequence=related_sequence,
            remote_path=remote_path,
            remote_job_id=sync_remote_job_id,
            metadata=metadata,
        )
    return updates

