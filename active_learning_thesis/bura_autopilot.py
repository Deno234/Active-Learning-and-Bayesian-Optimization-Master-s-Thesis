from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from active_learning_thesis.dashboard_profiles import load_cluster_profiles, validate_cluster_profile
from active_learning_thesis.dashboard_remote import (
    _build_scp_download_command,
    _build_scp_upload_command,
    _build_ssh_command,
    _posix_join,
    _posix_quote,
    _remote_campaign_dir,
    _remote_shell_path,
    extract_sbatch_job_ids,
)
from active_learning_thesis.dashboard_remote_state import downloads_root, update_sync_status
from active_learning_thesis.md_orchestrator import finalize_md_stage


FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def _remote_test(profile: dict[str, str], test_command: str) -> bool:
    command = _build_ssh_command(profile, f"{test_command} >/dev/null 2>&1")
    return _run(command, check=False).returncode == 0


def _run_remote(profile: dict[str, str], remote_command: str) -> subprocess.CompletedProcess[str]:
    return _run(_build_ssh_command(profile, remote_command))


def _related_run_from_campaign(campaign_dir: Path) -> str:
    if campaign_dir.parent.name == "md_campaigns":
        return str(campaign_dir.parent.parent)
    return str(campaign_dir.parent)


def _update_bura_status(
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    *,
    status: str,
    remote_path: str,
    remote_job_id: str = "",
    local_stage_path: str = "",
    metadata: dict[str, object] | None = None,
) -> None:
    payload = dict(metadata or {})
    if local_stage_path:
        payload["local_stage_path"] = local_stage_path
    update_sync_status(
        run_root,
        cluster="bura",
        target_key=str(campaign_dir),
        status=status,
        related_run=_related_run_from_campaign(campaign_dir),
        related_campaign=str(campaign_dir),
        related_sequence=sequence,
        remote_path=remote_path,
        remote_job_id=remote_job_id,
        metadata=payload,
    )


def _remote_final_outputs_exist(profile: dict[str, str], remote_package_dir: str, sequence: str) -> bool:
    package = _remote_shell_path(remote_package_dir, profile)
    return _remote_test(
        profile,
        f"test -f {package}/{_posix_quote(sequence + '_sasa_AP_SASA.txt')} "
        f"&& test -f {package}/{_posix_quote(sequence + '_sasa.xvg')}",
    )


def _remote_md_outputs_exist(profile: dict[str, str], remote_package_dir: str, sequence: str) -> bool:
    package = _remote_shell_path(remote_package_dir, profile)
    safe_sequence = re.sub(r"[^A-Za-z0-9_-]", "", sequence)
    if safe_sequence != sequence:
        raise ValueError(f"Unsafe peptide sequence for remote glob check: {sequence!r}")
    return _remote_test(
        profile,
        f"ls {package}/{safe_sequence}_*_CG.xtc >/dev/null 2>&1 "
        f"&& ls {package}/{safe_sequence}_*_CG.tpr >/dev/null 2>&1",
    )


def _repair_remote_analysis(profile: dict[str, str], remote_package_dir: str, sequence: str) -> None:
    package = _remote_shell_path(remote_package_dir, profile)
    print("Final AP/SASA files are missing; attempting direct analysis repair from completed MD outputs.", flush=True)
    _run_remote(
        profile,
        f"cd {package} && bash ./11_SASA_and_FrameDump.sh && bash ./12_AP_calc.sh",
    )
    if not _remote_final_outputs_exist(profile, remote_package_dir, sequence):
        raise RuntimeError("Remote analysis repair finished but AP/SASA result files are still missing.")


def _parse_squeue_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rows.append({"job_id": parts[0].strip(), "state": parts[1].strip(), "reason": parts[2].strip()})
    return rows


def _parse_campaign_squeue_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 5)
        if len(parts) < 6:
            continue
        rows.append(
            {
                "job_id": parts[0].strip(),
                "state": parts[1].strip(),
                "reason": parts[2].strip(),
                "job_name": parts[3].strip(),
                "workdir": parts[4].strip(),
                "command": parts[5].strip(),
            }
        )
    return rows


def _active_remote_campaign_jobs(
    profile: dict[str, str],
    *,
    remote_campaign_dir: str,
    remote_package_dir: str,
    sequence: str,
) -> list[dict[str, str]]:
    result = _run(
        _build_ssh_command(
            profile,
            "squeue "
            f"-u {_posix_quote(str(profile['username']))} "
            "-h -o '%i|%T|%R|%j|%Z|%o' || true",
        ),
        check=False,
    )
    remote_campaign = remote_campaign_dir.rstrip("/")
    remote_package = remote_package_dir.rstrip("/")
    sequence_prefix = sequence[:8]
    matches: list[dict[str, str]] = []
    for job in _parse_campaign_squeue_lines(result.stdout):
        haystack = "\n".join([job.get("workdir", ""), job.get("command", "")])
        if remote_package in haystack or remote_campaign in haystack:
            matches.append(job)
            continue
        # BURA truncates job names in the default queue view, but the generated
        # jobs keep a peptide prefix (for example DEDEDE_dyn_run). Treat that as
        # a hard resume guard too: duplicate submission is worse than attaching
        # to an existing same-peptide chain.
        if job.get("job_name", "").upper().startswith(sequence_prefix.upper()):
            matches.append(job)
    return matches


def _job_ids(jobs: list[dict[str, str]]) -> list[str]:
    ids: list[str] = []
    for job in jobs:
        job_id = str(job.get("job_id", "")).strip()
        if job_id and job_id not in ids:
            ids.append(job_id)
    return ids


def _wait_for_jobs(
    profile: dict[str, str],
    *,
    job_ids: list[str],
    poll_seconds: int,
    max_wait_seconds: int,
) -> None:
    ids = ",".join(job_ids)
    deadline = time.monotonic() + max_wait_seconds
    while True:
        result = _run(
            _build_ssh_command(profile, f"squeue -j {_posix_quote(ids)} -h -o '%i|%T|%R' || true"),
            check=False,
        )
        jobs = _parse_squeue_lines(result.stdout)
        if not jobs:
            print("Tracked BURA jobs are no longer visible in squeue.", flush=True)
            return
        print(f"Tracked BURA jobs: {jobs}", flush=True)
        for job in jobs:
            state = job["state"].upper()
            reason = job["reason"]
            if state in FAILED_SLURM_STATES or "DependencyNeverSatisfied" in reason:
                raise RuntimeError(f"BURA job failed or became unrecoverable: {job}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for BURA jobs after {max_wait_seconds} seconds: {ids}")
        time.sleep(max(1, poll_seconds))


def run_bura_full_autopilot(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    exclude_nodes: str = "",
    poll_seconds: int = 60,
    max_wait_seconds: int = 60 * 60 * 24 * 14,
) -> dict[str, str]:
    run_root = run_root.resolve()
    campaign_dir = campaign_dir.resolve()
    profiles_payload = load_cluster_profiles()
    profile = dict(profiles_payload.get("profiles", {}).get("bura", {})) if isinstance(profiles_payload.get("profiles"), dict) else {}
    missing = validate_cluster_profile("bura", profile)
    if missing:
        raise ValueError(f"BURA profile is incomplete: {', '.join(missing)}")

    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_package_dir = _posix_join(remote_campaign_dir, "packages", sequence)
    local_stage_parent = downloads_root(run_root, "bura") / campaign_dir.name / "packages"
    local_stage_path = local_stage_parent / sequence
    remote_campaign_shell = _remote_shell_path(remote_campaign_dir, profile)

    print(f"Autopilot campaign: {campaign_dir}", flush=True)
    print(f"Remote campaign: {remote_campaign_dir}", flush=True)

    if not _remote_test(profile, f"test -d {remote_campaign_shell}"):
        print("Checkpoint upload: remote campaign is missing, uploading local campaign.", flush=True)
        remote_root = str(profile.get("campaign_root", "~"))
        _run(_build_scp_upload_command(profile, campaign_dir, remote_root), cwd=campaign_dir.parent)
    else:
        print("Checkpoint upload: remote campaign already exists, continuing.", flush=True)
    _update_bura_status(run_root, campaign_dir, sequence, status="staged_remote", remote_path=remote_campaign_dir)

    if not _remote_final_outputs_exist(profile, remote_package_dir, sequence):
        existing_jobs = _active_remote_campaign_jobs(
            profile,
            remote_campaign_dir=remote_campaign_dir,
            remote_package_dir=remote_package_dir,
            sequence=sequence,
        )
        job_ids = _job_ids(existing_jobs)
        if job_ids:
            print(
                f"Checkpoint submit: found existing active BURA jobs for this campaign, attaching instead of submitting: {job_ids}",
                flush=True,
            )
            _update_bura_status(
                run_root,
                campaign_dir,
                sequence,
                status="submitted",
                remote_path=remote_campaign_dir,
                remote_job_id=job_ids[0],
                metadata={"remote_job_ids": job_ids, "attached_existing_jobs": True},
            )
        else:
            print("Checkpoint normalize: normalizing remote scripts.", flush=True)
            _run_remote(
                profile,
                f"cd {remote_campaign_shell} && find . -type f -name \"*.sh\" -exec dos2unix {{}} + "
                "&& find . -type f -name \"*.sh\" -exec chmod u+x {} +",
            )
            _update_bura_status(run_root, campaign_dir, sequence, status="staged_remote", remote_path=remote_campaign_dir)

            print("Checkpoint preflight: running remote BURA preflight.", flush=True)
            try:
                _run_remote(profile, f"cd {remote_campaign_shell} && {profile['module_load']} && bash ./preflight_bura.sh")
            except Exception:
                _update_bura_status(run_root, campaign_dir, sequence, status="staged_remote", remote_path=remote_campaign_dir)
                raise
            _update_bura_status(run_root, campaign_dir, sequence, status="staged_remote", remote_path=remote_campaign_dir)

            existing_jobs = _active_remote_campaign_jobs(
                profile,
                remote_campaign_dir=remote_campaign_dir,
                remote_package_dir=remote_package_dir,
                sequence=sequence,
            )
            job_ids = _job_ids(existing_jobs)
            if job_ids:
                print(
                    f"Checkpoint submit: found existing active BURA jobs for this campaign, attaching instead of submitting: {job_ids}",
                    flush=True,
                )
                _update_bura_status(
                    run_root,
                    campaign_dir,
                    sequence,
                    status="submitted",
                    remote_path=remote_campaign_dir,
                    remote_job_id=job_ids[0],
                    metadata={"remote_job_ids": job_ids, "attached_existing_jobs": True},
                )
            else:
                print("Checkpoint submit: submitting BURA chain.", flush=True)
                submit_cmd = "bash ./submit_chain.sh "
                exclude = exclude_nodes.strip() or str(profile.get("default_exclude_nodes", "")).strip()
                if exclude:
                    submit_cmd += f"--exclude {_posix_quote(exclude)} "
                submit_cmd += _posix_quote(sequence)
                try:
                    submit = _run_remote(profile, f"cd {remote_campaign_shell} && {submit_cmd}")
                except Exception:
                    _update_bura_status(run_root, campaign_dir, sequence, status="staged_remote", remote_path=remote_campaign_dir)
                    raise
                job_ids = extract_sbatch_job_ids((submit.stdout or "") + "\n" + (submit.stderr or ""))
                if not job_ids:
                    raise RuntimeError("BURA submit did not report any sbatch job ids.")
                _update_bura_status(
                    run_root,
                    campaign_dir,
                    sequence,
                    status="submitted",
                    remote_path=remote_campaign_dir,
                    remote_job_id=job_ids[0],
                    metadata={"remote_job_ids": job_ids, "exclude_nodes": exclude},
                )

        print("Checkpoint poll: waiting for BURA chain to leave the queue.", flush=True)
        try:
            _wait_for_jobs(profile, job_ids=job_ids, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds)
        except Exception:
            _update_bura_status(run_root, campaign_dir, sequence, status="submitted", remote_path=remote_campaign_dir, remote_job_id=job_ids[0])
            raise
    else:
        print("Remote final AP/SASA files already exist; skipping submit/poll.", flush=True)

    if not _remote_final_outputs_exist(profile, remote_package_dir, sequence):
        if not _remote_md_outputs_exist(profile, remote_package_dir, sequence):
            raise RuntimeError("BURA chain ended but neither final analysis outputs nor completed MD trajectory/tpr were found.")
        try:
            _repair_remote_analysis(profile, remote_package_dir, sequence)
        except Exception:
            _update_bura_status(run_root, campaign_dir, sequence, status="submitted", remote_path=remote_campaign_dir)
            raise

    print("Checkpoint copy-back: copying remote package outputs back to dashboard staging.", flush=True)
    local_stage_parent.mkdir(parents=True, exist_ok=True)
    if local_stage_path.exists():
        shutil.rmtree(local_stage_path)
    try:
        _run(_build_scp_download_command(profile, remote_package_dir, local_stage_parent), cwd=local_stage_parent)
    except Exception:
        _update_bura_status(run_root, campaign_dir, sequence, status="submitted", remote_path=remote_package_dir)
        raise
    _update_bura_status(
        run_root,
        campaign_dir,
        sequence,
        status="outputs_staged",
        remote_path=remote_package_dir,
        local_stage_path=str(local_stage_path),
    )

    print("Checkpoint finalize: parsing copied-back outputs locally.", flush=True)
    try:
        review_path, review_row, next_message = finalize_md_stage(campaign_dir, local_stage_path)
    except Exception:
        _update_bura_status(
            run_root,
            campaign_dir,
            sequence,
            status="outputs_staged",
            remote_path=remote_package_dir,
            local_stage_path=str(local_stage_path),
        )
        raise
    _update_bura_status(
        run_root,
        campaign_dir,
        sequence,
        status="finalized_local",
        remote_path=remote_package_dir,
        local_stage_path=str(local_stage_path),
    )
    print(f"Parsed MD results into: {review_path}", flush=True)
    print(next_message, flush=True)
    return {
        "review_path": str(review_path),
        "job_root_status": str(review_row.get("job_root_status", "")),
        "local_stage_path": str(local_stage_path),
    }


def recover_bura_outputs_only(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
) -> dict[str, str]:
    """Copy back and finalize an existing BURA campaign without submitting jobs."""

    run_root = run_root.resolve()
    campaign_dir = campaign_dir.resolve()
    profiles_payload = load_cluster_profiles()
    profile = dict(profiles_payload.get("profiles", {}).get("bura", {})) if isinstance(profiles_payload.get("profiles"), dict) else {}
    missing = validate_cluster_profile("bura", profile)
    if missing:
        raise ValueError(f"BURA profile is incomplete: {', '.join(missing)}")

    remote_campaign_dir = _remote_campaign_dir(profile, campaign_dir)
    remote_package_dir = _posix_join(remote_campaign_dir, "packages", sequence)
    local_stage_parent = downloads_root(run_root, "bura") / campaign_dir.name / "packages"
    local_stage_path = local_stage_parent / sequence
    remote_campaign_shell = _remote_shell_path(remote_campaign_dir, profile)

    print(f"Recovery campaign: {campaign_dir}", flush=True)
    print(f"Remote campaign: {remote_campaign_dir}", flush=True)
    if not _remote_test(profile, f"test -d {remote_campaign_shell}"):
        raise RuntimeError("Remote BURA campaign directory is missing; recovery will not upload or submit.")

    if not _remote_final_outputs_exist(profile, remote_package_dir, sequence):
        if not _remote_md_outputs_exist(profile, remote_package_dir, sequence):
            raise RuntimeError("Remote outputs are not ready; recovery will not submit a new BURA job.")
        print("Checkpoint recovery-analysis: final AP/SASA outputs missing; repairing analysis only.", flush=True)
        try:
            _repair_remote_analysis(profile, remote_package_dir, sequence)
        except Exception:
            _update_bura_status(run_root, campaign_dir, sequence, status="submitted", remote_path=remote_package_dir)
            raise
        if not _remote_final_outputs_exist(profile, remote_package_dir, sequence):
            raise RuntimeError("Remote analysis repair did not produce final outputs; recovery stopped without submit.")

    print("Checkpoint recovery-copy-back: copying existing remote package outputs back.", flush=True)
    local_stage_parent.mkdir(parents=True, exist_ok=True)
    if local_stage_path.exists():
        shutil.rmtree(local_stage_path)
    try:
        _run(_build_scp_download_command(profile, remote_package_dir, local_stage_parent), cwd=local_stage_parent)
    except Exception:
        _update_bura_status(run_root, campaign_dir, sequence, status="submitted", remote_path=remote_package_dir)
        raise
    _update_bura_status(
        run_root,
        campaign_dir,
        sequence,
        status="outputs_staged",
        remote_path=remote_package_dir,
        local_stage_path=str(local_stage_path),
    )

    print("Checkpoint recovery-finalize: parsing copied-back outputs locally.", flush=True)
    try:
        review_path, review_row, next_message = finalize_md_stage(campaign_dir, local_stage_path)
    except Exception:
        _update_bura_status(
            run_root,
            campaign_dir,
            sequence,
            status="outputs_staged",
            remote_path=remote_package_dir,
            local_stage_path=str(local_stage_path),
        )
        raise
    _update_bura_status(
        run_root,
        campaign_dir,
        sequence,
        status="finalized_local",
        remote_path=remote_package_dir,
        local_stage_path=str(local_stage_path),
    )
    print(f"Parsed MD results into: {review_path}", flush=True)
    print(next_message, flush=True)
    return {
        "review_path": str(review_path),
        "job_root_status": str(review_row.get("job_root_status", "")),
        "local_stage_path": str(local_stage_path),
        "recovery_mode": "copy_back_finalize_only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a full BURA campaign autopilot until local MD parsing completes.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--exclude-nodes", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=60 * 60 * 24 * 14)
    args = parser.parse_args(argv)
    run_bura_full_autopilot(
        run_root=Path(args.run_root),
        campaign_dir=Path(args.campaign_dir),
        sequence=args.sequence,
        exclude_nodes=args.exclude_nodes,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    return 0
