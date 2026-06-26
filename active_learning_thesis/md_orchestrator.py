from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from active_learning_thesis.md_workflow import (
    build_pdbs,
    parse_md_results,
    prepare_md_campaign,
)

STAGE_META_FILENAME = "md_stage_meta.json"
NEXT_COMMANDS_FILENAME = "NEXT_BURA_COMMANDS.md"
GUIDED_PROFILE_ORDER = ["line_smoke", "production_smoke", "full"]
EXPECTED_TERMINAL_STATUS = {
    "line_smoke": "dynamics_complete",
    "production_smoke": "dynamics_complete",
    "full": "analysis_complete",
}
NEXT_PROFILE_ON_SUCCESS = {
    "line_smoke": "production_smoke",
    "production_smoke": "full",
    "full": "",
}
MANUAL_BATCH_DIRNAME = "manual_md_batches"


@dataclass(frozen=True)
class StageMeta:
    sequence: str
    md_profile: str
    cluster: str
    source_batch_csv: str
    selected_batch_csv: str
    reuse_pdb_from: str
    exclude_nodes: str
    expected_terminal_status: str
    next_profile_on_success: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sequence": self.sequence,
            "md_profile": self.md_profile,
            "cluster": self.cluster,
            "source_batch_csv": self.source_batch_csv,
            "selected_batch_csv": self.selected_batch_csv,
            "reuse_pdb_from": self.reuse_pdb_from,
            "exclude_nodes": self.exclude_nodes,
            "expected_terminal_status": self.expected_terminal_status,
            "next_profile_on_success": self.next_profile_on_success,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Selected batch rows cannot be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper()


def _normalize_manual_campaign_name(campaign: str, sequence: str, md_profile: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", campaign.strip()).strip("._-").lower()
    if cleaned:
        return cleaned[:80]
    safe_sequence = re.sub(r"[^A-Za-z0-9]+", "", sequence.upper())[:16] or "peptide"
    return f"manual_{md_profile}_{safe_sequence}".lower()


def _normalize_exclude_nodes(exclude_nodes: str) -> str:
    return ",".join(part.strip() for part in exclude_nodes.split(",") if part.strip())


def _select_sequence_rows(batch_csv: Path, sequence: str) -> list[dict[str, str]]:
    target = _normalize_sequence(sequence)
    rows = _read_csv(batch_csv)
    matches = [dict(row) for row in rows if _normalize_sequence(row.get("sequence", "")) == target]
    if not matches:
        raise ValueError(f"Sequence not found in batch CSV: {target}")
    if len(matches) != 1:
        raise ValueError(f"Sequence must appear exactly once in batch CSV: {target}")
    matches[0]["sequence"] = target
    return matches


def _write_json(path: Path, payload: dict[str, str]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _load_stage_meta(campaign_dir: Path) -> StageMeta:
    meta_path = campaign_dir / STAGE_META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing stage metadata file: {meta_path}")
    return StageMeta(**json.loads(meta_path.read_text(encoding="utf-8")))


def _bura_submit_command(sequence: str, exclude_nodes: str) -> str:
    command = "bash ./submit_chain.sh"
    if exclude_nodes:
        command += f" --exclude {exclude_nodes}"
    command += f" {sequence}"
    return command


def _next_commands_text(campaign_name: str, sequence: str, exclude_nodes: str) -> str:
    submit_command = _bura_submit_command(sequence, exclude_nodes)
    return f'''# Next BURA Commands

Run these commands on BURA after uploading this campaign directory.

```bash
cd ~/{campaign_name}
find . -type f -name "*.sh" -exec dos2unix {{}} \+
find . -type f -name "*.sh" -exec chmod u+x {{}} \+
module load gromacs/2023.2_g13.1_p3.10.5
bash ./preflight_bura.sh
{submit_command}
```
'''


def _copy_reused_pdb(reuse_campaign_dir: Path, campaign_dir: Path, sequence: str) -> None:
    source = reuse_campaign_dir / "PDBs" / f"{sequence}.pdb"
    if not source.exists():
        raise FileNotFoundError(f"Missing reusable PDB: {source}")
    destination = campaign_dir / "PDBs" / f"{sequence}.pdb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _next_stage_message(meta: StageMeta, job_root_status: str) -> str:
    if meta.md_profile == "line_smoke":
        if job_root_status == "dynamics_complete":
            return "Next recommended stage: production_smoke"
        return "Next recommended stage: rerun or debug line_smoke until dynamics_complete."
    if meta.md_profile == "production_smoke":
        if job_root_status == "dynamics_complete":
            return "Next recommended stage: full"
        return "Next recommended stage: rerun or debug production_smoke until dynamics_complete."
    if job_root_status == "analysis_complete":
        return "Full stage is ready for manual review, cgmd_label assignment, and make-md-ingest-csv."
    if job_root_status == "sasa_complete":
        return "Full stage reached SASA only; it is not ingest-ready yet."
    return "Full stage is not complete yet; rerun or debug until analysis_complete."


def prepare_md_stage(
    run_dir: Path,
    batch_csv: Path,
    sequence: str,
    campaign: str,
    md_profile: str,
    *,
    cluster: str = "bura",
    reuse_pdb_from: Path | None = None,
    exclude_nodes: str = "",
) -> tuple[Path, Path]:
    selected_rows = _select_sequence_rows(batch_csv, sequence)
    exclude_nodes = _normalize_exclude_nodes(exclude_nodes)

    temp_batch_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=f"{campaign}_selected_",
            dir=run_dir,
            delete=False,
        ) as handle:
            temp_batch_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
        campaign_dir = prepare_md_campaign(run_dir, temp_batch_path, campaign, cluster, md_profile)
    finally:
        if temp_batch_path is not None and temp_batch_path.exists():
            temp_batch_path.unlink()

    selected_batch_csv = campaign_dir / "selected_batch.csv"
    _write_csv(selected_batch_csv, selected_rows)

    if reuse_pdb_from is not None:
        _copy_reused_pdb(reuse_pdb_from, campaign_dir, selected_rows[0]["sequence"])
        build_pdbs(campaign_dir, validate_only=True)
    else:
        build_pdbs(campaign_dir, validate_only=False)

    meta = StageMeta(
        sequence=selected_rows[0]["sequence"],
        md_profile=md_profile,
        cluster=cluster,
        source_batch_csv=str(batch_csv),
        selected_batch_csv=str(selected_batch_csv.relative_to(campaign_dir)),
        reuse_pdb_from=str(reuse_pdb_from) if reuse_pdb_from else "",
        exclude_nodes=exclude_nodes,
        expected_terminal_status=EXPECTED_TERMINAL_STATUS[md_profile],
        next_profile_on_success=NEXT_PROFILE_ON_SUCCESS[md_profile],
    )
    _write_json(campaign_dir / STAGE_META_FILENAME, meta.to_dict())

    next_commands_path = campaign_dir / NEXT_COMMANDS_FILENAME
    next_commands_path.write_text(
        _next_commands_text(campaign_dir.name, meta.sequence, exclude_nodes),
        encoding="utf-8",
        newline="\n",
    )
    return campaign_dir, next_commands_path


def prepare_manual_md_stage(
    run_dir: Path,
    sequence: str,
    campaign: str,
    md_profile: str,
    *,
    cluster: str = "bura",
    reuse_pdb_from: Path | None = None,
    exclude_nodes: str = "",
) -> tuple[Path, Path, Path]:
    """Prepare an MD campaign for an operator-chosen peptide outside the AL loop."""
    normalized_sequence = _normalize_sequence(sequence)
    normalized_campaign = _normalize_manual_campaign_name(campaign, normalized_sequence, md_profile)
    campaign_dir = run_dir / "md_campaigns" / normalized_campaign
    next_commands_path = campaign_dir / NEXT_COMMANDS_FILENAME
    sandbox_marker = campaign_dir / "manual_md_sandbox.json"
    batch_dir = run_dir / MANUAL_BATCH_DIRNAME
    batch_csv = batch_dir / f"{normalized_campaign}.csv"

    if campaign_dir.exists():
        if sandbox_marker.exists():
            return campaign_dir, next_commands_path, batch_csv
        raise FileExistsError(
            f"MD campaign already exists and is not a manual sandbox: {campaign_dir}"
        )

    batch_dir.mkdir(parents=True, exist_ok=True)
    if not batch_csv.exists():
        _write_csv(
            batch_csv,
            [
                {
                    "sequence": normalized_sequence,
                    "round_id": "manual",
                    "acquisition_strategy": "manual_md_sandbox",
                    "pred_mean": "",
                    "pred_std": "",
                    "pred_entropy": "",
                    "pred_mutual_information": "",
                    "acquisition_score": "",
                }
            ],
        )
    campaign_dir, next_commands_path = prepare_md_stage(
        run_dir,
        batch_csv,
        normalized_sequence,
        normalized_campaign,
        md_profile,
        cluster=cluster,
        reuse_pdb_from=reuse_pdb_from,
        exclude_nodes=exclude_nodes,
    )
    sandbox_meta = {
        "purpose": "manual_md_sandbox",
        "sequence": normalized_sequence,
        "campaign": normalized_campaign,
        "md_profile": md_profile,
        "source_batch_csv": str(batch_csv),
        "safe_to_ingest": "false",
        "note": "Operator-created MD test campaign; not generated by AL proposal/discovery.",
    }
    _write_json(campaign_dir / "manual_md_sandbox.json", sandbox_meta)
    return campaign_dir, next_commands_path, batch_csv


def _copy_tree_contents(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Staged package directory does not exist: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.rglob("*"):
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def finalize_md_stage(campaign_dir: Path, staged_package_dir: Path | None = None) -> tuple[Path, dict[str, str], str]:
    if staged_package_dir is not None:
        meta = _load_stage_meta(campaign_dir)
        _copy_tree_contents(staged_package_dir, campaign_dir / "packages" / meta.sequence)
    review_path = parse_md_results(campaign_dir)
    meta = _load_stage_meta(campaign_dir)
    review_rows = _read_csv(review_path)
    review_row = next((row for row in review_rows if row["sequence"] == meta.sequence), None)
    if review_row is None:
        raise ValueError(f"Sequence {meta.sequence} missing from review CSV: {review_path}")
    return review_path, review_row, _next_stage_message(meta, review_row["job_root_status"])


def md_ladder_status(run_dir: Path, sequence: str) -> dict[str, object]:
    target = _normalize_sequence(sequence)
    campaigns_root = run_dir / "md_campaigns"
    campaigns: list[dict[str, str]] = []
    successful_profiles: set[str] = set()

    if campaigns_root.exists():
        for campaign_dir in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
            meta_path = campaign_dir / STAGE_META_FILENAME
            if not meta_path.exists():
                continue
            meta = _load_stage_meta(campaign_dir)
            if meta.sequence != target:
                continue

            job_root_status = "package_prepared"
            review_path = campaign_dir / "md_review.csv"
            if review_path.exists():
                for row in _read_csv(review_path):
                    if row["sequence"] == target:
                        job_root_status = row["job_root_status"]
                        break

            campaigns.append(
                {
                    "campaign": campaign_dir.name,
                    "md_profile": meta.md_profile,
                    "job_root_status": job_root_status,
                }
            )
            if job_root_status == EXPECTED_TERMINAL_STATUS[meta.md_profile]:
                successful_profiles.add(meta.md_profile)

    next_profile = ""
    for profile in GUIDED_PROFILE_ORDER:
        if profile not in successful_profiles:
            next_profile = profile
            break

    return {
        "sequence": target,
        "campaigns": campaigns,
        "next_profile": next_profile,
        "ready_for_review": "full" in successful_profiles,
    }
