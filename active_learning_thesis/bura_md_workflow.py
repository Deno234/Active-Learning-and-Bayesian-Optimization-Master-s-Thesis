from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from active_learning_thesis.bura_autopilot import run_bura_full_autopilot
from active_learning_thesis.md_orchestrator import STAGE_META_FILENAME, prepare_manual_md_stage, prepare_md_stage


PAYLOAD_VERSION = "bura-md-workflow-v1"


def _payload_dict(payload_json: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload_json, dict):
        return dict(payload_json)
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError("BURA MD workflow payload must be a JSON object.")
    return payload


def _metadata(payload: dict[str, Any]) -> dict[str, str]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def _stage_meta_matches(campaign_dir: Path, payload: dict[str, Any]) -> bool:
    meta_path = campaign_dir / STAGE_META_FILENAME
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    metadata = _metadata(payload)
    sequence_profile_match = (
        str(meta.get("sequence", "")).strip().upper() == str(payload.get("sequence", "")).strip().upper()
        and str(meta.get("md_profile", "")) == str(payload.get("md_profile", ""))
    )
    if not sequence_profile_match:
        return False
    if str(payload.get("source_type", "")) == "real_al_proposal":
        return str(meta.get("source_batch_csv", "")) == metadata.get("source_path", "")
    return True


def _prepare_campaign(payload: dict[str, Any]) -> Path:
    metadata = _metadata(payload)
    sequence = str(payload.get("sequence", "")).strip().upper()
    source_type = str(payload.get("source_type", ""))
    md_profile = str(payload.get("md_profile", ""))
    campaign_name = str(payload.get("campaign_name", ""))
    campaign_dir = Path(str(payload.get("campaign_dir", ""))).resolve()
    run_dir_text = metadata.get("run_dir", "")
    run_dir = Path(run_dir_text).resolve() if run_dir_text else Path()
    if not sequence:
        raise ValueError("Missing peptide sequence in BURA MD workflow payload.")
    if not run_dir_text:
        raise ValueError("Missing run_dir in BURA MD workflow payload metadata.")
    if campaign_dir.exists():
        if _stage_meta_matches(campaign_dir, payload):
            print(f"Checkpoint prepare: matching campaign already exists at {campaign_dir}; reusing it.", flush=True)
            return campaign_dir
        raise FileExistsError(f"Campaign directory already exists and does not match payload: {campaign_dir}")
    print(f"Checkpoint prepare: preparing {md_profile} campaign for {sequence}.", flush=True)
    if source_type == "real_al_proposal" and metadata.get("source_path"):
        prepared_dir, _next_commands = prepare_md_stage(
            run_dir,
            Path(metadata["source_path"]),
            sequence,
            campaign_name,
            md_profile,
            cluster="bura",
        )
        return prepared_dir
    prepared_dir, _next_commands, _batch_csv = prepare_manual_md_stage(
        run_dir,
        sequence,
        campaign_name,
        md_profile,
        cluster="bura",
    )
    return prepared_dir


def run_bura_md_workflow_from_payload(payload_json: str | dict[str, Any]) -> dict[str, str]:
    payload = _payload_dict(payload_json)
    metadata = _metadata(payload)
    if str(payload.get("payload_version", "")) not in {"", PAYLOAD_VERSION}:
        raise ValueError(f"Unsupported BURA MD workflow payload version: {payload.get('payload_version')}")
    run_root_text = metadata.get("run_root", "")
    run_root = Path(run_root_text).resolve() if run_root_text else Path()
    if not run_root_text:
        raise ValueError("Missing run_root in BURA MD workflow payload metadata.")
    sequence = str(payload.get("sequence", "")).strip().upper()
    exclude_nodes = str(metadata.get("exclude_nodes", ""))
    print("Checkpoint prepare: starting local campaign preparation stage.", flush=True)
    campaign_dir = _prepare_campaign(payload)
    print("Checkpoint autopilot: starting existing BURA full autopilot backend.", flush=True)
    summary = run_bura_full_autopilot(
        run_root=run_root,
        campaign_dir=campaign_dir,
        sequence=sequence,
        exclude_nodes=exclude_nodes,
    )
    print("Checkpoint finalize: BURA full autopilot returned local finalization summary.", flush=True)
    return {str(key): str(value) for key, value in summary.items()}
