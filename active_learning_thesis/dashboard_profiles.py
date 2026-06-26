from __future__ import annotations

import json
import os
from pathlib import Path

PROFILE_ENV_VAR = "ACTIVE_LEARNING_THESIS_CLUSTER_PROFILES"
DEFAULT_CONFIG_DIRNAME = ".active_learning_thesis"
DEFAULT_PROFILE_FILENAME = "cluster_profiles.json"
SUPPORTED_CLUSTERS = ("supek", "bura")
REQUIRED_PROFILE_FIELDS = {
    "supek": (
        "host",
        "username",
        "repo_path",
        "scratch_run_root",
        "log_root",
        "conda_init_path",
        "env_name",
        "scheduler",
    ),
    "bura": (
        "host",
        "username",
        "campaign_root",
        "default_exclude_nodes",
        "module_load",
        "scheduler",
    ),
}


def default_cluster_profile_path() -> Path:
    override = os.environ.get(PROFILE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_CONFIG_DIRNAME / DEFAULT_PROFILE_FILENAME


def cluster_profile_template() -> dict[str, dict[str, str]]:
    return {
        "supek": {
            "enabled": "false",
            "host": "supek.example.invalid",
            "username": "your_user",
            "repo_path": "~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly",
            "scratch_run_root": "/lustre/scratch/$USER/ml_peptide_self_assembly_runs",
            "log_root": "~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly/supek_logs",
            "conda_init_path": "~/miniforge3/etc/profile.d/conda.sh",
            "env_name": "ml_peptide_self_assembly",
            "scheduler": "pbs",
            "default_branch": "codex/active-learning-thesis",
            "default_queue": "gpu",
            "default_walltime": "01:00:00",
            "pbs_select": "select=1:ncpus=4:ngpus=1:mem=40GB",
            "module_load": "",
        },
        "bura": {
            "enabled": "false",
            "host": "bura.example.invalid",
            "username": "your_user",
            "campaign_root": "~",
            "default_exclude_nodes": "",
            "module_load": "module load gromacs/2023.2_g13.1_p3.10.5",
            "scheduler": "slurm",
        },
    }


def _normalize_profile_payload(payload: dict[str, object] | None) -> dict[str, dict[str, str]]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for cluster_name, raw_values in payload.items():
        if cluster_name not in SUPPORTED_CLUSTERS or not isinstance(raw_values, dict):
            continue
        normalized[cluster_name] = {str(key): str(value) for key, value in raw_values.items()}
    return normalized


def load_cluster_profiles(profile_path: Path | None = None) -> dict[str, object]:
    path = (profile_path or default_cluster_profile_path()).expanduser()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "profiles": {},
            "template": cluster_profile_template(),
        }

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    profiles = _normalize_profile_payload(payload if isinstance(payload, dict) else None)
    return {
        "path": str(path),
        "exists": True,
        "profiles": profiles,
        "template": cluster_profile_template(),
    }



def _as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}



def validate_cluster_profile(cluster_name: str, profile: dict[str, str]) -> list[str]:
    required = REQUIRED_PROFILE_FIELDS.get(cluster_name, ())
    missing = [field for field in required if not str(profile.get(field, "")).strip()]
    scheduler = str(profile.get("scheduler", "")).strip().lower()
    if cluster_name == "supek" and scheduler not in {"pbs", ""}:
        missing.append("scheduler=pbs")
    if cluster_name == "bura" and scheduler not in {"slurm", ""}:
        missing.append("scheduler=slurm")
    return missing



def profile_rows(profiles_payload: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    profiles = profiles_payload.get("profiles", {}) if isinstance(profiles_payload, dict) else {}
    for cluster_name in SUPPORTED_CLUSTERS:
        profile = profiles.get(cluster_name, {}) if isinstance(profiles, dict) else {}
        enabled = _as_bool(profile.get("enabled", "true")) if profile else False
        missing = validate_cluster_profile(cluster_name, profile) if profile else list(REQUIRED_PROFILE_FIELDS[cluster_name])
        rows.append(
            {
                "cluster": cluster_name,
                "configured": "yes" if profile else "no",
                "enabled": "yes" if enabled else "no",
                "host": str(profile.get("host", "")),
                "username": str(profile.get("username", "")),
                "scheduler": str(profile.get("scheduler", "")),
                "missing_fields": ", ".join(missing),
            }
        )
    return rows



def get_cluster_profile(profiles_payload: dict[str, object], cluster_name: str) -> dict[str, str] | None:
    profiles = profiles_payload.get("profiles", {}) if isinstance(profiles_payload, dict) else {}
    profile = profiles.get(cluster_name) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        return None
    normalized = {str(key): str(value) for key, value in profile.items()}
    if not _as_bool(normalized.get("enabled", "true")):
        return None
    if validate_cluster_profile(cluster_name, normalized):
        return None
    return normalized

