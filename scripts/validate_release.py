"""Validate the cleaned thesis release repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PATHS = [
    "active_learning_thesis",
    "tests",
    "SA_ML_predictive/data/data_SA.csv",
    "SA_ML_generative/genetic_algorithm_library.py",
    "thesis_results/THESIS_RESULTS_HANDOFF_20260623/README.md",
    "thesis_results/THESIS_RESULTS_HANDOFF_20260623/00_overview/CLAIMS_AND_CAVEATS.md",
    "thesis_results/THESIS_RESULTS_HANDOFF_20260623/cgmd_peptide_evidence/phase3_cgmd_peptide_evidence.csv",
    "thesis_results/THESIS_RESULTS_HANDOFF_20260623/cgmd_peptide_evidence/phase4_primary_cgmd_peptide_evidence.csv",
    "thesis_reporting/README.md",
    "models/phase4_ap_sp_fixed_split_ensemble/model_manifest.json",
    "models/phase4_ap_sp_fixed_split_ensemble/member_calibrations.json",
    "models/phase4_ap_sp_fixed_split_ensemble/ensemble/ap_sp_member_00.h5",
    "models/phase3_round001_pre_proposal/predictive_entropy/pre_proposal/metrics.json",
    "models/phase3_round001_pre_proposal/predictive_entropy/pre_proposal/ensemble/ap_sp_member_00.h5",
    "models/phase3_round001_pre_proposal/family_qbc/pre_proposal/family/AP_SP.h5",
    "models/phase5_initial_replay_point_000/outer_1/initial_10/random/models/replay_point_000/embedding_manifest.json",
    "models/phase5_initial_replay_point_000/outer_1/initial_10/random/models/replay_point_000/ensemble/ap_sp_member_00.h5",
    "THESIS_METHODOLOGY_TECHNICAL_SUMMARY.md",
    "THESIS_RESULTS_DISCUSSION_HANDOFF.md",
    "PHASE5_RESULTS_SUMMARY.md",
]

EXCLUDED_SUFFIXES = {
    ".h5",
    ".keras",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".ckpt",
    ".xtc",
    ".trr",
    ".tpr",
    ".edr",
    ".cpt",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
}

ALLOWED_MODEL_PREFIXES = {
    "models/phase3_round001_pre_proposal/",
    "models/phase4_ap_sp_fixed_split_ensemble/",
    "models/phase5_initial_replay_point_000/",
}

SECRET_NAME_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\.env",
        r"id_rsa",
        r"id_ed25519",
        r"token",
        r"secret",
        r"password",
        r"credential",
    ]
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".pbs",
    ".sh",
    ".tex",
    ".cff",
}

SENSITIVE_TEXT_PATTERNS = [
    re.compile(r"BEGIN (?:RSA|OPENSSH|DSA|EC) PRIVATE KEY"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.parts
    )


def is_allowed_model_binary(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in ALLOWED_MODEL_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    for rel in EXPECTED_PATHS:
        if not (ROOT / rel).exists():
            failures.append(f"missing expected path: {rel}")

    files = iter_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        suffixes = [s.lower() for s in path.suffixes]
        if any(s in EXCLUDED_SUFFIXES for s in suffixes) and not is_allowed_model_binary(rel):
            failures.append(f"excluded heavy/binary artefact present: {rel}")
        if any(pattern.search(path.name) for pattern in SECRET_NAME_PATTERNS):
            failures.append(f"potential secret-like filename present: {rel}")
        if path.stat().st_size > 50 * 1024 * 1024:
            warnings.append(f"large file over 50 MB: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < 5 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SENSITIVE_TEXT_PATTERNS:
                if pattern.search(text):
                    failures.append(f"potential secret-like text in: {rel}")

    if args.write_manifest:
        manifest = ROOT / "RELEASE_FILE_MANIFEST.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["relative_path", "size_bytes", "sha256"])
            for path in files:
                if path.name == "RELEASE_FILE_MANIFEST.csv":
                    continue
                writer.writerow([
                    path.relative_to(ROOT).as_posix(),
                    path.stat().st_size,
                    sha256_file(path),
                ])

    print(f"release_root={ROOT}")
    print(f"file_count={len(files)}")
    print(f"warnings={len(warnings)}")
    for item in warnings:
        print(f"WARNING: {item}")
    print(f"failures={len(failures)}")
    for item in failures:
        print(f"FAIL: {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
