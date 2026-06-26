from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import warnings
from pathlib import Path
from typing import Iterable

from active_learning_thesis.config import RunConfig
from active_learning_thesis.md_review_evidence import LABEL_REVIEW_FIELDS
from active_learning_thesis.paths import MD_BURA_TEMPLATE_DIR

REQUIRED_BATCH_FIELDS = {
    "sequence",
    "round_id",
    "acquisition_strategy",
    "pred_mean",
    "pred_std",
    "pred_entropy",
    "pred_mutual_information",
    "acquisition_score",
}

CANONICAL_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
DYNAMICS_CHAIN = [
    "0_CG_pol_sysprep.sh",
    "1_Softcore_mini_a.sh",
    "2_Softcore_mini_b.sh",
    "3_Steepest_mini_a.sh",
    "4_Steepest_mini_b.sh",
    "5_Berendsen_equi_a.sh",
    "6_Berendsen_equi_b.sh",
    "7_Parr_Rah_equi_a.sh",
    "8_Parr_Rah_equi_b.sh",
    "9_Dynamics_a.sh",
    "10_Dynamics_b.sh",
]
ANALYSIS_CHAIN = [
    "11_SASA_and_FrameDump.sh",
    "12_AP_calc.sh",
    "13_Extract_last10ns_for_paper_APcontact.sh",
]
FULL_CHAIN = [*DYNAMICS_CHAIN, *ANALYSIS_CHAIN]
TERMINAL_SCRIPT = "12_AP_calc.sh"
PROFILE_CHOICES = ("line_smoke", "production_smoke", "full")
AP_TARGET_NS = (5, 12, 25, 50, 100, 200)
AP_CONTACT_CUTOFF_NM = 0.6
AP_CONTACT_SENSITIVITY_CUTOFFS_NM = (0.35, 0.40, 0.45, 0.50, 0.60)
PAPER_AP_CONTACT_MIDPOINT_NM = 0.4425
PAPER_AP_CONTACT_STEEPNESS = 12.0
PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM = 0.4
PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM = 1.2
PAPER_PATH_AP_CONTACT_EXACT_MAX_MOLECULES = 16
PAPER_PATH_AP_CONTACT_BEAM_WIDTH = 256
PAPER_PATH_AP_CONTACT_BRANCHING = 8
PAPER_PATH_LAST10_START_NS = 190
PAPER_PATH_LAST10_END_NS = 200
PAPER_PATH_LAST10_STEP_NS = 1
PAPER_AP_SASA_LAST10_WINDOW_NS = 10.0
PAPER_AP_SASA_GROUP_NAME = "Peptide_non_solvent"
OPERATIONAL_AP_SASA_THRESHOLD = 1.75
OPERATIONAL_PAPER_PATH_AP_CONTACT_THRESHOLD = 0.5
BURA_DEFAULT_EXCLUDE_NODES = ""


def operational_cgmd_threshold_label(
    ap_sasa_200ns: float,
    paper_path_ap_contact_last10ns: float,
) -> int:
    """Return the threshold label used by the retained Phase 3 review files.

    This helper documents and tests the operational rule. The workflow still
    requires a human-reviewed ``cgmd_label`` and does not assign labels
    automatically.
    """

    return int(
        float(ap_sasa_200ns) >= OPERATIONAL_AP_SASA_THRESHOLD
        and float(paper_path_ap_contact_last10ns)
        >= OPERATIONAL_PAPER_PATH_AP_CONTACT_THRESHOLD
    )


DEFAULT_BURA_BENCHMARK_LAYOUTS = {
    "1n_1mpi_48omp": {"nodes": 1, "ntasks_per_node": 1, "cpus_per_task": 48},
    "1n_2mpi_24omp": {"nodes": 1, "ntasks_per_node": 2, "cpus_per_task": 24},
    "1n_4mpi_12omp": {"nodes": 1, "ntasks_per_node": 4, "cpus_per_task": 12},
    "1n_6mpi_8omp": {"nodes": 1, "ntasks_per_node": 6, "cpus_per_task": 8},
    "1n_8mpi_6omp": {"nodes": 1, "ntasks_per_node": 8, "cpus_per_task": 6},
    "2n_2mpi_24omp": {"nodes": 2, "ntasks_per_node": 2, "cpus_per_task": 24},
    "2n_4mpi_12omp": {"nodes": 2, "ntasks_per_node": 4, "cpus_per_task": 12},
    "2n_6mpi_8omp": {"nodes": 2, "ntasks_per_node": 6, "cpus_per_task": 8},
    "2n_8mpi_6omp": {"nodes": 2, "ntasks_per_node": 8, "cpus_per_task": 6},
    "4n_4mpi_12omp": {"nodes": 4, "ntasks_per_node": 4, "cpus_per_task": 12},
    "4n_6mpi_8omp": {"nodes": 4, "ntasks_per_node": 6, "cpus_per_task": 8},
    "4n_8mpi_6omp": {"nodes": 4, "ntasks_per_node": 8, "cpus_per_task": 6},
    "6n_4mpi_12omp": {"nodes": 6, "ntasks_per_node": 4, "cpus_per_task": 12},
    "6n_6mpi_8omp": {"nodes": 6, "ntasks_per_node": 6, "cpus_per_task": 8},
    "6n_8mpi_6omp": {"nodes": 6, "ntasks_per_node": 8, "cpus_per_task": 6},
}
BENCHMARK_FIELDS = [
    "layout",
    "nodes",
    "ntasks_per_node",
    "cpus_per_task",
    "total_mpi_tasks",
    "total_requested_cpus",
    "status",
    "md_runtime_wall_hms",
    "md_runtime_wall_seconds",
    "md_runtime_core_seconds",
    "md_runtime_ns_per_day",
    "observed_mpi_processes",
    "observed_omp_threads",
    "log_file",
    "stderr_file",
]
MDP_FILENAMES = [
    "martini_22P_vacuum_mini.mdp",
    "martini_22P_mini_soft.mdp",
    "martini_22P_mini_steep.mdp",
    "martini_22P_equi_Berendsen.mdp",
    "martini_22P_equi_ParrRah.mdp",
    "martini_22P_md.mdp",
]
PRODUCTION_MDP = "martini_22P_md.mdp"
NSTEP_PATTERN = re.compile(r"(?m)^(\s*nsteps\s*=\s*)(\d+)(.*)$")
REQUIRED_TEMPLATE_ASSETS = [
    "martinize.py",
    "triple-w.py",
    *MDP_FILENAMES,
    "martini_v2.2refP.itp",
    "water.gro",
]
MANIFEST_FIELDS = [
    "sequence",
    "round_id",
    "acquisition_strategy",
    "pred_mean",
    "pred_std",
    "pred_entropy",
    "pred_mutual_information",
    "acquisition_score",
    "campaign",
    "cluster",
    "md_profile",
    "package_dir",
    "pdb_path",
    "pdb_status",
]
REVIEW_FIELDS = [
    "sequence",
    "round_id",
    "campaign",
    "cluster",
    "md_profile",
    "package_dir",
    "job_root_status",
    "ap_5ns",
    "ap_12ns",
    "ap_25ns",
    "ap_50ns",
    "ap_100ns",
    "ap_200ns",
    "ap_contact_5ns",
    "ap_contact_12ns",
    "ap_contact_25ns",
    "ap_contact_50ns",
    "ap_contact_100ns",
    "ap_contact_200ns",
    "paper_ap_contact_5ns",
    "paper_ap_contact_12ns",
    "paper_ap_contact_25ns",
    "paper_ap_contact_50ns",
    "paper_ap_contact_100ns",
    "paper_ap_contact_200ns",
    "ap_contact_same_paper_formula_5ns",
    "ap_contact_same_paper_formula_12ns",
    "ap_contact_same_paper_formula_25ns",
    "ap_contact_same_paper_formula_50ns",
    "ap_contact_same_paper_formula_100ns",
    "ap_contact_same_paper_formula_200ns",
    "paper_path_ap_contact_5ns",
    "paper_path_ap_contact_12ns",
    "paper_path_ap_contact_25ns",
    "paper_path_ap_contact_50ns",
    "paper_path_ap_contact_100ns",
    "paper_path_ap_contact_200ns",
    "paper_path_ap_contact_last10ns_mean",
    "paper_path_ap_contact_last10ns_sd",
    "paper_path_ap_contact_last10ns_n_frames",
    "paper_path_ap_contact_last10ns_method",
    "paper_path_ap_contact_last10ns_status",
    "paper_ap_sasa_last10ns_mean",
    "paper_ap_sasa_last10ns_sd",
    "paper_ap_sasa_last10ns_n_frames",
    "paper_ap_sasa_initial_sasa",
    "paper_ap_sasa_initial_source",
    "paper_ap_sasa_final10_mean_sasa",
    "paper_ap_sasa_status",
    "paper_ap_sasa_method",
    "paper_ap_sasa_group_selection",
    "cluster_largest_fraction_200ns",
    "cluster_count_200ns",
    "cluster_singleton_fraction_200ns",
    "cluster_mean_contacts_200ns",
    "md_runtime_wall_hms",
    "md_runtime_wall_seconds",
    "md_runtime_core_seconds",
    "md_runtime_ns_per_day",
    "sasa_file",
    "ap_file",
    "ap_contact_file",
    "paper_ap_contact_file",
    "ap_contact_same_paper_formula_file",
    "paper_path_ap_contact_file",
    "paper_path_ap_contact_last10ns_file",
    "paper_path_ap_contact_last10ns_script",
    "paper_path_ap_contact_last10ns_status_file",
    "paper_ap_sasa_last10ns_file",
    "paper_ap_sasa_recompute_script",
    "paper_ap_sasa_status_file",
    "aggregate_summary_file",
    "review_notes",
    "cgmd_label",
    *LABEL_REVIEW_FIELDS,
]


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_config(run_dir: Path) -> RunConfig:
    return RunConfig.load(run_dir / "config.json")


def _campaign_paths(run_dir: Path, campaign: str) -> dict[str, Path]:
    campaign_dir = run_dir / "md_campaigns" / campaign
    return {
        "campaign_dir": campaign_dir,
        "manifest": campaign_dir / "manifest.csv",
        "sequences": campaign_dir / "sequences.txt",
        "pdbs": campaign_dir / "PDBs",
        "packages": campaign_dir / "packages",
        "submit": campaign_dir / "submit_chain.sh",
        "preflight": campaign_dir / "preflight_bura.sh",
        "readme": campaign_dir / "README_BURA.md",
        "review": campaign_dir / "md_review.csv",
        "ingest": campaign_dir / "cgmd_ingest.csv",
    }


def _validate_cluster(cluster: str) -> str:
    normalized = cluster.strip().lower()
    if normalized != "bura":
        raise ValueError("Only cluster='bura' is supported in MD v1.")
    return normalized


def _validate_md_profile(md_profile: str) -> str:
    normalized = md_profile.strip().lower()
    if normalized not in PROFILE_CHOICES:
        raise ValueError(
            "md_profile must be one of: " + ", ".join(PROFILE_CHOICES)
        )
    return normalized


def _profile_has_analysis(md_profile: str) -> bool:
    return md_profile == "full"


def _chain_for_profile(md_profile: str) -> list[str]:
    return FULL_CHAIN if _profile_has_analysis(md_profile) else DYNAMICS_CHAIN


def _template_source(md_profile: str = "full") -> Path:
    if not MD_BURA_TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Missing MD BURA template directory: {MD_BURA_TEMPLATE_DIR}")
    for asset in REQUIRED_TEMPLATE_ASSETS:
        asset_path = MD_BURA_TEMPLATE_DIR / asset
        if not asset_path.exists():
            raise FileNotFoundError(f"Missing required MD template asset: {asset_path}")
    if _profile_has_analysis(md_profile):
        if not (MD_BURA_TEMPLATE_DIR / "11_SASA_and_FrameDump.sh").exists():
            raise FileNotFoundError("Missing canonical analysis script 11_SASA_and_FrameDump.sh")
        if not (MD_BURA_TEMPLATE_DIR / TERMINAL_SCRIPT).exists():
            raise FileNotFoundError(f"Missing canonical terminal script {TERMINAL_SCRIPT}")
    return MD_BURA_TEMPLATE_DIR


def _validate_batch_rows(rows: list[dict[str, str]], config: RunConfig) -> list[dict[str, str]]:
    if not rows:
        raise ValueError("Batch CSV is empty.")
    fieldnames = set(rows[0])
    missing = REQUIRED_BATCH_FIELDS - fieldnames
    if missing:
        raise ValueError(f"Batch CSV is missing required columns: {', '.join(sorted(missing))}")
    sequence_counts: dict[str, int] = {}
    validated: list[dict[str, str]] = []
    for row in rows:
        sequence = row["sequence"].strip().upper()
        if not CANONICAL_SEQUENCE.fullmatch(sequence):
            raise ValueError(f"Sequence contains non-canonical amino acids: {sequence}")
        if len(sequence) > config.max_initial_peptide_length:
            raise ValueError(
                f"Sequence exceeds configured maximum length {config.max_initial_peptide_length}: {sequence}"
            )
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        normalized = dict(row)
        normalized["sequence"] = sequence
        normalized["_sequence_occurrence"] = str(sequence_counts[sequence])
        validated.append(normalized)
    duplicates = {sequence: count for sequence, count in sequence_counts.items() if count > 1}
    if duplicates:
        details = ", ".join(f"{sequence} x{count}" for sequence, count in sorted(duplicates.items()))
        warnings.warn(
            "Duplicate peptide sequence(s) in MD batch will be simulated as independent replicate packages: "
            + details,
            RuntimeWarning,
            stacklevel=2,
        )
    return validated


def _package_dir(paths: dict[str, Path], sequence: str) -> Path:
    return paths["packages"] / sequence


def _package_name_for_sequence(sequence: str, occurrence: int) -> str:
    if occurrence <= 1:
        return sequence
    return f"{sequence}__rep{occurrence:02d}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _chmod_shell(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def _sequence_file_text(rows: Iterable[dict[str, str]]) -> str:
    return "\n".join(row["sequence"] for row in rows) + "\n"


def _empty_review_row(manifest_row: dict[str, str]) -> dict[str, str]:
    return {
        "sequence": manifest_row["sequence"],
        "round_id": manifest_row["round_id"],
        "campaign": manifest_row["campaign"],
        "cluster": manifest_row["cluster"],
        "md_profile": manifest_row["md_profile"],
        "package_dir": manifest_row["package_dir"],
        "job_root_status": "package_prepared",
        "ap_5ns": "",
        "ap_12ns": "",
        "ap_25ns": "",
        "ap_50ns": "",
        "ap_100ns": "",
        "ap_200ns": "",
        "ap_contact_5ns": "",
        "ap_contact_12ns": "",
        "ap_contact_25ns": "",
        "ap_contact_50ns": "",
        "ap_contact_100ns": "",
        "ap_contact_200ns": "",
        "paper_ap_contact_5ns": "",
        "paper_ap_contact_12ns": "",
        "paper_ap_contact_25ns": "",
        "paper_ap_contact_50ns": "",
        "paper_ap_contact_100ns": "",
        "paper_ap_contact_200ns": "",
        "ap_contact_same_paper_formula_5ns": "",
        "ap_contact_same_paper_formula_12ns": "",
        "ap_contact_same_paper_formula_25ns": "",
        "ap_contact_same_paper_formula_50ns": "",
        "ap_contact_same_paper_formula_100ns": "",
        "ap_contact_same_paper_formula_200ns": "",
        "paper_path_ap_contact_5ns": "",
        "paper_path_ap_contact_12ns": "",
        "paper_path_ap_contact_25ns": "",
        "paper_path_ap_contact_50ns": "",
        "paper_path_ap_contact_100ns": "",
        "paper_path_ap_contact_200ns": "",
        "paper_path_ap_contact_last10ns_mean": "",
        "paper_path_ap_contact_last10ns_sd": "",
        "paper_path_ap_contact_last10ns_n_frames": "",
        "paper_path_ap_contact_last10ns_method": "",
        "paper_path_ap_contact_last10ns_status": "",
        "paper_ap_sasa_last10ns_mean": "",
        "paper_ap_sasa_last10ns_sd": "",
        "paper_ap_sasa_last10ns_n_frames": "",
        "paper_ap_sasa_initial_sasa": "",
        "paper_ap_sasa_initial_source": "",
        "paper_ap_sasa_final10_mean_sasa": "",
        "paper_ap_sasa_status": "",
        "paper_ap_sasa_method": "",
        "paper_ap_sasa_group_selection": "",
        "cluster_largest_fraction_200ns": "",
        "cluster_count_200ns": "",
        "cluster_singleton_fraction_200ns": "",
        "cluster_mean_contacts_200ns": "",
        "md_runtime_wall_hms": "",
        "md_runtime_wall_seconds": "",
        "md_runtime_core_seconds": "",
        "md_runtime_ns_per_day": "",
        "sasa_file": "",
        "ap_file": "",
        "ap_contact_file": "",
        "paper_ap_contact_file": "",
        "ap_contact_same_paper_formula_file": "",
        "paper_path_ap_contact_file": "",
        "paper_path_ap_contact_last10ns_file": "",
        "paper_path_ap_contact_last10ns_script": "",
        "paper_path_ap_contact_last10ns_status_file": "",
        "paper_ap_sasa_last10ns_file": "",
        "paper_ap_sasa_recompute_script": "",
        "paper_ap_sasa_status_file": "",
        "aggregate_summary_file": "",
        "review_notes": "",
        "cgmd_label": "",
        **{field: "" for field in LABEL_REVIEW_FIELDS},
    }

def _env_script_text() -> str:
    return """#!/bin/bash -l
if [ -f /etc/profile ]; then source /etc/profile >/dev/null 2>&1 || true; fi
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh >/dev/null 2>&1 || true; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh >/dev/null 2>&1 || true; fi
set -euo pipefail
module purge >/dev/null 2>&1 || true
module load gromacs/2023.2_g13.1_p3.10.5 >/dev/null 2>&1 || true

BURA_GMX_BIN="/home/.OPT/GROMACS/gromacs-2023.2-gcc13.1-p3.10.5/bin"
BURA_INTEL_MPI_BIN="/opt/intel/impi/2021.17.1/mpi/2021.17/bin"
BURA_RUNTIME_LIB_DIRS=(
  "/home/.OPT/GROMACS/gromacs-2023.2-gcc13.1-p3.10.5/lib"
  "/home/.OPT/GCC/gcc-11.5.0/lib64"
  "/home/.OPT/IntelMKL/2025.3.0.462/mkl/2025.3/lib"
  "/opt/intel/impi/2021.17.1/mpi/2021.17/lib"
  "/opt/intel/impi/2021.17.1/mpi/2021.17/lib/release"
  "/opt/intel/oneapi/mkl/2025.3/lib"
  "/opt/intel/oneapi/mkl/2025.3/lib/intel64"
  "/opt/intel/oneapi/compiler/2025.3/lib"
  "/opt/intel/oneapi/compiler/2025.3/lib/intel64_lin"
)
if [[ -x "$BURA_GMX_BIN/gmx_mpi" ]]; then
  export PATH="$BURA_GMX_BIN:$PATH"
fi
if [[ -x "$BURA_INTEL_MPI_BIN/mpirun" ]]; then
  export PATH="$BURA_INTEL_MPI_BIN:$PATH"
fi
for runtime_lib_dir in "${BURA_RUNTIME_LIB_DIRS[@]}"; do
  if [[ -d "$runtime_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$runtime_lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done
if command -v gmx_mpi >/dev/null 2>&1; then
  export GMX_CMD=${GMX_CMD:-$(command -v gmx_mpi)}
else
  echo "GROMACS gmx_mpi unavailable. Tried module gromacs/2023.2_g13.1_p3.10.5 and $BURA_GMX_BIN/gmx_mpi." >&2
  exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
  echo "MPI launcher mpirun unavailable. Tried module gromacs/2023.2_g13.1_p3.10.5 and $BURA_INTEL_MPI_BIN/mpirun." >&2
  exit 1
fi

ensure_python2_runtime() {
  local candidate
  local candidates=(
    "python/Python-2.7.18"
    "Python/2.7.18"
    "python/2.7.18"
    "Python/2.7"
    "python/2.7"
  )
  if command -v python2 >/dev/null 2>&1; then
    return 0
  fi
  for candidate in "${candidates[@]}"; do
    if module load "$candidate" >/dev/null 2>&1 && command -v python2 >/dev/null 2>&1; then
      return 0
    fi
  done
  echo "Python 2 runtime unavailable. Load a BURA Python 2 module manually or set PY2_CMD before submission." >&2
  exit 1
}

ensure_python2_runtime
export PY2_CMD=${PY2_CMD:-python2}
"""


def _common_script_text() -> str:
    return """#!/bin/bash -l
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env_bura.sh"
set -euo pipefail

peptide_name_from_pdb() {
  local package_name
  local first_pdb
  package_name=$(basename "$SCRIPT_DIR")
  if [[ -f "$SCRIPT_DIR/${package_name}.pdb" ]]; then
    echo "$package_name"
    return 0
  fi
  first_pdb=$(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.pdb" ! -name "*_CG.pdb" | sort | head -n 1)
  if [[ -z "$first_pdb" ]]; then
    first_pdb=$(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.pdb" | sort | head -n 1)
  fi
  if [[ -z "$first_pdb" ]]; then
    echo "No .pdb file found in $SCRIPT_DIR" >&2
    return 1
  fi
  basename "$first_pdb" .pdb | sed 's/_CG$//'
}

nmol_from_sequence() {
  local peptide_name="$1"
  local length=${#peptide_name}
  if [[ "$length" -le 0 ]]; then
    echo "Invalid peptide length for $peptide_name" >&2
    return 1
  fi
  echo $((1200 / length))
}
"""


def _rewrite_nsteps(content: str, nsteps: int) -> str:
    rewritten, count = NSTEP_PATTERN.subn(rf"\g<1>{nsteps}\g<3>", content, count=1)
    if count != 1:
        raise ValueError("Unable to rewrite nsteps in MDP template.")
    return rewritten


def _copy_non_mdp_assets(package_dir: Path) -> None:
    template_dir = _template_source()
    for asset in REQUIRED_TEMPLATE_ASSETS:
        if asset in MDP_FILENAMES:
            continue
        shutil.copy2(template_dir / asset, package_dir / asset)


def _copy_profiled_mdp_files(package_dir: Path, md_profile: str) -> None:
    template_dir = _template_source(md_profile)
    for filename in MDP_FILENAMES:
        content = (template_dir / filename).read_text(encoding="utf-8")
        if md_profile == "line_smoke":
            content = _rewrite_nsteps(content, 200)
        elif md_profile == "production_smoke" and filename == PRODUCTION_MDP:
            content = _rewrite_nsteps(content, 200)
        _write_text(package_dir / filename, content)


def _resource_profile(md_profile: str, *, compute: bool) -> list[str]:
    if not compute:
        if md_profile == "line_smoke":
            return [
                "#SBATCH --time=00:30:00",
                "#SBATCH --partition=computes_thin",
                "#SBATCH --ntasks=1",
                "#SBATCH --mem-per-cpu=1G",
            ]
        if md_profile == "production_smoke":
            return [
                "#SBATCH --time=02:00:00",
                "#SBATCH --partition=computes_thin",
                "#SBATCH --ntasks=1",
                "#SBATCH --mem-per-cpu=1G",
            ]
        return [
            "#SBATCH --time=21:00:00",
            "#SBATCH --partition=computes_thin",
            "#SBATCH --ntasks=1",
            "#SBATCH --mem-per-cpu=1G",
        ]

    if md_profile == "line_smoke":
        return [
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --time=00:30:00",
        ]
    if md_profile == "production_smoke":
        return [
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --time=04:00:00",
        ]
    return [
        "#SBATCH --nodes=6",
        "#SBATCH --ntasks-per-node=8",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --time=320:00:00",
    ]


def _script_header(
    job_name: str,
    stdout_name: str,
    stderr_name: str,
    *,
    compute: bool,
    md_profile: str,
) -> str:
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH -o {stdout_name}",
        f"#SBATCH -e {stderr_name}",
    ]
    if BURA_DEFAULT_EXCLUDE_NODES:
        lines.append(f"#SBATCH --exclude={BURA_DEFAULT_EXCLUDE_NODES}")
    lines.extend(_resource_profile(md_profile, compute=compute))
    lines.append('SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"')
    lines.append('source "$SCRIPT_DIR/common.sh"')
    lines.append("set -euo pipefail")
    lines.append('cd "$SCRIPT_DIR"')
    return "\n".join(lines) + "\n\n"


def _mdrun_command(md_profile: str, args: str) -> str:
    if md_profile == "full":
        return f'mpirun -np "${{SLURM_NTASKS:-1}}" "$GMX_CMD" mdrun {args}\n'
    return f'"$GMX_CMD" mdrun {args}\n'


def _step_scripts(sequence: str, md_profile: str) -> dict[str, str]:
    peptide = sequence
    compute_prefix = 'export GMX_MAXCONSTRWARN=-1\nexport OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"\n\n'
    return {
        "0_CG_pol_sysprep.sh": _script_header(f"{peptide}_prep", "prep.out", "prep.err", compute=False, md_profile=md_profile)
        + "peptide_name=$(peptide_name_from_pdb)\n"
        + "nmol=$(nmol_from_sequence \"$peptide_name\")\n"
        + '"$PY2_CMD" martinize.py -f "${peptide_name}.pdb" -x "${peptide_name}_CG.pdb" -o topol.top -p backbone -ff martini22p -ss $(printf "E%.0s" $(seq 1 ${#peptide_name}))\n'
        + "sed -i -e 's/martini\\.itp/martini_v2.2refP.itp/' topol.top\n"
        + '"$GMX_CMD" insert-molecules -ci "${peptide_name}_CG.pdb" -nmol "$nmol" -box 20 20 20 -o box.gro\n'
        + 'cp box.gro "${peptide_name}_inserted_initial_noncontact.gro"\n'
        + 'sed -i -e "s/1/${nmol}/" topol.top\n'
        + '"$GMX_CMD" grompp -f martini_22P_vacuum_mini.mdp -c box.gro -p topol.top -o box.tpr -r box.gro -maxwarn 2\n'
        + '"$GMX_CMD" mdrun -v -s box.tpr -deffnm box\n'
        + '"$GMX_CMD" solvate -cp box.gro -cs water.gro -radius 0.21 -p topol.top -o solvbox.gro\n'
        + '"$PY2_CMD" triple-w.py solvbox.gro\n'
        + 'sed -i -e "s/${nmol}W/${nmol}\\nPW/" topol.top\n'
        + 'echo -e "name 13 PW\\nq" | "$GMX_CMD" make_ndx -f solvbox_PW.gro -o index.ndx\n',
        "1_Softcore_mini_a.sh": _script_header(f"{peptide}_soft_grompp", "1mini_a.out", "1mini_a.err", compute=False, md_profile=md_profile)
        + '"$GMX_CMD" grompp -f martini_22P_mini_soft.mdp -c solvbox_PW.gro -p topol.top -o em0.tpr -r solvbox_PW.gro -n index.ndx -maxwarn 2\n',
        "2_Softcore_mini_b.sh": _script_header(f"{peptide}_soft_run", "2mini_b.out", "2mini_b.err", compute=True, md_profile=md_profile)
        + compute_prefix
        + _mdrun_command(md_profile, "-v -deffnm em0 -s em0.tpr"),
        "3_Steepest_mini_a.sh": _script_header(f"{peptide}_steep_grompp", "3mini_a.out", "3mini_a.err", compute=False, md_profile=md_profile)
        + '"$GMX_CMD" grompp -f martini_22P_mini_steep.mdp -c em0.gro -p topol.top -o em1.tpr -r em0.gro -n index.ndx -maxwarn 1\n',
        "4_Steepest_mini_b.sh": _script_header(f"{peptide}_steep_run", "4mini_b.out", "4mini_b.err", compute=True, md_profile=md_profile)
        + compute_prefix
        + _mdrun_command(md_profile, "-v -deffnm em1"),
        "5_Berendsen_equi_a.sh": _script_header(f"{peptide}_ber_grompp", "5equi_a.out", "5equi_a.err", compute=False, md_profile=md_profile)
        + "find . -type f -regex '.*/step.*\\.pdb' -delete\n"
        + "find . -type f -name '*#*#*' -exec rm -f {} \\;\n"
        + '"$GMX_CMD" grompp -f martini_22P_equi_Berendsen.mdp -c em1.gro -r em1.gro -p topol.top -o equi1.tpr -n index.ndx -maxwarn 1\n',
        "6_Berendsen_equi_b.sh": _script_header(f"{peptide}_ber_run", "6equi_b.out", "6equi_b.err", compute=True, md_profile=md_profile)
        + compute_prefix
        + _mdrun_command(md_profile, "-v -deffnm equi1"),
        "7_Parr_Rah_equi_a.sh": _script_header(f"{peptide}_parr_grompp", "7equi_a.out", "7equi_a.err", compute=False, md_profile=md_profile)
        + '"$GMX_CMD" grompp -f martini_22P_equi_ParrRah.mdp -c equi1.gro -r equi1.gro -p topol.top -o equi2.tpr -n index.ndx -maxwarn 1\n',
        "8_Parr_Rah_equi_b.sh": _script_header(f"{peptide}_parr_run", "8equi_b.out", "8equi_b.err", compute=True, md_profile=md_profile)
        + compute_prefix
        + _mdrun_command(md_profile, "-v -deffnm equi2"),
        "9_Dynamics_a.sh": _script_header(f"{peptide}_dyn_grompp", "9md_a.out", "9md_a.err", compute=False, md_profile=md_profile)
        + "peptide_name=$(peptide_name_from_pdb)\n"
        + "nmol=$(nmol_from_sequence \"$peptide_name\")\n"
        + '"$GMX_CMD" grompp -f martini_22P_md.mdp -c equi2.gro -t equi2.cpt -p topol.top -o "${peptide_name}_${nmol}_CG.tpr"\n',
        "10_Dynamics_b.sh": _script_header(f"{peptide}_dyn_run", "10md_b.out", "10md_b.err", compute=True, md_profile=md_profile)
        + compute_prefix
        + "peptide_name=$(peptide_name_from_pdb)\n"
        + "nmol=$(nmol_from_sequence \"$peptide_name\")\n"
        + _mdrun_command(md_profile, '-v -s "${peptide_name}_${nmol}_CG.tpr" -deffnm "${peptide_name}_${nmol}_CG"'),
    }

def _analysis_scripts(sequence: str, md_profile: str) -> dict[str, str]:
    peptide = sequence
    script_11 = _script_header(f"{peptide}_sasa", "11_ana.out", "11_ana.err", compute=False, md_profile=md_profile)
    script_11 += "peptide_name=$(peptide_name_from_pdb)\n"
    script_11 += "nmol=$(nmol_from_sequence \"$peptide_name\")\n"
    script_11 += f'PEPTIDE_GROUP="{PAPER_AP_SASA_GROUP_NAME}"\n'
    script_11 += r"""write_peptide_non_solvent_index() {
  local gro="$1"
  local ndx="$2"
  local atom_count
  atom_count=$(sed -n '2p' "$gro" | tr -d ' ')
  if [[ -z "$atom_count" ]]; then
    echo "Cannot read atom count from $gro" >&2
    return 1
  fi
  {
    echo "[ ${PEPTIDE_GROUP} ]"
    awk -v n="$atom_count" '
      NR > 2 && NR <= n + 2 {
        res = substr($0, 6, 5)
        gsub(/ /, "", res)
        atom = substr($0, 16, 5) + 0
        if (res != "PW" && res != "W" && res != "SOL" && res != "NA" && res != "CL" && res != "ION") {
          printf "%d ", atom
          c++
          if (c % 15 == 0) printf "\n"
        }
      }
      END { printf "\n" }
    ' "$gro"
    echo "[ System ]"
    awk -v n="$atom_count" 'BEGIN { for (i=1; i<=n; i++) { printf "%d ", i; if (i % 15 == 0) printf "\n" } printf "\n" }'
  } > "$ndx"
}
"""
    script_11 += "contact_frames=(0 5 12 25 50 100 200)\n"
    script_11 += "for frame_ns in \"${contact_frames[@]}\"; do\n"
    script_11 += '  dump_ps=$((frame_ns * 1000))\n'
    script_11 += '  echo -e "0" | "$GMX_CMD" trjconv -f "${peptide_name}_${nmol}_CG.xtc" -s "${peptide_name}_${nmol}_CG.tpr" -o "${peptide_name}_${frame_ns}ns.gro" -dump "${dump_ps}"\n'
    script_11 += '  echo -e "r1\\nq" | "$GMX_CMD" make_ndx -f "${peptide_name}_${nmol}_CG.tpr" -o index_image.ndx\n'
    script_11 += '  echo -e "0" | "$GMX_CMD" trjconv -s "${peptide_name}_${nmol}_CG.tpr" -f "${peptide_name}_${frame_ns}ns.gro" -n index_image.ndx -o "${peptide_name}_${frame_ns}ns.gro" -pbc whole\n'
    script_11 += '  echo -e "0" | "$GMX_CMD" trjconv -f "${peptide_name}_${frame_ns}ns.gro" -s "${peptide_name}_${nmol}_CG.tpr" -pbc nojump -o "${peptide_name}_${frame_ns}ns.gro"\n'
    script_11 += '  echo -e "1\\n0" | "$GMX_CMD" trjconv -s "${peptide_name}_${nmol}_CG.tpr" -f "${peptide_name}_${frame_ns}ns.gro" -n index_image.ndx -o "${peptide_name}_${frame_ns}ns.gro" -pbc mol -center\n'
    script_11 += '  echo -e "0" | "$GMX_CMD" trjconv -f "${peptide_name}_${nmol}_CG.xtc" -s "${peptide_name}_${nmol}_CG.tpr" -o "${peptide_name}_${frame_ns}ns.pdb" -dump "${dump_ps}"\n'
    script_11 += '  echo -e "0" | "$GMX_CMD" trjconv -s "${peptide_name}_${nmol}_CG.tpr" -f "${peptide_name}_${frame_ns}ns.pdb" -n index_image.ndx -o "${peptide_name}_${frame_ns}ns.pdb" -pbc whole\n'
    script_11 += "done\n"
    script_11 += 'write_peptide_non_solvent_index "${peptide_name}_${nmol}_CG.gro" paper_ap_sasa_trajectory.ndx\n'
    script_11 += 'printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_CMD" sasa -f "${peptide_name}_${nmol}_CG.xtc" -s "${peptide_name}_${nmol}_CG.tpr" -n paper_ap_sasa_trajectory.ndx -tu ns -o "${peptide_name}_sasa.xvg"\n'
    script_11 += 'if [[ -f "${peptide_name}_inserted_initial_noncontact.gro" && -f box.tpr ]]; then\n'
    script_11 += '  write_peptide_non_solvent_index "${peptide_name}_inserted_initial_noncontact.gro" paper_ap_sasa_initial.ndx\n'
    script_11 += '  printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_CMD" sasa -f "${peptide_name}_inserted_initial_noncontact.gro" -s box.tpr -n paper_ap_sasa_initial.ndx -o "${peptide_name}_paper_initial_sasa.xvg"\n'
    script_11 += '  printf "inserted_initial_noncontact\\n" > "${peptide_name}_paper_initial_sasa_source.txt"\n'
    script_11 += 'fi\n'

    script_12 = _script_header(f"{peptide}_ap", "AP.out", "AP.err", compute=False, md_profile=md_profile)
    script_12 += "file=$(find . -maxdepth 1 -name '*_sasa.xvg' | head -n 1)\n"
    script_12 += 'if [[ -z "$file" ]]; then echo "Missing *_sasa.xvg" >&2; exit 1; fi\n'
    script_12 += 'peptidename=$(basename "$file" .xvg)\n'
    script_12 += 'outfile="${peptidename}_AP_SASA.txt"\n'
    script_12 += "declare -A values\n"
    script_12 += "while IFS= read -r line; do\n"
    script_12 += "  if [[ $line != \\#* ]] && [[ $line != \\@* ]]; then\n"
    script_12 += "    time=$(echo $line | awk '{print $1}')\n"
    script_12 += "    value=$(echo $line | awk '{print $2}')\n"
    script_12 += "    values[$time]=$value\n"
    script_12 += "  fi\n"
    script_12 += "done < \"$file\"\n"
    script_12 += ": > \"$outfile\"\n"
    script_12 += "missing_targets=()\n"
    script_12 += "for target in 5 12 25 50 100 200; do\n"
    script_12 += "  key=$(printf '%.3f' \"$target\")\n"
    script_12 += "  if [[ -n ${values[0.000]:-} ]] && [[ -n ${values[$key]:-} ]]; then\n"
    script_12 += "    ratio=$(echo \"${values[0.000]} / ${values[$key]}\" | bc -l)\n"
    script_12 += "    echo \"The AP for ${target} ns is: ${ratio}\" >> \"$outfile\"\n"
    script_12 += "  else\n"
    script_12 += "    missing_targets+=(\"${target}\")\n"
    script_12 += "  fi\n"
    script_12 += "done\n"
    script_12 += "if (( ${#missing_targets[@]} > 0 )); then\n"
    script_12 += "  rm -f \"$outfile\"\n"
    script_12 += "  echo \"Missing SASA time points for AP calculation: ${missing_targets[*]} ns\" >&2\n"
    script_12 += "  exit 1\n"
    script_12 += "fi\n"

    frame_cases = "\n".join(
        f"  {ns}) dump_ps={ns * 1000} ;;" for ns in _paper_path_last10_frame_ns()
    )
    script_13 = _script_header(
        f"{peptide}_last10_apcontact",
        "13_last10_apcontact.out",
        "13_last10_apcontact.err",
        compute=False,
        md_profile=md_profile,
    )
    script_13 += "peptide_name=$(peptide_name_from_pdb)\n"
    script_13 += "nmol=$(nmol_from_sequence \"$peptide_name\")\n"
    script_13 += 'XTC="${peptide_name}_${nmol}_CG.xtc"\n'
    script_13 += 'TPR="${peptide_name}_${nmol}_CG.tpr"\n'
    script_13 += 'OUT_DIR="${peptide_name}_paper_path_last10ns_frames"\n'
    script_13 += 'STATUS_JSON="${peptide_name}_paper_path_APcontact_last10ns_status.json"\n'
    script_13 += '[[ -f "$XTC" ]] || { echo "Missing trajectory: $PWD/$XTC" >&2; exit 1; }\n'
    script_13 += '[[ -f "$TPR" ]] || { echo "Missing run input: $PWD/$TPR" >&2; exit 1; }\n'
    script_13 += 'if [[ ! -f index_image.ndx ]]; then\n'
    script_13 += '  printf "r1\\nq\\n" | "$GMX_CMD" make_ndx -f "$TPR" -o index_image.ndx\n'
    script_13 += 'fi\n'
    script_13 += 'mkdir -p "$OUT_DIR"\n'
    script_13 += f'for frame_ns in {" ".join(str(ns) for ns in _paper_path_last10_frame_ns())}; do\n'
    script_13 += '  case "$frame_ns" in\n'
    script_13 += frame_cases + "\n"
    script_13 += '    *) echo "Unsupported frame: $frame_ns" >&2; exit 1 ;;\n'
    script_13 += '  esac\n'
    script_13 += '  raw="$OUT_DIR/${peptide_name}_${frame_ns}ns.raw.gro"\n'
    script_13 += '  out="$OUT_DIR/${peptide_name}_${frame_ns}ns.gro"\n'
    script_13 += '  echo "[paper_path_APcontact] extracting $frame_ns ns from $XTC"\n'
    script_13 += '  printf "0\\n" | "$GMX_CMD" trjconv -s "$TPR" -f "$XTC" -o "$raw" -dump "$dump_ps"\n'
    script_13 += '  printf "0\\n" | "$GMX_CMD" trjconv -s "$TPR" -f "$raw" -n index_image.ndx -o "$out.tmp.gro" -pbc whole\n'
    script_13 += '  printf "0\\n" | "$GMX_CMD" trjconv -s "$TPR" -f "$out.tmp.gro" -o "$out.tmp2.gro" -pbc nojump\n'
    script_13 += '  printf "1\\n0\\n" | "$GMX_CMD" trjconv -s "$TPR" -f "$out.tmp2.gro" -n index_image.ndx -o "$out" -pbc mol -center\n'
    script_13 += '  rm -f "$out.tmp.gro" "$out.tmp2.gro"\n'
    script_13 += 'done\n'
    script_13 += 'cat > "$STATUS_JSON" <<EOF\n'
    script_13 += '{\n'
    script_13 += '  "status": "frames_extracted",\n'
    script_13 += '  "sequence": "${peptide_name}",\n'
    script_13 += '  "purpose": "Extract final-10-ns GRO frames for paper-style APcontact averaging.",\n'
    script_13 += '  "frames_dir": "${OUT_DIR}",\n'
    script_13 += '  "message": "Final-10-ns frames were extracted automatically by the BURA full-analysis chain. Run parse-md-results after copy-back to compute paper_path_APcontact_last10ns."\n'
    script_13 += '}\n'
    script_13 += 'EOF\n'
    script_13 += 'echo "[paper_path_APcontact] extracted final-10-ns frames into $OUT_DIR"\n'
    return {
        "11_SASA_and_FrameDump.sh": script_11,
        "12_AP_calc.sh": script_12,
        "13_Extract_last10ns_for_paper_APcontact.sh": script_13,
    }


def _write_package(package_dir: Path, sequence: str, md_profile: str) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _copy_non_mdp_assets(package_dir)
    _copy_profiled_mdp_files(package_dir, md_profile)
    _write_text(package_dir / "env_bura.sh", _env_script_text())
    _chmod_shell(package_dir / "env_bura.sh")
    _write_text(package_dir / "common.sh", _common_script_text())
    _chmod_shell(package_dir / "common.sh")
    scripts = _step_scripts(sequence, md_profile)
    if _profile_has_analysis(md_profile):
        scripts.update(_analysis_scripts(sequence, md_profile))
    for name in _chain_for_profile(md_profile):
        script_path = package_dir / name
        _write_text(script_path, scripts[name])
        _chmod_shell(script_path)

def _submit_helper_text(campaign: str, rows: list[dict[str, str]], md_profile: str) -> str:
    quoted_packages = " ".join(f'"{Path(row["package_dir"]).name}"' for row in rows)
    quoted_sequences = " ".join(f'"{row["sequence"]}"' for row in rows)
    steps = " ".join(_chain_for_profile(md_profile))
    return f"""#!/bin/bash
set -eo pipefail
CAMPAIGN_DIR=$(cd "$(dirname "$0")" && pwd)
PACKAGES_DIR="$CAMPAIGN_DIR/packages"
ALL_PACKAGES=({quoted_packages})
ALL_SEQUENCES=({quoted_sequences})
STEPS=({steps})

usage() {{
  echo "Usage: $0 [--exclude NODELIST] --all | [--exclude NODELIST] PEPTIDE_OR_PACKAGE [PEPTIDE_OR_PACKAGE ...]" >&2
  exit 1
}}

DEFAULT_EXCLUDE_NODES="{BURA_DEFAULT_EXCLUDE_NODES}"
EXCLUDE_ARGS=()
if [[ -n "$DEFAULT_EXCLUDE_NODES" ]]; then
  EXCLUDE_ARGS=(--exclude "$DEFAULT_EXCLUDE_NODES")
fi
SUBMIT_ALL=0
TARGETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exclude)
      shift
      if [[ $# -eq 0 ]]; then
        usage
      fi
      if [[ -n "$DEFAULT_EXCLUDE_NODES" ]]; then
        EXCLUDE_ARGS=(--exclude "$DEFAULT_EXCLUDE_NODES,$1")
      else
        EXCLUDE_ARGS=(--exclude "$1")
      fi
      shift
      ;;
    --all)
      SUBMIT_ALL=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        TARGETS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

submit_index() {{
  local index="$1"
  local package_name="${{ALL_PACKAGES[$index]}}"
  local peptide="${{ALL_SEQUENCES[$index]}}"
  local package_dir="$PACKAGES_DIR/$package_name"
  if [[ ! -d "$package_dir" ]]; then
    echo "Missing package for $package_name: $package_dir" >&2
    exit 1
  fi
  if [[ ! -f "$package_dir/$peptide.pdb" ]]; then
    echo "Missing PDB for $peptide in $package_name. Build or copy $peptide.pdb before submitting." >&2
    exit 1
  fi
  local previous_job=""
  local job_id=""
  echo "Submitting canonical chain for $peptide ($package_name)"
  for step in "${{STEPS[@]}}"; do
    if [[ -z "$previous_job" ]]; then
      job_id=$(cd "$package_dir" && sbatch "${{EXCLUDE_ARGS[@]}}" --parsable "$step")
    else
      job_id=$(cd "$package_dir" && sbatch "${{EXCLUDE_ARGS[@]}}" --parsable --dependency=afterok:$previous_job "$step")
    fi
    previous_job="$job_id"
    echo "  $step -> $job_id"
  done
}}

submit_target() {{
  local target="$1"
  local matched=0
  local index
  for index in "${{!ALL_PACKAGES[@]}}"; do
    if [[ "$target" == "${{ALL_PACKAGES[$index]}}" || "$target" == "${{ALL_SEQUENCES[$index]}}" ]]; then
      submit_index "$index"
      matched=1
    fi
  done
  if [[ "$matched" -ne 1 ]]; then
    echo "No package or peptide matched target: $target" >&2
    exit 1
  fi
}}

if [[ "$SUBMIT_ALL" -eq 1 ]]; then
  if [[ ${{#TARGETS[@]}} -ne 0 ]]; then
    usage
  fi
  for index in "${{!ALL_PACKAGES[@]}}"; do
    submit_index "$index"
  done
  exit 0
fi

if [[ ${{#TARGETS[@]}} -eq 0 ]]; then
  usage
fi

for target in "${{TARGETS[@]}}"; do
  submit_target "$target"
done
"""


def _preflight_text(rows: list[dict[str, str]], md_profile: str) -> str:
    expected_packages = " ".join(f'"{Path(row["package_dir"]).name}"' for row in rows)
    expected_sequences = " ".join(f'"{row["sequence"]}"' for row in rows)
    steps = " ".join(_chain_for_profile(md_profile))
    return f"""#!/bin/bash
set -euo pipefail
CAMPAIGN_DIR=$(cd "$(dirname "$0")" && pwd)
PACKAGES_DIR="$CAMPAIGN_DIR/packages"
EXPECTED_PACKAGES=({expected_packages})
EXPECTED_SEQUENCES=({expected_sequences})

echo "BURA-safe preflight only."
echo "Before any submission on BURA, run exactly:"
echo '  find . -type f -name "*.sh" -exec dos2unix {{}} +'
echo '  module load gromacs/2023.2_g13.1_p3.10.5'
echo "Do not run gmx, martinize.py, triple-w.py, or long Python jobs on login nodes."

command -v sbatch >/dev/null 2>&1 || {{ echo "sbatch not available" >&2; exit 1; }}
command -v dos2unix >/dev/null 2>&1 || {{ echo "dos2unix not available" >&2; exit 1; }}
type module >/dev/null 2>&1 || {{ echo "Environment modules are not available in this shell" >&2; exit 1; }}
module load gromacs/2023.2_g13.1_p3.10.5 >/dev/null 2>&1 || {{ echo "Required module unavailable: gromacs/2023.2_g13.1_p3.10.5" >&2; exit 1; }}

if ! command -v python2 >/dev/null 2>&1; then
  py2_found=0
  for candidate in python/Python-2.7.18 Python/2.7.18 python/2.7.18 Python/2.7 python/2.7; do
    if module load "$candidate" >/dev/null 2>&1 && command -v python2 >/dev/null 2>&1; then
      py2_found=1
      break
    fi
  done
  if [[ "$py2_found" -ne 1 ]]; then
    echo "Python 2 runtime unavailable. Check available modules with 'module avail python' or 'module spider Python'." >&2
    exit 1
  fi
fi

for index in "${{!EXPECTED_PACKAGES[@]}}"; do
  package_name="${{EXPECTED_PACKAGES[$index]}}"
  peptide="${{EXPECTED_SEQUENCES[$index]}}"
  package_dir="$PACKAGES_DIR/$package_name"
  [[ -d "$package_dir" ]] || {{ echo "Missing package: $package_dir" >&2; exit 1; }}
  [[ -f "$CAMPAIGN_DIR/PDBs/$peptide.pdb" ]] || echo "PDB not present yet for $peptide"
  [[ -f "$package_dir/$peptide.pdb" ]] || echo "Package PDB not present yet for $peptide in $package_name"
  for step in {steps}; do
    [[ -f "$package_dir/$step" ]] || {{ echo "Missing step script $step for $peptide" >&2; exit 1; }}
  done
  for mdp in {' '.join(MDP_FILENAMES)}; do
    [[ -f "$package_dir/$mdp" ]] || {{ echo "Missing MDP file $mdp for $peptide" >&2; exit 1; }}
  done
done

echo "Preflight checks completed. Next: run the dos2unix command, then submit one peptide with ./submit_chain.sh PEPTIDE"
"""


def _readme_text(campaign: str, cluster: str, md_profile: str) -> str:
    return f"""# MD Campaign: {campaign}

This campaign is prepared for **{cluster.upper()}** with MD profile **{md_profile}** and follows the safe v1 policy:

- Build PDBs locally only.
- Upload this campaign directory to BURA.
- On BURA login/access nodes, only stage files, run the exact line-ending fix command, inspect modules, run `./preflight_bura.sh`, and submit with `./submit_chain.sh`.
- Do **not** run `gmx`, `martinize.py`, `triple-w.py`, or full simulation steps on login/access nodes.
- Heavy MD work must run only through Slurm jobs.
- `cgmd_label` must be assigned by a human after reviewing `md_review.csv`.

Exact BURA login-node preparation commands:

```bash
find . -type f -name "*.sh" -exec dos2unix {{}} \\+
module load gromacs/2023.2_g13.1_p3.10.5
```

Recommended execution order across profiles:

1. `line_smoke`
2. `production_smoke`
3. `full`

Canonical automated chain in this campaign:

0. `0_CG_pol_sysprep.sh`
1. `1_Softcore_mini_a.sh`
2. `2_Softcore_mini_b.sh`
3. `3_Steepest_mini_a.sh`
4. `4_Steepest_mini_b.sh`
5. `5_Berendsen_equi_a.sh`
6. `6_Berendsen_equi_b.sh`
7. `7_Parr_Rah_equi_a.sh`
8. `8_Parr_Rah_equi_b.sh`
9. `9_Dynamics_a.sh`
10. `10_Dynamics_b.sh`
11. `11_SASA_and_FrameDump.sh`
12. `12_AP_calc.sh`
13. `13_Extract_last10ns_for_paper_APcontact.sh`

Excluded from automation:

- `11_Analysis_SASAnFrameDump.sh` because it references a missing script.
- The original `13_Array_APcontact.sh` is not used because it is missing from
  the inherited BURA template. Instead, this campaign adds
  `13_Extract_last10ns_for_paper_APcontact.sh` to extract final-10-ns frame
  dumps for the paper-style APcontact calculation. During local finalization,
  the dashboard computes documented AP_contact metrics from returned `.gro`
  frame dumps when available.

Profile behavior:

- `line_smoke`: all minimization, equilibration, and production `nsteps` are rewritten to `200`, and the chain stops after `10_Dynamics_b.sh`.
- `production_smoke`: minimization and equilibration keep the template values; only production `nsteps` is rewritten to `200`, and the chain stops after `10_Dynamics_b.sh`.
- `full`: all `.mdp` files keep the template `nsteps` values, including the bundled 200 ns production run required for AP targets through 200 ns, and the chain continues through post-analysis plus final-10-ns frame extraction.

Submission chain by profile:

- `line_smoke`: submits only steps `0_CG_pol_sysprep.sh` through `10_Dynamics_b.sh`.
- `production_smoke`: submits only steps `0_CG_pol_sysprep.sh` through `10_Dynamics_b.sh`.
- `full`: submits the full chain through `13_Extract_last10ns_for_paper_APcontact.sh`.
"""


def prepare_md_campaign(
    run_dir: Path,
    batch_csv: Path,
    campaign: str,
    cluster: str = "bura",
    md_profile: str = "full",
) -> Path:
    cluster_name = _validate_cluster(cluster)
    profile_name = _validate_md_profile(md_profile)
    config = _load_config(run_dir)
    rows = _validate_batch_rows(_read_csv(batch_csv), config)
    paths = _campaign_paths(run_dir, campaign)
    if paths["campaign_dir"].exists():
        raise FileExistsError(f"MD campaign already exists: {paths['campaign_dir']}")
    _template_source(profile_name)

    paths["pdbs"].mkdir(parents=True, exist_ok=True)
    paths["packages"].mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        sequence = row["sequence"]
        occurrence = int(row.get("_sequence_occurrence", "1"))
        package_name = _package_name_for_sequence(sequence, occurrence)
        package_dir = _package_dir(paths, package_name)
        _write_package(package_dir, sequence, profile_name)
        manifest_rows.append(
            {
                "sequence": sequence,
                "round_id": row["round_id"],
                "acquisition_strategy": row["acquisition_strategy"],
                "pred_mean": row["pred_mean"],
                "pred_std": row["pred_std"],
                "pred_entropy": row["pred_entropy"],
                "pred_mutual_information": row["pred_mutual_information"],
                "acquisition_score": row["acquisition_score"],
                "campaign": campaign,
                "cluster": cluster_name,
                "md_profile": profile_name,
                "package_dir": str(package_dir.relative_to(paths["campaign_dir"])),
                "pdb_path": str((paths["pdbs"] / f"{sequence}.pdb").relative_to(paths["campaign_dir"])),
                "pdb_status": "pending_local_build",
            }
        )

    _write_text(paths["sequences"], _sequence_file_text(manifest_rows))
    _save_csv(paths["manifest"], MANIFEST_FIELDS, manifest_rows)
    _save_csv(paths["review"], REVIEW_FIELDS, [_empty_review_row(row) for row in manifest_rows])
    _write_text(paths["submit"], _submit_helper_text(campaign, manifest_rows, profile_name))
    _chmod_shell(paths["submit"])
    _write_text(paths["preflight"], _preflight_text(manifest_rows, profile_name))
    _chmod_shell(paths["preflight"])
    _write_text(paths["readme"], _readme_text(campaign, cluster_name, profile_name))
    return paths["campaign_dir"]

def _copy_pdb_into_package(campaign_dir: Path, sequence: str, package_dir: Path) -> None:
    pdb_path = campaign_dir / "PDBs" / f"{sequence}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB for {sequence}: {pdb_path}")
    shutil.copy2(pdb_path, package_dir / f"{sequence}.pdb")


def build_pdbs(campaign_dir: Path, validate_only: bool = False) -> Path:
    manifest_path = campaign_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in campaign: {campaign_dir}")
    manifest_rows = _read_csv(manifest_path)
    sequences = [row["sequence"] for row in manifest_rows]
    unique_sequences = list(dict.fromkeys(sequences))

    built_status = "manual_ready" if validate_only else "built_local"
    if not validate_only:
        built = False
        try:
            import pymol2  # type: ignore

            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                for sequence in unique_sequences:
                    cmd.reinitialize()
                    cmd.fab(sequence, sequence)
                    cmd.alter(sequence, 'chain="A"')
                    cmd.save(str(campaign_dir / "PDBs" / f"{sequence}.pdb"), sequence)
            built = True
        except ImportError:
            try:
                from pymol import cmd, finish_launching  # type: ignore

                finish_launching(["pymol", "-cq"])
                for sequence in unique_sequences:
                    cmd.delete("all")
                    cmd.fab(sequence, sequence)
                    cmd.alter(sequence, 'chain="A"')
                    cmd.save(str(campaign_dir / "PDBs" / f"{sequence}.pdb"), sequence)
                built = True
            except ImportError as exc:
                raise RuntimeError(
                    "PyMOL is not available. Place PDBs manually into the campaign PDBs directory and rerun with --validate-only."
                ) from exc
        if not built:
            raise RuntimeError("Failed to build PDBs.")

    for row in manifest_rows:
        _copy_pdb_into_package(campaign_dir, row["sequence"], campaign_dir / row["package_dir"])

    for row in manifest_rows:
        row["pdb_status"] = built_status
    _save_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    return campaign_dir / "PDBs"


def _parse_xvg_values(path: Path) -> dict[float, float]:
    values: dict[float, float] = {}
    for time, value in _parse_xvg_series(path):
        values[time] = value
    return values


def _parse_xvg_series(path: Path) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                values.append((round(float(parts[0]), 3), float(parts[1])))
            except ValueError:
                continue
    return values


def _first_xvg_value(path: Path) -> float | None:
    values = _parse_xvg_series(path)
    return values[0][1] if values else None


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _paper_ap_sasa_empty(status: str = "") -> dict[str, str]:
    return {
        "paper_ap_sasa_last10ns_mean": "",
        "paper_ap_sasa_last10ns_sd": "",
        "paper_ap_sasa_last10ns_n_frames": "",
        "paper_ap_sasa_initial_sasa": "",
        "paper_ap_sasa_initial_source": "",
        "paper_ap_sasa_final10_mean_sasa": "",
        "paper_ap_sasa_status": status,
        "paper_ap_sasa_method": "",
        "paper_ap_sasa_group_selection": "",
    }


def _paper_ap_sasa_initial_sasa(package_dir: Path, sequence: str, trajectory_sasa: list[tuple[float, float]]) -> tuple[float | None, str]:
    initial_sasa_file = package_dir / f"{sequence}_paper_initial_sasa.xvg"
    if initial_sasa_file.exists():
        initial = _first_xvg_value(initial_sasa_file)
        if initial is not None:
            source_file = package_dir / f"{sequence}_paper_initial_sasa_source.txt"
            if source_file.exists():
                source = source_file.read_text(encoding="utf-8", errors="replace").strip()
                if source:
                    return initial, source
            if (package_dir / f"{sequence}_inserted_initial_noncontact.gro").exists():
                return initial, "inserted_initial_noncontact"
            if (package_dir / "box.gro").exists():
                return initial, "pre_solvent_vacuum_minimized"
            return initial, "initial_sasa_xvg"
    if trajectory_sasa:
        return trajectory_sasa[0][1], "approximate_production_0ns"
    return None, ""


def _write_paper_ap_sasa_recompute_script(package_dir: Path, sequence: str) -> tuple[Path, Path]:
    script_path = package_dir / f"{sequence}_recompute_paper_APSASA.sh"
    status_path = package_dir / f"{sequence}_paper_APSASA_last10ns_status.json"
    peptide_group = PAPER_AP_SASA_GROUP_NAME
    script = f"""#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -f common.sh ]]; then
  source "$SCRIPT_DIR/common.sh"
elif [[ -f env_bura.sh ]]; then
  source "$SCRIPT_DIR/env_bura.sh"
else
  module load gromacs/2023.2_g13.1_p3.10.5 >/dev/null 2>&1 || true
  export GMX_CMD=${{GMX_CMD:-gmx_mpi}}
fi
GMX_RUN="${{GMX_CMD:-gmx_mpi}}"
if ! command -v "$GMX_RUN" >/dev/null 2>&1; then
  if command -v gmx >/dev/null 2>&1; then
    GMX_RUN=gmx
  else
    echo "GROMACS command not found. Run this on BURA or another environment with GROMACS installed." >&2
    exit 1
  fi
fi
SEQUENCE="{sequence}"
PEPTIDE_GROUP="{peptide_group}"
XTC=$(find . -maxdepth 1 -name "${{SEQUENCE}}*_CG.xtc" | sort | tail -n 1)
TPR=$(find . -maxdepth 1 -name "${{SEQUENCE}}*_CG.tpr" | sort | tail -n 1)
GRO=$(find . -maxdepth 1 -name "${{SEQUENCE}}*_CG.gro" | sort | tail -n 1)
INITIAL_GRO="${{SEQUENCE}}_inserted_initial_noncontact.gro"
if [[ -z "$XTC" || -z "$TPR" || -z "$GRO" ]]; then
  echo "Missing trajectory inputs: need *_CG.xtc, *_CG.tpr, and *_CG.gro." >&2
  exit 1
fi
write_peptide_non_solvent_index() {{
  local gro="$1"
  local ndx="$2"
  local atom_count
  atom_count=$(sed -n '2p' "$gro" | tr -d ' ')
  {{
    echo "[ $PEPTIDE_GROUP ]"
    awk -v n="$atom_count" '
      NR > 2 && NR <= n + 2 {{
        res = substr($0, 6, 5)
        gsub(/ /, "", res)
        atom = substr($0, 16, 5) + 0
        if (res != "PW" && res != "W" && res != "SOL" && res != "NA" && res != "CL" && res != "ION") {{
          printf "%d ", atom
          c++
          if (c % 15 == 0) printf "\\n"
        }}
      }}
      END {{ printf "\\n" }}
    ' "$gro"
  }} > "$ndx"
}}
write_peptide_non_solvent_index "$GRO" paper_ap_sasa_trajectory.ndx
printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_RUN" sasa -f "$XTC" -s "$TPR" -n paper_ap_sasa_trajectory.ndx -tu ns -o "${{SEQUENCE}}_sasa.xvg"
if [[ -f "$INITIAL_GRO" && -f box.tpr ]]; then
  write_peptide_non_solvent_index "$INITIAL_GRO" paper_ap_sasa_initial.ndx
  printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_RUN" sasa -f "$INITIAL_GRO" -s box.tpr -n paper_ap_sasa_initial.ndx -o "${{SEQUENCE}}_paper_initial_sasa.xvg"
  printf "inserted_initial_noncontact\\n" > "${{SEQUENCE}}_paper_initial_sasa_source.txt"
elif [[ -f box.gro && -f box.tpr ]]; then
  write_peptide_non_solvent_index box.gro paper_ap_sasa_initial.ndx
  printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_RUN" sasa -f box.gro -s box.tpr -n paper_ap_sasa_initial.ndx -o "${{SEQUENCE}}_paper_initial_sasa.xvg"
  printf "pre_solvent_vacuum_minimized\\n" > "${{SEQUENCE}}_paper_initial_sasa_source.txt"
else
  echo "Missing $INITIAL_GRO/box.gro or box.tpr; parser will fall back to approximate production 0 ns baseline." >&2
fi
echo "Recomputed SASA. Copy outputs back if needed, then run: python -m active_learning_thesis parse-md-results --campaign-dir <campaign_dir>"
"""
    _write_text(script_path, script)
    _chmod_shell(script_path)
    status = {
        "status": "script_ready",
        "sequence": sequence,
        "script": str(script_path),
        "group_selection": peptide_group,
        "message": "Run this on BURA or another environment with GROMACS installed if paper-style AP_SASA needs recomputation.",
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return script_path, status_path


def _write_paper_ap_sasa_last10_file(package_dir: Path, sequence: str) -> tuple[Path | None, Path, Path]:
    script_path, status_path = _write_paper_ap_sasa_recompute_script(package_dir, sequence)
    output_path = package_dir / f"{sequence}_paper_APSASA_last10ns.txt"
    sasa_candidates = [path for path in sorted(package_dir.glob("*_sasa.xvg")) if "paper_initial" not in path.name]
    sasa_file = sasa_candidates[0] if sasa_candidates else None
    if not sasa_file:
        output_path.write_text(
            "# paper_APSASA_last10ns is not computed yet.\n"
            "# status=missing_sasa_xvg\n",
            encoding="utf-8",
        )
        return output_path, script_path, status_path

    trajectory_sasa = _parse_xvg_series(sasa_file)
    initial_sasa, initial_source = _paper_ap_sasa_initial_sasa(package_dir, sequence, trajectory_sasa)
    if initial_sasa is None or not trajectory_sasa:
        output_path.write_text(
            "# paper_APSASA_last10ns is not computed yet.\n"
            "# status=missing_initial_or_trajectory_sasa\n",
            encoding="utf-8",
        )
        return output_path, script_path, status_path

    max_time = max(time for time, _ in trajectory_sasa)
    window_start = max_time - PAPER_AP_SASA_LAST10_WINDOW_NS
    final_window = [(time, value) for time, value in trajectory_sasa if window_start <= time <= max_time]
    if not final_window:
        output_path.write_text(
            "# paper_APSASA_last10ns is not computed yet.\n"
            "# status=missing_final10_window\n",
            encoding="utf-8",
        )
        return output_path, script_path, status_path

    final_sasa_values = [value for _, value in final_window if value > 0]
    if not final_sasa_values:
        output_path.write_text(
            "# paper_APSASA_last10ns is not computed yet.\n"
            "# status=invalid_final_sasa\n",
            encoding="utf-8",
        )
        return output_path, script_path, status_path

    ap_values = [initial_sasa / value for value in final_sasa_values]
    mean_ap = statistics.fmean(ap_values)
    sd_ap = statistics.stdev(ap_values) if len(ap_values) > 1 else 0.0
    final10_mean_sasa = statistics.fmean(final_sasa_values)
    if initial_source == "inserted_initial_noncontact":
        status = "computed"
    elif initial_source == "pre_solvent_vacuum_minimized":
        status = "computed_preproduction_initial"
    else:
        status = "computed_approximate_initial"
    method = (
        "AP_SASA=SASA_initial_noncontact/mean_SASA_final10ns"
        if initial_source == "inserted_initial_noncontact"
        else (
            "AP_SASA=SASA_preproduction_presolvent/mean_SASA_final10ns"
            if initial_source == "pre_solvent_vacuum_minimized"
            else "AP_SASA=SASA_production_0ns/mean_SASA_final10ns"
        )
    )
    lines = [
        "# paper_APSASA_last10ns used by active_learning_thesis.",
        "# Keeps legacy AP_sasa unchanged; this is the Njirjak/Thapa-style evidence field.",
        f"# source_sasa_xvg={sasa_file.name}",
        f"# initial_source={initial_source}",
        f"# group_selection={PAPER_AP_SASA_GROUP_NAME}",
        f"The paper_APSASA last10ns mean is: {_format_metric(mean_ap)} "
        f"(sd={_format_metric(sd_ap)}; n_frames={len(final_sasa_values)}; method={method})",
        f"The paper_APSASA initial SASA is: {_format_metric(initial_sasa)}",
        f"The paper_APSASA final10 mean SASA is: {_format_metric(final10_mean_sasa)}",
        f"status={status}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status_doc = {
        "status": status,
        "sequence": sequence,
        "initial_sasa": initial_sasa,
        "initial_source": initial_source,
        "final10_mean_sasa": final10_mean_sasa,
        "last10_window_start_ns": window_start,
        "last10_window_end_ns": max_time,
        "n_frames": len(final_sasa_values),
        "mean": mean_ap,
        "sd": sd_ap,
        "method": method,
        "group_selection": PAPER_AP_SASA_GROUP_NAME,
        "recompute_script": str(script_path),
    }
    status_path.write_text(json.dumps(status_doc, indent=2), encoding="utf-8")
    return output_path, script_path, status_path


def _parse_paper_ap_sasa_last10_file(path: Path) -> dict[str, str]:
    parsed = _paper_ap_sasa_empty()
    if not path.exists():
        return parsed
    mean_pattern = re.compile(
        r"The paper_APSASA last10ns mean is:\s*([0-9eE+\-.]+)\s*"
        r"\(sd=([0-9eE+\-.]+);\s*n_frames=(\d+);\s*method=([^)]+)\)"
    )
    initial_pattern = re.compile(r"The paper_APSASA initial SASA is:\s*([0-9eE+\-.]+)")
    final_pattern = re.compile(r"The paper_APSASA final10 mean SASA is:\s*([0-9eE+\-.]+)")
    initial_source_pattern = re.compile(r"initial_source=([A-Za-z0-9_\-]+)")
    group_pattern = re.compile(r"group_selection=([A-Za-z0-9_\-]+)")
    status_pattern = re.compile(r"status=([A-Za-z0-9_\-]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if mean_match := mean_pattern.search(line):
                parsed["paper_ap_sasa_last10ns_mean"] = mean_match.group(1)
                parsed["paper_ap_sasa_last10ns_sd"] = mean_match.group(2)
                parsed["paper_ap_sasa_last10ns_n_frames"] = mean_match.group(3)
                parsed["paper_ap_sasa_method"] = mean_match.group(4)
            if initial_match := initial_pattern.search(line):
                parsed["paper_ap_sasa_initial_sasa"] = initial_match.group(1)
            if final_match := final_pattern.search(line):
                parsed["paper_ap_sasa_final10_mean_sasa"] = final_match.group(1)
            if initial_source_match := initial_source_pattern.search(line):
                parsed["paper_ap_sasa_initial_source"] = initial_source_match.group(1)
            if group_match := group_pattern.search(line):
                parsed["paper_ap_sasa_group_selection"] = group_match.group(1)
            if status_match := status_pattern.search(line):
                parsed["paper_ap_sasa_status"] = status_match.group(1)
    return parsed


def _parse_ap_file(path: Path) -> dict[str, str]:
    parsed = {
        "ap_5ns": "",
        "ap_12ns": "",
        "ap_25ns": "",
        "ap_50ns": "",
        "ap_100ns": "",
        "ap_200ns": "",
    }
    pattern = re.compile(r"The AP for (\d+) ns is:\s*([0-9eE+\-.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            parsed[f"ap_{match.group(1)}ns"] = match.group(2)
    return parsed


def _parse_ap_contact_file(path: Path) -> dict[str, str]:
    parsed = {f"ap_contact_{target}ns": "" for target in AP_TARGET_NS}
    pattern = re.compile(r"The AP_contact for (\d+) ns is:\s*([0-9eE+\-.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            key = f"ap_contact_{match.group(1)}ns"
            if key in parsed:
                parsed[key] = match.group(2)
    return parsed


def _parse_paper_ap_contact_file(path: Path) -> dict[str, str]:
    parsed = {f"paper_ap_contact_{target}ns": "" for target in AP_TARGET_NS}
    pattern = re.compile(r"The paper_APcontact for (\d+) ns is:\s*([0-9eE+\-.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            key = f"paper_ap_contact_{match.group(1)}ns"
            if key in parsed:
                parsed[key] = match.group(2)
    return parsed


def _parse_ap_contact_same_paper_formula_file(path: Path) -> dict[str, str]:
    parsed = {f"ap_contact_same_paper_formula_{target}ns": "" for target in AP_TARGET_NS}
    pattern = re.compile(r"The AP_contact_same_paper_formula for (\d+) ns is:\s*([0-9eE+\-.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            key = f"ap_contact_same_paper_formula_{match.group(1)}ns"
            if key in parsed:
                parsed[key] = match.group(2)
    return parsed


def _parse_paper_path_ap_contact_file(path: Path) -> dict[str, str]:
    parsed = {f"paper_path_ap_contact_{target}ns": "" for target in AP_TARGET_NS}
    pattern = re.compile(r"The paper_path_APcontact for (\d+) ns is:\s*([0-9eE+\-.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            key = f"paper_path_ap_contact_{match.group(1)}ns"
            if key in parsed:
                parsed[key] = match.group(2)
    return parsed


def _parse_paper_path_last10_ap_contact_file(path: Path) -> dict[str, str]:
    parsed = {
        "paper_path_ap_contact_last10ns_mean": "",
        "paper_path_ap_contact_last10ns_sd": "",
        "paper_path_ap_contact_last10ns_n_frames": "",
        "paper_path_ap_contact_last10ns_method": "",
        "paper_path_ap_contact_last10ns_status": "",
    }
    if not path.exists():
        return parsed
    mean_pattern = re.compile(
        r"The paper_path_APcontact last10ns mean is:\s*([0-9eE+\-.]+)\s*"
        r"\(sd=([0-9eE+\-.]+);\s*n_frames=(\d+);\s*method=([^)]+)\)"
    )
    status_pattern = re.compile(r"status=([A-Za-z0-9_\-]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            mean_match = mean_pattern.search(line)
            if mean_match:
                parsed["paper_path_ap_contact_last10ns_mean"] = mean_match.group(1)
                parsed["paper_path_ap_contact_last10ns_sd"] = mean_match.group(2)
                parsed["paper_path_ap_contact_last10ns_n_frames"] = mean_match.group(3)
                parsed["paper_path_ap_contact_last10ns_method"] = mean_match.group(4)
                parsed["paper_path_ap_contact_last10ns_status"] = "computed"
                continue
            status_match = status_pattern.search(line)
            if status_match and not parsed["paper_path_ap_contact_last10ns_status"]:
                parsed["paper_path_ap_contact_last10ns_status"] = status_match.group(1)
    return parsed


def _paper_ap_contact_file_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    midpoint_match = re.search(r"midpoint_nm=([0-9eE+\-.]+)", text)
    steepness_match = re.search(r"steepness=([0-9eE+\-.]+)", text)
    if not midpoint_match or not steepness_match:
        return False
    try:
        midpoint = float(midpoint_match.group(1))
        steepness = float(steepness_match.group(1))
    except ValueError:
        return False
    return math.isclose(midpoint, PAPER_AP_CONTACT_MIDPOINT_NM, rel_tol=0.0, abs_tol=1e-9) and math.isclose(
        steepness,
        PAPER_AP_CONTACT_STEEPNESS,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _ap_contact_same_paper_formula_file_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    low_match = re.search(r"full_weight_below_nm=([0-9eE+\-.]+)", text)
    high_match = re.search(r"zero_weight_above_nm=([0-9eE+\-.]+)", text)
    if not low_match or not high_match:
        return False
    try:
        low = float(low_match.group(1))
        high = float(high_match.group(1))
    except ValueError:
        return False
    return math.isclose(
        low,
        PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM,
        rel_tol=0.0,
        abs_tol=1e-9,
    ) and math.isclose(
        high,
        PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _paper_path_ap_contact_file_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return (
        "path_score=maximum mean consecutive-edge weight over Hamiltonian paths" in text
        and f"full_weight_below_nm={PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:g}" in text
        and f"zero_weight_above_nm={PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:g}" in text
    )


def _gro_box_lengths(lines: list[str]) -> tuple[float, float, float] | None:
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _gro_protein_beads(path: Path, sequence_length: int) -> tuple[list[tuple[int, float, float, float]], int, tuple[float, float, float] | None]:
    amino_acids = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    box = _gro_box_lengths(lines)
    beads: list[tuple[int, float, float, float]] = []
    residue_counter = 0
    previous_residue: tuple[int, str] | None = None
    for line in lines[2:-1]:
        if len(line) < 44:
            continue
        residue_name = line[5:10].strip()
        if residue_name not in amino_acids:
            if beads:
                break
            continue
        try:
            residue_id = int(line[:5])
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
        except ValueError:
            continue
        residue_key = (residue_id, residue_name)
        if residue_key != previous_residue:
            residue_counter += 1
            previous_residue = residue_key
        molecule_index = max(0, (residue_counter - 1) // sequence_length)
        beads.append((molecule_index, x, y, z))
    molecule_count = 0 if not beads else max(bead[0] for bead in beads) + 1
    return beads, molecule_count, box


def _minimum_image_delta(delta: float, box_length: float | None) -> float:
    if box_length and box_length > 0:
        delta -= round(delta / box_length) * box_length
    return delta


def _ap_contact_for_gro(path: Path, sequence_length: int, *, cutoff_nm: float = AP_CONTACT_CUTOFF_NM) -> tuple[float, int, int]:
    beads, molecule_count, box = _gro_protein_beads(path, sequence_length)
    graph = _contact_graph_from_beads(beads, molecule_count, box, cutoff_nm=cutoff_nm)
    return graph["ap_contact"], graph["contacted_molecules"], graph["molecule_count"]


def _paper_ap_contact_weight(distance_nm: float) -> float:
    exponent = PAPER_AP_CONTACT_STEEPNESS * (distance_nm - PAPER_AP_CONTACT_MIDPOINT_NM)
    if exponent > 60:
        return 0.0
    if exponent < -60:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def _ap_contact_same_paper_formula_weight(distance_nm: float) -> float:
    if distance_nm <= PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:
        return 1.0
    if distance_nm >= PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:
        return 0.0
    distance_angstrom = distance_nm * 10.0
    return math.exp(-(distance_angstrom - 4.0))


def _mean_per_peptide_strongest_contact_weight_for_gro(
    path: Path,
    sequence_length: int,
    weight_fn,
) -> tuple[float, int]:
    beads, molecule_count, box = _gro_protein_beads(path, sequence_length)
    if molecule_count <= 1 or not beads:
        return 0.0, 0
    molecules: list[list[tuple[float, float, float]]] = [[] for _ in range(molecule_count)]
    for molecule_index, x, y, z in beads:
        if 0 <= molecule_index < molecule_count:
            molecules[molecule_index].append((x, y, z))
    best_weights = [0.0 for _ in range(molecule_count)]
    pair_count = 0
    for i in range(molecule_count):
        if not molecules[i]:
            continue
        for j in range(i + 1, molecule_count):
            if not molecules[j]:
                continue
            min_sq: float | None = None
            for x, y, z in molecules[i]:
                for ox, oy, oz in molecules[j]:
                    dx = _minimum_image_delta(x - ox, box[0] if box else None)
                    dy = _minimum_image_delta(y - oy, box[1] if box else None)
                    dz = _minimum_image_delta(z - oz, box[2] if box else None)
                    dist_sq = dx * dx + dy * dy + dz * dz
                    if min_sq is None or dist_sq < min_sq:
                        min_sq = dist_sq
            if min_sq is None:
                continue
            weight = weight_fn(math.sqrt(min_sq))
            if weight > best_weights[i]:
                best_weights[i] = weight
            if weight > best_weights[j]:
                best_weights[j] = weight
            pair_count += 1
    return (sum(best_weights) / molecule_count if pair_count else 0.0), pair_count


def _paper_ap_contact_for_gro(path: Path, sequence_length: int) -> tuple[float, int]:
    """Paper-compatible APcontact from weighted closest inter-peptide contacts.

    The older diagnostic AP_contact records the fraction of molecules with at
    least one hard-cutoff contact. This score uses a smooth closest-distance
    weight and averages each peptide's strongest interpeptide contact. That
    preserves the APcontact threshold behavior without the hard-cutoff
    saturation seen in the legacy diagnostic.
    """

    return _mean_per_peptide_strongest_contact_weight_for_gro(path, sequence_length, _paper_ap_contact_weight)


def _ap_contact_same_paper_formula_for_gro(path: Path, sequence_length: int) -> tuple[float, int]:
    """AP contact using the literal distance-weight equation from the paper.

    The aggregation over peptides intentionally mirrors the existing
    paper_APcontact diagnostic so the two columns differ only in the distance
    weighting function.
    """

    return _mean_per_peptide_strongest_contact_weight_for_gro(
        path,
        sequence_length,
        _ap_contact_same_paper_formula_weight,
    )


def _paper_formula_pair_weight_matrix(
    molecules: list[list[tuple[float, float, float]]],
    box: tuple[float, float, float] | None,
) -> list[list[float]]:
    molecule_count = len(molecules)
    weights = [[0.0 for _ in range(molecule_count)] for _ in range(molecule_count)]
    cutoff_nm = PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM
    cutoff_sq = cutoff_nm * cutoff_nm
    cells: dict[tuple[int, int, int], list[tuple[int, float, float, float]]] = {}
    cell_counts = tuple(max(1, int(length // cutoff_nm) + 1) for length in box) if box else None
    closest_sq: dict[tuple[int, int], float] = {}
    for molecule_index, molecule in enumerate(molecules):
        for x, y, z in molecule:
            raw_cell = (int(x // cutoff_nm), int(y // cutoff_nm), int(z // cutoff_nm))
            cell = (
                raw_cell[0] % cell_counts[0],
                raw_cell[1] % cell_counts[1],
                raw_cell[2] % cell_counts[2],
            ) if cell_counts else raw_cell
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for dz_cell in (-1, 0, 1):
                        neighbor_cell = (cell[0] + dx_cell, cell[1] + dy_cell, cell[2] + dz_cell)
                        if cell_counts:
                            neighbor_cell = (
                                neighbor_cell[0] % cell_counts[0],
                                neighbor_cell[1] % cell_counts[1],
                                neighbor_cell[2] % cell_counts[2],
                            )
                        for other_molecule, ox, oy, oz in cells.get(neighbor_cell, []):
                            if other_molecule == molecule_index:
                                continue
                            dx = _minimum_image_delta(x - ox, box[0] if box else None)
                            dy = _minimum_image_delta(y - oy, box[1] if box else None)
                            dz = _minimum_image_delta(z - oz, box[2] if box else None)
                            dist_sq = dx * dx + dy * dy + dz * dz
                            if dist_sq > cutoff_sq:
                                continue
                            pair = tuple(sorted((molecule_index, other_molecule)))
                            previous = closest_sq.get(pair)
                            if previous is None or dist_sq < previous:
                                closest_sq[pair] = dist_sq
            cells.setdefault(cell, []).append((molecule_index, x, y, z))
    for (i, j), dist_sq in closest_sq.items():
        weight = _ap_contact_same_paper_formula_weight(math.sqrt(dist_sq))
        weights[i][j] = weight
        weights[j][i] = weight
    return weights


def _max_hamiltonian_path_mean_weight(weights: list[list[float]]) -> tuple[float, str]:
    n = len(weights)
    if n <= 1:
        return 0.0, "trivial"
    if n <= PAPER_PATH_AP_CONTACT_EXACT_MAX_MOLECULES:
        full_mask = (1 << n) - 1
        dp = [[float("-inf") for _ in range(n)] for _ in range(1 << n)]
        for node in range(n):
            dp[1 << node][node] = 0.0
        for mask in range(1 << n):
            for last in range(n):
                current = dp[mask][last]
                if current == float("-inf"):
                    continue
                remaining = full_mask ^ mask
                bitset = remaining
                while bitset:
                    bit = bitset & -bitset
                    nxt = bit.bit_length() - 1
                    new_mask = mask | bit
                    candidate = current + weights[last][nxt]
                    if candidate > dp[new_mask][nxt]:
                        dp[new_mask][nxt] = candidate
                    bitset ^= bit
        best_sum = max(dp[full_mask])
        return best_sum / (n - 1), "exact_dynamic_programming"

    # Exact Hamiltonian-path optimization is exponential. For full CG systems
    # with many peptide copies, keep the paper's path-scoring structure but use
    # a deterministic multi-start beam search.
    states: list[tuple[float, int, tuple[int, ...], frozenset[int]]] = [
        (0.0, start, (start,), frozenset({start})) for start in range(n)
    ]
    for _step in range(1, n):
        candidates: list[tuple[float, int, tuple[int, ...], frozenset[int]]] = []
        for score_sum, last, path, visited in states:
            next_nodes = sorted(
                (node for node in range(n) if node not in visited),
                key=lambda node: weights[last][node],
                reverse=True,
            )[:PAPER_PATH_AP_CONTACT_BRANCHING]
            for nxt in next_nodes:
                new_path = (*path, nxt)
                new_visited = visited | {nxt}
                candidates.append((score_sum + weights[last][nxt], nxt, new_path, new_visited))
        if not candidates:
            break
        candidates.sort(key=lambda state: state[0], reverse=True)
        states = candidates[:PAPER_PATH_AP_CONTACT_BEAM_WIDTH]
    best_sum = max((state[0] for state in states if len(state[2]) == n), default=0.0)
    return best_sum / (n - 1), (
        f"deterministic_beam_search_width_{PAPER_PATH_AP_CONTACT_BEAM_WIDTH}"
        f"_branching_{PAPER_PATH_AP_CONTACT_BRANCHING}"
    )


def _paper_path_ap_contact_for_gro(path: Path, sequence_length: int) -> tuple[float, int, str]:
    """APcontact using the paper distance formula and path-based scoring.

    The paper describes constructing paths that visit each peptide once,
    scoring inter-peptide distances along each path, and taking the maximum
    weighted-average value as APcontact. This implements that structure over
    the complete graph of peptide copies using the reported distance weight.
    """

    beads, molecule_count, box = _gro_protein_beads(path, sequence_length)
    if molecule_count <= 1 or not beads:
        return 0.0, 0, "trivial"
    molecules: list[list[tuple[float, float, float]]] = [[] for _ in range(molecule_count)]
    for molecule_index, x, y, z in beads:
        if 0 <= molecule_index < molecule_count:
            molecules[molecule_index].append((x, y, z))
    weights = _paper_formula_pair_weight_matrix(molecules, box)
    score, search_method = _max_hamiltonian_path_mean_weight(weights)
    return score, molecule_count * (molecule_count - 1) // 2, search_method


def _contact_graph_from_beads(
    beads: list[tuple[int, float, float, float]],
    molecule_count: int,
    box: tuple[float, float, float] | None,
    *,
    cutoff_nm: float,
) -> dict[str, object]:
    if molecule_count <= 0 or not beads:
        return {
            "ap_contact": 0.0,
            "contacted_molecules": 0,
            "molecule_count": 0,
            "cluster_count": 0,
            "largest_cluster_size": 0,
            "largest_cluster_fraction": 0.0,
            "singleton_count": 0,
            "singleton_fraction": 0.0,
            "mean_contacts_per_molecule": 0.0,
        }
    cutoff_sq = cutoff_nm * cutoff_nm
    cells: dict[tuple[int, int, int], list[tuple[int, float, float, float]]] = {}
    cell_counts = tuple(max(1, int(length // cutoff_nm) + 1) for length in box) if box else None
    contacted: set[int] = set()
    adjacency: list[set[int]] = [set() for _ in range(molecule_count)]
    for molecule_index, x, y, z in beads:
        raw_cell = (int(x // cutoff_nm), int(y // cutoff_nm), int(z // cutoff_nm))
        cell = (
            raw_cell[0] % cell_counts[0],
            raw_cell[1] % cell_counts[1],
            raw_cell[2] % cell_counts[2],
        ) if cell_counts else raw_cell
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                for dz_cell in (-1, 0, 1):
                    neighbor_cell = (cell[0] + dx_cell, cell[1] + dy_cell, cell[2] + dz_cell)
                    if cell_counts:
                        neighbor_cell = (
                            neighbor_cell[0] % cell_counts[0],
                            neighbor_cell[1] % cell_counts[1],
                            neighbor_cell[2] % cell_counts[2],
                        )
                    for other_molecule, ox, oy, oz in cells.get(neighbor_cell, []):
                        if other_molecule == molecule_index:
                            continue
                        dx = _minimum_image_delta(x - ox, box[0] if box else None)
                        dy = _minimum_image_delta(y - oy, box[1] if box else None)
                        dz = _minimum_image_delta(z - oz, box[2] if box else None)
                        if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                            contacted.add(molecule_index)
                            contacted.add(other_molecule)
                            adjacency[molecule_index].add(other_molecule)
                            adjacency[other_molecule].add(molecule_index)
        cells.setdefault(cell, []).append((molecule_index, x, y, z))
    cluster_sizes: list[int] = []
    seen: set[int] = set()
    for molecule_index in range(molecule_count):
        if molecule_index in seen:
            continue
        stack = [molecule_index]
        seen.add(molecule_index)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        cluster_sizes.append(size)
    singleton_count = sum(1 for size in cluster_sizes if size == 1)
    largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    contact_degrees = [len(neighbors) for neighbors in adjacency]
    return {
        "ap_contact": len(contacted) / molecule_count,
        "contacted_molecules": len(contacted),
        "molecule_count": molecule_count,
        "cluster_count": len(cluster_sizes),
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": largest_cluster_size / molecule_count,
        "singleton_count": singleton_count,
        "singleton_fraction": singleton_count / molecule_count,
        "mean_contacts_per_molecule": sum(contact_degrees) / molecule_count,
    }


def _contact_frame_paths(package_dir: Path, sequence: str) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    pattern = re.compile(rf"^{re.escape(sequence)}_(\d+)ns\.gro$")
    for path in package_dir.glob(f"{sequence}_*ns.gro"):
        match = pattern.match(path.name)
        if match:
            frames[int(match.group(1))] = path
    final_frame = package_dir / f"{sequence}_200_CG.gro"
    if 200 not in frames and final_frame.exists():
        frames[200] = final_frame
    return frames


def _write_ap_contact_file(package_dir: Path, sequence: str) -> Path | None:
    frames = _contact_frame_paths(package_dir, sequence)
    if not frames:
        return None
    output_path = package_dir / f"{sequence}_AP_contact.txt"
    lines = [
        "# AP_contact diagnostic definition used by active_learning_thesis.",
        f"# Score = fraction of peptide molecules with at least one inter-peptide bead contact within {AP_CONTACT_CUTOFF_NM:g} nm.",
        "# This is supporting contact-fraction evidence and does not define the retained Phase 3 label.",
        "# The primary Phase 3 contact criterion is paper_path_APcontact_last10ns >= 0.5.",
    ]
    for target in AP_TARGET_NS:
        frame = frames.get(target)
        if frame is None:
            continue
        score, contacted, total = _ap_contact_for_gro(frame, len(sequence))
        lines.append(
            f"The AP_contact for {target} ns is: {score:.6f} "
            f"(contacted_molecules={contacted}; total_molecules={total}; cutoff_nm={AP_CONTACT_CUTOFF_NM:g}; source={frame.name})"
        )
    if len(lines) == 4:
        return None
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_paper_ap_contact_file(package_dir: Path, sequence: str) -> Path | None:
    frames = _contact_frame_paths(package_dir, sequence)
    if not frames:
        return None
    output_path = package_dir / f"{sequence}_paper_APcontact.txt"
    lines = [
        "# Paper-compatible APcontact used by active_learning_thesis.",
        "# Score = mean per-peptide strongest smooth contact weight from closest inter-peptide bead distances.",
        "# The legacy ap_contact_* fields remain the contacted-molecule-fraction diagnostic.",
        f"# weight(distance_nm)=1/(1+exp({PAPER_AP_CONTACT_STEEPNESS:g}*(distance_nm-{PAPER_AP_CONTACT_MIDPOINT_NM:g}))).",
    ]
    for target in AP_TARGET_NS:
        frame = frames.get(target)
        if frame is None:
            continue
        score, pair_count = _paper_ap_contact_for_gro(frame, len(sequence))
        lines.append(
            f"The paper_APcontact for {target} ns is: {score:.6f} "
            f"(peptide_pairs={pair_count}; midpoint_nm={PAPER_AP_CONTACT_MIDPOINT_NM:g}; "
            f"steepness={PAPER_AP_CONTACT_STEEPNESS:g}; source={frame.name})"
        )
    if len(lines) == 4:
        return None
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_ap_contact_same_paper_formula_file(package_dir: Path, sequence: str) -> Path | None:
    frames = _contact_frame_paths(package_dir, sequence)
    if not frames:
        return None
    output_path = package_dir / f"{sequence}_AP_contact_same_paper_formula.txt"
    lines = [
        "# AP_contact_same_paper_formula used by active_learning_thesis.",
        "# Score = mean per-peptide strongest inter-peptide closest-bead contact weight.",
        "# Distance weight is the literal piecewise equation reported in the paper:",
        "# w(x)=1 for x<=4 A; w(x)=exp(-(x-4)) for 4 A < x < 12 A; w(x)=0 for x>=12 A.",
        f"# full_weight_below_nm={PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:g}; "
        f"zero_weight_above_nm={PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:g}.",
    ]
    for target in AP_TARGET_NS:
        frame = frames.get(target)
        if frame is None:
            continue
        score, pair_count = _ap_contact_same_paper_formula_for_gro(frame, len(sequence))
        lines.append(
            f"The AP_contact_same_paper_formula for {target} ns is: {score:.6f} "
            f"(peptide_pairs={pair_count}; full_weight_below_nm={PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:g}; "
            f"zero_weight_above_nm={PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:g}; source={frame.name})"
        )
    if len(lines) == 5:
        return None
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_paper_path_ap_contact_file(package_dir: Path, sequence: str) -> Path | None:
    frames = _contact_frame_paths(package_dir, sequence)
    if not frames:
        return None
    output_path = package_dir / f"{sequence}_paper_path_APcontact.txt"
    lines = [
        "# paper_path_APcontact used by active_learning_thesis.",
        "# path_score=maximum mean consecutive-edge weight over Hamiltonian paths",
        "# Edge weights use the paper's piecewise closest inter-peptide bead-distance equation:",
        "# w(x)=1 for x<=4 A; w(x)=exp(-(x-4)) for 4 A < x < 12 A; w(x)=0 for x>=12 A.",
        "# For systems with <=16 peptide copies the maximum path is exact dynamic programming; larger systems use deterministic beam search.",
        f"# full_weight_below_nm={PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:g}; "
        f"zero_weight_above_nm={PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:g}.",
    ]
    for target in AP_TARGET_NS:
        frame = frames.get(target)
        if frame is None:
            continue
        score, pair_count, search_method = _paper_path_ap_contact_for_gro(frame, len(sequence))
        lines.append(
            f"The paper_path_APcontact for {target} ns is: {score:.6f} "
            f"(peptide_pairs={pair_count}; search={search_method}; "
            f"full_weight_below_nm={PAPER_FORMULA_AP_CONTACT_FULL_WEIGHT_NM:g}; "
            f"zero_weight_above_nm={PAPER_FORMULA_AP_CONTACT_ZERO_WEIGHT_NM:g}; source={frame.name})"
        )
    if len(lines) == 6:
        return None
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _paper_path_last10_frame_ns() -> list[int]:
    return list(range(PAPER_PATH_LAST10_START_NS, PAPER_PATH_LAST10_END_NS + 1, PAPER_PATH_LAST10_STEP_NS))


def _paper_path_last10_dir(package_dir: Path, sequence: str) -> Path:
    return package_dir / f"{sequence}_paper_path_last10ns_frames"


def _paper_path_last10_frame_paths(package_dir: Path, sequence: str) -> dict[int, Path]:
    frames_dir = _paper_path_last10_dir(package_dir, sequence)
    frames: dict[int, Path] = {}
    for ns in _paper_path_last10_frame_ns():
        path = frames_dir / f"{sequence}_{ns}ns.gro"
        if path.exists():
            frames[ns] = path
    return frames


def _trajectory_inputs(package_dir: Path, sequence: str) -> tuple[Path | None, Path | None, Path | None]:
    xtc = next(package_dir.glob(f"{sequence}_*_CG.xtc"), None) or next(package_dir.glob("*.xtc"), None)
    tpr = next(package_dir.glob(f"{sequence}_*_CG.tpr"), None) or next(package_dir.glob("*.tpr"), None)
    gro = next(package_dir.glob(f"{sequence}_*_CG.gro"), None) or next(package_dir.glob("*.gro"), None)
    return xtc, tpr, gro


def _write_paper_path_last10_extraction_script(package_dir: Path, sequence: str) -> tuple[Path | None, Path]:
    status_path = package_dir / f"{sequence}_paper_path_APcontact_last10ns_status.json"
    xtc, tpr, gro = _trajectory_inputs(package_dir, sequence)
    frames_dir = _paper_path_last10_dir(package_dir, sequence)
    script_path = package_dir / f"{sequence}_extract_last10ns_for_paper_APcontact.sh"
    status: dict[str, object] = {
        "status": "blocked",
        "sequence": sequence,
        "purpose": "Extract final-10-ns GRO frames for paper-style APcontact averaging.",
        "message": (
            "Run this script on BURA or another environment with GROMACS installed, "
            "then copy the generated frames back and run parse-md-results again."
        ),
        "input_xtc": str(xtc) if xtc else "",
        "input_tpr": str(tpr) if tpr else "",
        "input_gro": str(gro) if gro else "",
        "frames_dir": str(frames_dir),
        "frame_ns": _paper_path_last10_frame_ns(),
        "script": str(script_path),
        "manual_group_selection_required": False,
    }
    if not xtc or not tpr:
        missing = []
        if not xtc:
            missing.append("xtc")
        if not tpr:
            missing.append("tpr")
        status["blocker"] = f"Missing required trajectory input(s): {', '.join(missing)}."
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return None, status_path

    frame_cases = "\n".join(
        f"  {ns}) dump_ps={ns * 1000} ;;" for ns in _paper_path_last10_frame_ns()
    )
    script_text = f"""#!/bin/bash
set -eo pipefail

# Extract final-10-ns snapshots for paper-style APcontact.
# Run this on BURA or another machine with GROMACS installed.
# It is safest to run inside packages/{sequence}, but the script also tolerates
# being launched from the campaign root if packages/{sequence} exists.
# Raw .xtc/.tpr/.gro files are not modified.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

XTC="{xtc.name}"
TPR="{tpr.name}"
OUT_DIR="{frames_dir.name}"

if [[ ! -f "$XTC" && -d "$SCRIPT_DIR/packages/{sequence}" ]]; then
  cd "$SCRIPT_DIR/packages/{sequence}"
fi

if [[ -f ./common.sh ]]; then
  source ./common.sh
elif [[ -f ./env_bura.sh ]]; then
  source ./env_bura.sh
fi

if ! command -v "${{GMX_CMD:-gmx}}" >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then
    module load gromacs/2023.2_g13.1_p3.10.5 >/dev/null 2>&1 || true
  fi
fi

if command -v "${{GMX_CMD:-gmx}}" >/dev/null 2>&1; then
  GMX_RUN="${{GMX_CMD:-gmx}}"
elif command -v gmx_mpi >/dev/null 2>&1; then
  GMX_RUN="gmx_mpi"
elif command -v gmx >/dev/null 2>&1; then
  GMX_RUN="gmx"
else
  echo "GROMACS command not found. Run on BURA after: module load gromacs/2023.2_g13.1_p3.10.5" >&2
  exit 1
fi

[[ -f "$XTC" ]] || {{ echo "Missing trajectory: $PWD/$XTC" >&2; exit 1; }}
[[ -f "$TPR" ]] || {{ echo "Missing run input: $PWD/$TPR" >&2; exit 1; }}

mkdir -p "$OUT_DIR"

for frame_ns in {' '.join(str(ns) for ns in _paper_path_last10_frame_ns())}; do
  case "$frame_ns" in
{frame_cases}
    *) echo "Unsupported frame: $frame_ns" >&2; exit 1 ;;
  esac
  raw="$OUT_DIR/{sequence}_${{frame_ns}}ns.raw.gro"
  out="$OUT_DIR/{sequence}_${{frame_ns}}ns.gro"
  echo "[paper_path_APcontact] extracting $frame_ns ns from $XTC"
  printf "0\\n" | "$GMX_RUN" trjconv -s "$TPR" -f "$XTC" -o "$raw" -dump "$dump_ps"
  if [[ -f index_image.ndx ]]; then
    printf "0\\n" | "$GMX_RUN" trjconv -s "$TPR" -f "$raw" -n index_image.ndx -o "$out.tmp.gro" -pbc whole
    printf "0\\n" | "$GMX_RUN" trjconv -s "$TPR" -f "$out.tmp.gro" -o "$out.tmp2.gro" -pbc nojump
    printf "1\\n0\\n" | "$GMX_RUN" trjconv -s "$TPR" -f "$out.tmp2.gro" -n index_image.ndx -o "$out" -pbc mol -center
    rm -f "$out.tmp.gro" "$out.tmp2.gro"
  else
    mv "$raw" "$out"
  fi
done

echo "[paper_path_APcontact] extracted final-10-ns frames into $OUT_DIR"
echo "[paper_path_APcontact] copy this folder back if needed, then run: python -m active_learning_thesis parse-md-results --campaign-dir <campaign_dir>"
"""
    script_path.write_text(script_text, encoding="utf-8", newline="\n")
    _chmod_shell(script_path)
    status.update(
        {
            "status": "extraction_script_ready",
            "blocker": "",
            "script": str(script_path),
            "command": f"bash {script_path.name}",
        }
    )
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return script_path, status_path


def _write_paper_path_last10_ap_contact_file(package_dir: Path, sequence: str) -> tuple[Path | None, Path | None, Path]:
    frames = _paper_path_last10_frame_paths(package_dir, sequence)
    status_path = package_dir / f"{sequence}_paper_path_APcontact_last10ns_status.json"
    expected = _paper_path_last10_frame_ns()
    script_path, status_path = _write_paper_path_last10_extraction_script(package_dir, sequence)
    output_path = package_dir / f"{sequence}_paper_path_APcontact_last10ns.txt"
    if len(frames) != len(expected):
        missing = [ns for ns in expected if ns not in frames]
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update(
            {
                "status": "extraction_needed",
                "present_frame_ns": sorted(frames),
                "missing_frame_ns": missing,
                "output": str(output_path),
            }
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        lines = [
            "# paper_path_APcontact final-10-ns average is not computed yet.",
            "# status=extraction_needed",
            "# Run the extraction script on BURA/GROMACS, copy frames back, then re-run parse-md-results.",
            f"# expected_frame_ns={','.join(str(ns) for ns in expected)}",
            f"# present_frame_ns={','.join(str(ns) for ns in sorted(frames))}",
            f"# missing_frame_ns={','.join(str(ns) for ns in missing)}",
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path, script_path, status_path

    frame_scores: list[tuple[int, float, int, str]] = []
    for ns in expected:
        score, pair_count, search_method = _paper_path_ap_contact_for_gro(frames[ns], len(sequence))
        frame_scores.append((ns, score, pair_count, search_method))
    values = [score for _ns, score, _pair_count, _method in frame_scores]
    mean_score = sum(values) / len(values)
    sd_score = math.sqrt(sum((value - mean_score) ** 2 for value in values) / len(values)) if len(values) > 1 else 0.0
    methods = sorted({method for _ns, _score, _pair_count, method in frame_scores})
    lines = [
        "# paper_path_APcontact final-10-ns average used by active_learning_thesis.",
        "# Score = mean over extracted final-10-ns frames of the maximum mean consecutive-edge path score.",
        "# Edge weights use the paper's piecewise closest inter-peptide bead-distance equation:",
        "# w(x)=1 for x<=4 A; w(x)=exp(-(x-4)) for 4 A < x < 12 A; w(x)=0 for x>=12 A.",
        "# Large systems use deterministic beam search for the maximum path; exact DP is used only for <=16 peptide copies.",
        f"The paper_path_APcontact last10ns mean is: {mean_score:.6f} "
        f"(sd={sd_score:.6f}; n_frames={len(values)}; method={'+'.join(methods)})",
    ]
    for ns, score, pair_count, search_method in frame_scores:
        lines.append(
            f"Frame {ns} ns: {score:.6f} "
            f"(peptide_pairs={pair_count}; search={search_method}; source={frames[ns].name})"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {
        "status": "computed",
        "sequence": sequence,
        "frame_ns": expected,
        "n_frames": len(values),
        "mean": mean_score,
        "sd": sd_score,
        "method": "+".join(methods),
        "output": str(output_path),
        "script": str(script_path) if script_path else "",
        "message": "Final-10-ns paper_path_APcontact average computed from extracted GRO frames.",
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return output_path, script_path, status_path


def _aggregate_summary_fields() -> list[str]:
    return [
        "frame_ns",
        "source",
        "cutoff_nm",
        "molecule_count",
        "contacted_molecules",
        "ap_contact",
        "cluster_count",
        "largest_cluster_size",
        "largest_cluster_fraction",
        "singleton_count",
        "singleton_fraction",
        "mean_contacts_per_molecule",
    ]


def _write_aggregate_summary_file(package_dir: Path, sequence: str) -> Path | None:
    frames = _contact_frame_paths(package_dir, sequence)
    if not frames:
        return None
    output_path = package_dir / f"{sequence}_aggregate_summary.csv"
    rows: list[dict[str, str]] = []
    frame_cache: dict[Path, tuple[list[tuple[int, float, float, float]], int, tuple[float, float, float] | None]] = {}
    for target in AP_TARGET_NS:
        frame = frames.get(target)
        if frame is None:
            continue
        if frame not in frame_cache:
            frame_cache[frame] = _gro_protein_beads(frame, len(sequence))
        beads, molecule_count, box = frame_cache[frame]
        for cutoff in AP_CONTACT_SENSITIVITY_CUTOFFS_NM:
            graph = _contact_graph_from_beads(beads, molecule_count, box, cutoff_nm=cutoff)
            rows.append(
                {
                    "frame_ns": str(target),
                    "source": frame.name,
                    "cutoff_nm": f"{cutoff:.2f}",
                    "molecule_count": str(graph["molecule_count"]),
                    "contacted_molecules": str(graph["contacted_molecules"]),
                    "ap_contact": f"{float(graph['ap_contact']):.6f}",
                    "cluster_count": str(graph["cluster_count"]),
                    "largest_cluster_size": str(graph["largest_cluster_size"]),
                    "largest_cluster_fraction": f"{float(graph['largest_cluster_fraction']):.6f}",
                    "singleton_count": str(graph["singleton_count"]),
                    "singleton_fraction": f"{float(graph['singleton_fraction']):.6f}",
                    "mean_contacts_per_molecule": f"{float(graph['mean_contacts_per_molecule']):.6f}",
                }
            )
    if not rows:
        return None
    _save_csv(output_path, _aggregate_summary_fields(), rows)
    return output_path


def _parse_aggregate_summary_file(path: Path) -> dict[str, str]:
    parsed = {
        "cluster_largest_fraction_200ns": "",
        "cluster_count_200ns": "",
        "cluster_singleton_fraction_200ns": "",
        "cluster_mean_contacts_200ns": "",
    }
    for row in _read_csv(path):
        if row.get("frame_ns") != "200" or row.get("cutoff_nm") != f"{AP_CONTACT_CUTOFF_NM:.2f}":
            continue
        parsed["cluster_largest_fraction_200ns"] = row.get("largest_cluster_fraction", "")
        parsed["cluster_count_200ns"] = row.get("cluster_count", "")
        parsed["cluster_singleton_fraction_200ns"] = row.get("singleton_fraction", "")
        parsed["cluster_mean_contacts_200ns"] = row.get("mean_contacts_per_molecule", "")
        break
    return parsed


def _format_duration_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_gromacs_runtime(path: Path) -> dict[str, str]:
    parsed = {
        "md_runtime_wall_hms": "",
        "md_runtime_wall_seconds": "",
        "md_runtime_core_seconds": "",
        "md_runtime_ns_per_day": "",
    }
    text = path.read_text(encoding="utf-8", errors="replace")
    time_matches = re.findall(r"(?m)^\s*Time:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
    if time_matches:
        core_seconds, wall_seconds, _percent = time_matches[-1]
        parsed["md_runtime_core_seconds"] = core_seconds
        parsed["md_runtime_wall_seconds"] = wall_seconds
        try:
            parsed["md_runtime_wall_hms"] = _format_duration_hms(float(wall_seconds))
        except ValueError:
            pass
    performance_matches = re.findall(r"(?m)^\s*Performance:\s+([0-9.]+)\s+([0-9.]+)", text)
    if performance_matches:
        parsed["md_runtime_ns_per_day"] = performance_matches[-1][0]
    return parsed


def _matching_runtime_outputs(package_dir: Path, sequence: str, suffix: str) -> list[Path]:
    return sorted(package_dir.glob(f"{sequence}*_CG{suffix}"))


def _ap_metrics_present(path: Path) -> bool:
    return any(value for value in _parse_ap_file(path).values())


def _job_status(package_dir: Path, sequence: str, md_profile: str) -> str:
    if not (package_dir / f"{sequence}.pdb").exists():
        return "pdb_missing"
    if _profile_has_analysis(md_profile):
        if _matching_runtime_outputs(package_dir, sequence, ".xtc"):
            ap_file = next(package_dir.glob("*_AP_SASA.txt"), None)
            if ap_file and _ap_metrics_present(ap_file):
                return "analysis_complete"
            sasa_candidates = [path for path in package_dir.glob("*_sasa.xvg") if "paper_initial" not in path.name]
            if sasa_candidates or ap_file:
                return "sasa_complete"
            return "dynamics_complete"
        return "package_prepared"

    dynamics_artifacts = []
    for suffix in (".gro", ".cpt", ".xtc", ".tpr"):
        dynamics_artifacts.extend(_matching_runtime_outputs(package_dir, sequence, suffix))
    if dynamics_artifacts:
        return "dynamics_complete"
    return "package_prepared"


def parse_md_results(campaign_dir: Path) -> Path:
    manifest_rows = _read_csv(campaign_dir / "manifest.csv")
    review_path = campaign_dir / "md_review.csv"
    sequence_counts: dict[str, int] = {}
    for row in manifest_rows:
        sequence_counts[row["sequence"]] = sequence_counts.get(row["sequence"], 0) + 1
    existing_rows_by_package = {}
    existing_rows_by_sequence = {}
    if review_path.exists():
        for row in _read_csv(review_path):
            package_key = row.get("package_dir", "")
            if package_key:
                existing_rows_by_package[package_key] = row
            sequence_key = row.get("sequence", "")
            if sequence_key and sequence_counts.get(sequence_key, 0) <= 1:
                existing_rows_by_sequence[sequence_key] = row

    review_rows: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        sequence = manifest_row["sequence"]
        package_dir = campaign_dir / manifest_row["package_dir"]
        review_row = _empty_review_row(manifest_row)
        previous = existing_rows_by_package.get(manifest_row["package_dir"]) or existing_rows_by_sequence.get(sequence)
        if previous:
            review_row["review_notes"] = previous.get("review_notes", "")
            review_row["cgmd_label"] = previous.get("cgmd_label", "")
            for field in LABEL_REVIEW_FIELDS:
                review_row[field] = previous.get(field, "")
        review_row["job_root_status"] = _job_status(package_dir, sequence, manifest_row["md_profile"])

        sasa_candidates = [path for path in sorted(package_dir.glob("*_sasa.xvg")) if "paper_initial" not in path.name]
        sasa_file = sasa_candidates[0] if sasa_candidates else None
        ap_file = next(package_dir.glob("*_AP_SASA.txt"), None)
        ap_contact_file = next(package_dir.glob("*_AP_contact.txt"), None)
        paper_ap_contact_file = next(package_dir.glob("*_paper_APcontact.txt"), None)
        ap_contact_same_paper_formula_file = next(package_dir.glob("*_AP_contact_same_paper_formula.txt"), None)
        paper_path_ap_contact_file = next(package_dir.glob("*_paper_path_APcontact.txt"), None)
        paper_path_last10_ap_contact_file = next(package_dir.glob("*_paper_path_APcontact_last10ns.txt"), None)
        paper_path_last10_script = next(package_dir.glob("*_extract_last10ns_for_paper_APcontact.sh"), None)
        paper_path_last10_status_file = next(package_dir.glob("*_paper_path_APcontact_last10ns_status.json"), None)
        paper_ap_sasa_last10_file = next(package_dir.glob("*_paper_APSASA_last10ns.txt"), None)
        paper_ap_sasa_script = next(package_dir.glob("*_recompute_paper_APSASA.sh"), None)
        paper_ap_sasa_status_file = next(package_dir.glob("*_paper_APSASA_last10ns_status.json"), None)
        aggregate_summary_file = next(package_dir.glob("*_aggregate_summary.csv"), None)
        if not ap_contact_file and _profile_has_analysis(manifest_row["md_profile"]):
            ap_contact_file = _write_ap_contact_file(package_dir, sequence)
        if paper_ap_contact_file and not _paper_ap_contact_file_is_current(paper_ap_contact_file):
            paper_ap_contact_file = None
        if not paper_ap_contact_file and _profile_has_analysis(manifest_row["md_profile"]):
            paper_ap_contact_file = _write_paper_ap_contact_file(package_dir, sequence)
        if ap_contact_same_paper_formula_file and not _ap_contact_same_paper_formula_file_is_current(ap_contact_same_paper_formula_file):
            ap_contact_same_paper_formula_file = None
        if not ap_contact_same_paper_formula_file and _profile_has_analysis(manifest_row["md_profile"]):
            ap_contact_same_paper_formula_file = _write_ap_contact_same_paper_formula_file(package_dir, sequence)
        if paper_path_ap_contact_file and not _paper_path_ap_contact_file_is_current(paper_path_ap_contact_file):
            paper_path_ap_contact_file = None
        if not paper_path_ap_contact_file and _profile_has_analysis(manifest_row["md_profile"]):
            paper_path_ap_contact_file = _write_paper_path_ap_contact_file(package_dir, sequence)
        if _profile_has_analysis(manifest_row["md_profile"]):
            paper_path_last10_ap_contact_file, paper_path_last10_script, paper_path_last10_status_file = (
                _write_paper_path_last10_ap_contact_file(package_dir, sequence)
            )
            paper_ap_sasa_last10_file, paper_ap_sasa_script, paper_ap_sasa_status_file = (
                _write_paper_ap_sasa_last10_file(package_dir, sequence)
            )
        if not aggregate_summary_file and _profile_has_analysis(manifest_row["md_profile"]):
            aggregate_summary_file = _write_aggregate_summary_file(package_dir, sequence)
        review_row["sasa_file"] = str(sasa_file.relative_to(campaign_dir)) if sasa_file else ""
        review_row["ap_file"] = str(ap_file.relative_to(campaign_dir)) if ap_file else ""
        review_row["ap_contact_file"] = str(ap_contact_file.relative_to(campaign_dir)) if ap_contact_file else ""
        review_row["paper_ap_contact_file"] = str(paper_ap_contact_file.relative_to(campaign_dir)) if paper_ap_contact_file else ""
        review_row["ap_contact_same_paper_formula_file"] = (
            str(ap_contact_same_paper_formula_file.relative_to(campaign_dir)) if ap_contact_same_paper_formula_file else ""
        )
        review_row["paper_path_ap_contact_file"] = (
            str(paper_path_ap_contact_file.relative_to(campaign_dir)) if paper_path_ap_contact_file else ""
        )
        review_row["paper_path_ap_contact_last10ns_file"] = (
            str(paper_path_last10_ap_contact_file.relative_to(campaign_dir)) if paper_path_last10_ap_contact_file else ""
        )
        review_row["paper_path_ap_contact_last10ns_script"] = (
            str(paper_path_last10_script.relative_to(campaign_dir)) if paper_path_last10_script else ""
        )
        review_row["paper_path_ap_contact_last10ns_status_file"] = (
            str(paper_path_last10_status_file.relative_to(campaign_dir)) if paper_path_last10_status_file else ""
        )
        review_row["paper_ap_sasa_last10ns_file"] = (
            str(paper_ap_sasa_last10_file.relative_to(campaign_dir)) if paper_ap_sasa_last10_file else ""
        )
        review_row["paper_ap_sasa_recompute_script"] = (
            str(paper_ap_sasa_script.relative_to(campaign_dir)) if paper_ap_sasa_script else ""
        )
        review_row["paper_ap_sasa_status_file"] = (
            str(paper_ap_sasa_status_file.relative_to(campaign_dir)) if paper_ap_sasa_status_file else ""
        )
        review_row["aggregate_summary_file"] = str(aggregate_summary_file.relative_to(campaign_dir)) if aggregate_summary_file else ""

        if ap_file:
            review_row.update(_parse_ap_file(ap_file))
        elif sasa_file:
            values = _parse_xvg_values(sasa_file)
            baseline = values.get(0.0)
            for target in (5, 12, 25, 50, 100, 200):
                current = values.get(float(target))
                if baseline and current:
                    review_row[f"ap_{target}ns"] = str(baseline / current)
        if ap_contact_file:
            review_row.update(_parse_ap_contact_file(ap_contact_file))
        if paper_ap_contact_file:
            review_row.update(_parse_paper_ap_contact_file(paper_ap_contact_file))
        if ap_contact_same_paper_formula_file:
            review_row.update(_parse_ap_contact_same_paper_formula_file(ap_contact_same_paper_formula_file))
        if paper_path_ap_contact_file:
            review_row.update(_parse_paper_path_ap_contact_file(paper_path_ap_contact_file))
        if paper_path_last10_ap_contact_file:
            review_row.update(_parse_paper_path_last10_ap_contact_file(paper_path_last10_ap_contact_file))
        if paper_ap_sasa_last10_file:
            review_row.update(_parse_paper_ap_sasa_last10_file(paper_ap_sasa_last10_file))
        if aggregate_summary_file:
            review_row.update(_parse_aggregate_summary_file(aggregate_summary_file))

        runtime_logs = _matching_runtime_outputs(package_dir, sequence, ".log")
        if runtime_logs:
            review_row.update(_parse_gromacs_runtime(runtime_logs[-1]))

        review_rows.append(review_row)

    _save_csv(review_path, REVIEW_FIELDS, review_rows)
    return review_path


def make_md_ingest_csv(campaign_dir: Path, review_csv: Path) -> Path:
    review_rows = _read_csv(review_csv)
    if not review_rows:
        raise ValueError("Review CSV is empty.")
    ingest_rows: list[dict[str, str]] = []
    for row in review_rows:
        label = row.get("cgmd_label", "").strip()
        if label not in {"0", "1"}:
            raise ValueError(
                f"Each review row must include cgmd_label in {{0,1}}. Problem sequence: {row.get('sequence', '')}"
            )
        ingest_rows.append(
            {
                "sequence": row["sequence"].strip(),
                "round_id": row["round_id"].strip(),
                "cgmd_label": label,
            }
        )
    ingest_path = campaign_dir / "cgmd_ingest.csv"
    _save_csv(ingest_path, ["sequence", "round_id", "cgmd_label"], ingest_rows)
    return ingest_path


def _normalize_benchmark_layouts(layouts: Iterable[str] | str | None) -> list[str]:
    if layouts is None:
        return list(DEFAULT_BURA_BENCHMARK_LAYOUTS)
    if isinstance(layouts, str):
        requested = [item.strip() for item in layouts.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in layouts if str(item).strip()]
    unknown = [item for item in requested if item not in DEFAULT_BURA_BENCHMARK_LAYOUTS]
    if unknown:
        raise ValueError(
            "Unknown BURA benchmark layout(s): "
            + ", ".join(unknown)
            + ". Available layouts: "
            + ", ".join(DEFAULT_BURA_BENCHMARK_LAYOUTS)
        )
    return requested or list(DEFAULT_BURA_BENCHMARK_LAYOUTS)


def _benchmark_package_inputs(source_package_dir: Path) -> list[Path]:
    required = [
        source_package_dir / "common.sh",
        source_package_dir / "env_bura.sh",
        source_package_dir / "topol.top",
        source_package_dir / "equi2.gro",
        source_package_dir / "equi2.cpt",
        source_package_dir / PRODUCTION_MDP,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot prepare BURA benchmark because the source package is missing: "
            + ", ".join(missing)
            + ". Run or copy back the equilibration/full package first."
        )
    copied = [
        source_package_dir / "common.sh",
        source_package_dir / "env_bura.sh",
        source_package_dir / "topol.top",
        source_package_dir / "equi2.gro",
        source_package_dir / "equi2.cpt",
        source_package_dir / PRODUCTION_MDP,
    ]
    copied.extend(sorted(source_package_dir.glob("*.itp")))
    # The benchmark starts from equi2 and intentionally avoids heavy trajectories.
    return sorted(set(copied))


def _benchmark_run_script(sequence: str, layout_name: str, layout: dict[str, int], *, walltime: str) -> str:
    safe_job = f"bench_{sequence[:8]}_{layout_name}"[:64]
    return f"""#!/bin/bash
#SBATCH --job-name={safe_job}
#SBATCH -o benchmark.out
#SBATCH -e benchmark.err
#SBATCH --nodes={layout["nodes"]}
#SBATCH --ntasks-per-node={layout["ntasks_per_node"]}
#SBATCH --cpus-per-task={layout["cpus_per_task"]}
#SBATCH --time={walltime}

set -euo pipefail
SCRIPT_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export GMX_MAXCONSTRWARN=-1
echo "BURA benchmark layout: {layout_name}"
echo "SLURM_NTASKS=${{SLURM_NTASKS:-unknown}} OMP_NUM_THREADS=$OMP_NUM_THREADS"

"$GMX_CMD" grompp -f martini_22P_md_benchmark.mdp -c equi2.gro -t equi2.cpt -p topol.top -o benchmark.tpr
mpirun -np "${{SLURM_NTASKS:-1}}" "$GMX_CMD" mdrun -v -s benchmark.tpr -deffnm benchmark
"""


def _benchmark_readme(
    benchmark_dir: Path,
    *,
    sequence: str,
    nsteps: int,
    layouts: list[str],
) -> str:
    layout_lines = "\n".join(f"- `{layout}`" for layout in layouts)
    return f"""# BURA MD Performance Benchmark

This is an optional benchmark sandbox for `{sequence}`. It does not modify the real MD campaign.

It rewrites only `martini_22P_md.mdp` to `{nsteps}` production steps and tests several SLURM/MPI/OpenMP layouts with BURA's Intel `mpirun`.

Layouts:
{layout_lines}

## On BURA

From the uploaded benchmark folder:

```bash
find . -type f -name "*.sh" -exec dos2unix {{}} \\;
find . -type f -name "*.sh" -exec chmod u+x {{}} \\;
bash submit_bura_benchmarks.sh
```

When all jobs finish, copy this benchmark folder back and run:

```bash
python -m active_learning_thesis parse-bura-md-benchmark --benchmark-dir "{benchmark_dir}"
```

The parser writes `benchmark_results.csv`. Pick the fastest stable layout before changing the real full-MD launcher.
"""


def prepare_bura_md_benchmark(
    campaign_dir: Path,
    sequence: str,
    *,
    benchmark_name: str = "bura_perf",
    nsteps: int = 50000,
    layouts: Iterable[str] | str | None = None,
    walltime: str = "02:00:00",
) -> Path:
    if nsteps <= 0:
        raise ValueError("nsteps must be positive.")
    sequence = sequence.strip().upper()
    if not CANONICAL_SEQUENCE.fullmatch(sequence):
        raise ValueError(f"Invalid peptide sequence for benchmark: {sequence!r}")
    selected_layouts = _normalize_benchmark_layouts(layouts)
    source_package_dir = campaign_dir / "packages" / sequence
    if not source_package_dir.exists():
        raise FileNotFoundError(f"Source package does not exist: {source_package_dir}")

    benchmark_dir = campaign_dir / "bura_benchmarks" / benchmark_name
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    source_inputs = _benchmark_package_inputs(source_package_dir)
    for layout_name in selected_layouts:
        layout = DEFAULT_BURA_BENCHMARK_LAYOUTS[layout_name]
        layout_dir = benchmark_dir / layout_name
        layout_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_inputs:
            target = layout_dir / source_path.name
            shutil.copy2(source_path, target)
        mdp_text = (source_package_dir / PRODUCTION_MDP).read_text(encoding="utf-8")
        _write_text(layout_dir / "martini_22P_md_benchmark.mdp", _rewrite_nsteps(mdp_text, nsteps))
        _write_text(layout_dir / "run_benchmark.sh", _benchmark_run_script(sequence, layout_name, layout, walltime=walltime))
        total_tasks = layout["nodes"] * layout["ntasks_per_node"]
        manifest_rows.append(
            {
                "layout": layout_name,
                "nodes": str(layout["nodes"]),
                "ntasks_per_node": str(layout["ntasks_per_node"]),
                "cpus_per_task": str(layout["cpus_per_task"]),
                "total_mpi_tasks": str(total_tasks),
                "total_requested_cpus": str(total_tasks * layout["cpus_per_task"]),
                "status": "prepared",
                "md_runtime_wall_hms": "",
                "md_runtime_wall_seconds": "",
                "md_runtime_core_seconds": "",
                "md_runtime_ns_per_day": "",
                "observed_mpi_processes": "",
                "observed_omp_threads": "",
                "log_file": str((layout_dir / "benchmark.log").relative_to(benchmark_dir)),
                "stderr_file": str((layout_dir / "benchmark.err").relative_to(benchmark_dir)),
            }
        )

    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
        'echo "Submitting BURA benchmark layouts from $ROOT_DIR"',
    ]
    for layout_name in selected_layouts:
        submit_lines.append(f'(cd "$ROOT_DIR/{layout_name}" && sbatch run_benchmark.sh)')
    _write_text(benchmark_dir / "submit_bura_benchmarks.sh", "\n".join(submit_lines) + "\n")
    _save_csv(benchmark_dir / "benchmark_manifest.csv", BENCHMARK_FIELDS, manifest_rows)
    _write_text(
        benchmark_dir / "NEXT_BURA_BENCHMARK_COMMANDS.md",
        _benchmark_readme(benchmark_dir, sequence=sequence, nsteps=nsteps, layouts=selected_layouts),
    )
    return benchmark_dir


def parse_bura_md_benchmark(benchmark_dir: Path) -> Path:
    manifest_path = benchmark_dir / "benchmark_manifest.csv"
    if manifest_path.exists():
        rows = _read_csv(manifest_path)
    else:
        rows = []
        for layout_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            layout_name = layout_dir.name
            layout = DEFAULT_BURA_BENCHMARK_LAYOUTS.get(layout_name, {})
            nodes = int(layout.get("nodes", 0))
            ntasks_per_node = int(layout.get("ntasks_per_node", 0))
            cpus_per_task = int(layout.get("cpus_per_task", 0))
            total_tasks = nodes * ntasks_per_node
            rows.append(
                {
                    "layout": layout_name,
                    "nodes": str(nodes or ""),
                    "ntasks_per_node": str(ntasks_per_node or ""),
                    "cpus_per_task": str(cpus_per_task or ""),
                    "total_mpi_tasks": str(total_tasks or ""),
                    "total_requested_cpus": str(total_tasks * cpus_per_task if total_tasks and cpus_per_task else ""),
                    "log_file": str((layout_dir / "benchmark.log").relative_to(benchmark_dir)),
                    "stderr_file": str((layout_dir / "benchmark.err").relative_to(benchmark_dir)),
                }
            )

    parsed_rows: list[dict[str, str]] = []
    for row in rows:
        layout_dir = benchmark_dir / row["layout"]
        log_path = layout_dir / "benchmark.log"
        err_path = layout_dir / "benchmark.err"
        parsed = {field: row.get(field, "") for field in BENCHMARK_FIELDS}
        parsed["status"] = "missing_log"
        parsed["log_file"] = str(log_path.relative_to(benchmark_dir))
        parsed["stderr_file"] = str(err_path.relative_to(benchmark_dir))
        text = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            runtime = _parse_gromacs_runtime(log_path)
            parsed.update(runtime)
            parsed["status"] = "finished" if "Finished mdrun" in text else "log_present"
            mpi_match = re.findall(r"(?m)^\s*Using\s+(\d+)\s+MPI process", text)
            omp_match = re.findall(r"(?m)^\s*Using\s+\d+\s+OpenMP thread", text)
            if mpi_match:
                parsed["observed_mpi_processes"] = mpi_match[-1]
            if omp_match:
                thread_match = re.findall(r"(?m)^\s*Using\s+(?:\d+\s+MPI process(?:es)?\s*)?(\d+)\s+OpenMP thread", text)
                if thread_match:
                    parsed["observed_omp_threads"] = thread_match[-1]
        if parsed["status"] != "finished" and err_path.exists():
            err_text = err_path.read_text(encoding="utf-8", errors="replace")
            if "error" in err_text.lower() or "failed" in err_text.lower():
                parsed["status"] = "failed"
        parsed_rows.append(parsed)

    results_path = benchmark_dir / "benchmark_results.csv"
    _save_csv(results_path, BENCHMARK_FIELDS, parsed_rows)
    return results_path
