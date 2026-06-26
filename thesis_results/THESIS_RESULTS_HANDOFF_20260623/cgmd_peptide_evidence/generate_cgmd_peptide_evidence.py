"""Create auditable per-peptide CG-MD evidence exports for thesis writing.

The script intentionally does not edit the thesis LaTeX project.  It combines
Phase 3 review files, primary Phase 4 proposal records, and package-level
aggregate summaries into peptide-level evidence CSV files plus validation and
appendix-ready reports.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

PHASE3_ROOT = (
    REPO_ROOT
    / "thesis_results"
    / "03_real_al"
    / "phase3_results_20260620"
    / "thesis_results"
    / "03_real_al"
)
PHASE4_HANDOFF = (
    REPO_ROOT
    / "thesis_results"
    / "THESIS_RESULTS_HANDOFF_20260623"
    / "04_phase4_bayesian_optimization"
    / "tables"
)

PHASE3_OUT = OUT_DIR / "phase3_cgmd_peptide_evidence.csv"
PHASE4_OUT = OUT_DIR / "phase4_primary_cgmd_peptide_evidence.csv"
VALIDATION_OUT = OUT_DIR / "cgmd_peptide_evidence_validation.md"
APPENDIX_OUT = OUT_DIR / "cgmd_peptide_evidence_appendix.tex"
MAPPING_OUT = OUT_DIR / "cgmd_peptide_evidence_field_mappings.json"

BRANCHES = [
    "predictive_entropy",
    "family_qbc",
    "cluster_diverse_representative",
]
BRANCH_DISPLAY = {
    "predictive_entropy": "predictive entropy",
    "family_qbc": "family QBC",
    "cluster_diverse_representative": "cluster-diverse representative",
}
POLICIES = ["random", "greedy", "ucb", "pi", "ei", "mes"]

AP_THRESHOLD = 1.75
CONTACT_THRESHOLD = 0.5
AGGREGATE_CUTOFF_NM = 0.60
AGGREGATE_FRAME_NS = 200


PHASE3_COLUMNS = [
    "sequence",
    "canonical_sequence",
    "acquisition_branch",
    "round_id",
    "campaign",
    "simulation_package_directory",
    "CG_MD_label",
    "review_confidence",
    "review_evidence_summary",
    "AP_SASA_legacy_200ns",
    "AP_contact_path_190_200ns_mean",
    "cluster_largest_fraction_200ns",
    "cluster_count_200ns",
    "cluster_singleton_fraction_200ns",
    "cluster_mean_contacts_200ns",
    "molecule_count",
    "largest_cluster_size_200ns",
    "singleton_count_200ns",
    "contacted_molecule_count_200ns",
    "hard_cutoff_AP_contact_fraction_200ns",
    "criterion_positive",
    "label_matches_criterion",
    "validation_notes",
    "source_AP_SASA_file",
    "source_path_AP_contact_file",
    "source_aggregate_summary_file",
    "trajectory_path",
    "source_review_file",
    "missing_fields",
]

PHASE4_COLUMNS = [
    "sequence",
    "canonical_sequence",
    "policy",
    "policy_rank",
    "simulation_status",
    "archived_simulation_status",
    "exclusion_reason",
    "campaign",
    "simulation_package_directory",
    "CG_MD_label",
    "review_confidence",
    "review_evidence_summary",
    "AP_SASA_legacy_200ns",
    "AP_contact_path_190_200ns_mean",
    "cluster_largest_fraction_200ns",
    "cluster_count_200ns",
    "cluster_singleton_fraction_200ns",
    "cluster_mean_contacts_200ns",
    "molecule_count",
    "largest_cluster_size_200ns",
    "singleton_count_200ns",
    "contacted_molecule_count_200ns",
    "hard_cutoff_AP_contact_fraction_200ns",
    "criterion_positive",
    "label_matches_criterion",
    "validation_notes",
    "source_AP_SASA_file",
    "source_path_AP_contact_file",
    "source_aggregate_summary_file",
    "trajectory_path",
    "source_review_file",
    "missing_fields",
]


FIELD_MAPPINGS = {
    "phase3": {
        "row_source": "thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/branches/{branch}/rounds/round_{001..008}/review/md_review.csv",
        "sequence": "md_review.csv: sequence",
        "canonical_sequence": "uppercase(sequence)",
        "acquisition_branch": "md_review.csv: source_branch, fallback branch_strategy",
        "round_id": "md_review.csv: round_id, fallback source_round_id",
        "campaign": "md_review.csv: campaign_name",
        "simulation_package_directory": "md_review.csv: metric_source_package_dir",
        "CG_MD_label": "md_review.csv: cgmd_label",
        "review_confidence": "md_review.csv: label_confidence",
        "review_evidence_summary": "md_review.csv: label_evidence_summary",
        "AP_SASA_legacy_200ns": "md_review.csv: AP_sasa",
        "AP_contact_path_190_200ns_mean": "md_review.csv: paper_path_APcontact_last10ns",
        "cluster_largest_fraction_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: largest_cluster_fraction",
        "cluster_count_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: cluster_count",
        "cluster_singleton_fraction_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: singleton_fraction",
        "cluster_mean_contacts_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: mean_contacts_per_molecule",
        "molecule_count": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: molecule_count",
        "largest_cluster_size_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: largest_cluster_size",
        "singleton_count_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: singleton_count",
        "contacted_molecule_count_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: contacted_molecules",
        "hard_cutoff_AP_contact_fraction_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: ap_contact",
        "criterion_positive": "calculated from AP_SASA_legacy_200ns >= 1.75 and AP_contact_path_190_200ns_mean >= 0.5",
        "label_matches_criterion": "calculated by comparing criterion_positive with archived CG_MD_label",
        "validation_notes": "calculated validation status",
        "source_AP_SASA_file": "{metric_source_package_dir}/{sequence}_sasa_AP_SASA.txt",
        "source_path_AP_contact_file": "{metric_source_package_dir}/{sequence}_paper_path_APcontact_last10ns.txt",
        "source_aggregate_summary_file": "{metric_source_package_dir}/{sequence}_aggregate_summary.csv",
        "trajectory_path": "first existing {metric_source_package_dir}/{sequence}_*_CG.xtc, sorted by name",
        "source_review_file": "path of contributing md_review.csv",
        "missing_fields": "semicolon-separated list of blank required evidence/provenance fields",
    },
    "phase4_primary": {
        "row_source": "thesis_results/THESIS_RESULTS_HANDOFF_20260623/04_phase4_bayesian_optimization/tables/primary_phase4_selected_peptides_with_cgmd.csv",
        "proposal_source": "thesis_results/THESIS_RESULTS_HANDOFF_20260623/04_phase4_bayesian_optimization/tables/primary_phase4_all_selected_peptides.csv",
        "sequence": "primary_phase4_selected_peptides_with_cgmd.csv: sequence",
        "canonical_sequence": "uppercase(sequence)",
        "policy": "primary_phase4_selected_peptides_with_cgmd.csv: policy",
        "policy_rank": "primary_phase4_selected_peptides_with_cgmd.csv: selection_rank",
        "simulation_status": "complete if archived complete; excluded_not_simulated for the 25-residue MES proposal",
        "archived_simulation_status": "primary_phase4_selected_peptides_with_cgmd.csv: simulation_status",
        "exclusion_reason": "derived for excluded 25-residue MES proposal; blank otherwise",
        "campaign": "primary_phase4_selected_peptides_with_cgmd.csv: source_campaign, fallback md_review.csv: campaign",
        "simulation_package_directory": "source_review_csv parent / md_review.csv: package_dir",
        "CG_MD_label": "primary_phase4_selected_peptides_with_cgmd.csv: operational_label",
        "review_confidence": "source md_review.csv: label_confidence when present",
        "review_evidence_summary": "source md_review.csv: label_evidence_summary when present; operational_rubric fallback",
        "AP_SASA_legacy_200ns": "source md_review.csv: ap_200ns, fallback primary_phase4_selected_peptides_with_cgmd.csv: ap_sasa_200ns",
        "AP_contact_path_190_200ns_mean": "source md_review.csv: paper_path_ap_contact_last10ns_mean, fallback primary_phase4_selected_peptides_with_cgmd.csv: paper_path_apcontact_last10ns",
        "cluster_largest_fraction_200ns": "source md_review.csv: cluster_largest_fraction_200ns, checked against aggregate-summary row when available",
        "cluster_count_200ns": "source md_review.csv: cluster_count_200ns, checked against aggregate-summary row when available",
        "cluster_singleton_fraction_200ns": "source md_review.csv: cluster_singleton_fraction_200ns, checked against aggregate-summary row when available",
        "cluster_mean_contacts_200ns": "source md_review.csv: cluster_mean_contacts_200ns, checked against aggregate-summary row when available",
        "molecule_count": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: molecule_count",
        "largest_cluster_size_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: largest_cluster_size",
        "singleton_count_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: singleton_count",
        "contacted_molecule_count_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: contacted_molecules",
        "hard_cutoff_AP_contact_fraction_200ns": "{sequence}_aggregate_summary.csv row frame_ns=200 cutoff_nm=0.60: ap_contact",
        "criterion_positive": "calculated for simulated rows from AP_SASA_legacy_200ns >= 1.75 and AP_contact_path_190_200ns_mean >= 0.5; blank for excluded row",
        "label_matches_criterion": "calculated by comparing criterion_positive with archived CG_MD_label; blank for excluded row",
        "validation_notes": "calculated validation/exclusion status",
        "source_AP_SASA_file": "source md_review.csv: ap_file resolved relative to review campaign root",
        "source_path_AP_contact_file": "source md_review.csv: paper_path_ap_contact_last10ns_file resolved relative to review campaign root",
        "source_aggregate_summary_file": "source md_review.csv: aggregate_summary_file resolved relative to review campaign root",
        "trajectory_path": "first existing package {sequence}_*_CG.xtc, sorted by name",
        "source_review_file": "primary_phase4_selected_peptides_with_cgmd.csv: source_review_csv",
        "missing_fields": "semicolon-separated list of blank required evidence/provenance fields",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(str(path))
    try:
        if p.is_absolute():
            return str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("/", "\\")
    except Exception:
        pass
    return str(p).replace("/", "\\")


def repo_path_from_text(value: str) -> Path | None:
    if not value:
        return None
    text = value.strip().replace("/", "\\")
    p = Path(text)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def canonical(sequence: str) -> str:
    return (sequence or "").strip().upper()


def parse_float(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(str(value))
        if math.isfinite(number):
            return number
    except Exception:
        return None
    return None


def parse_label(value: str) -> bool | None:
    if str(value).strip() == "1":
        return True
    if str(value).strip() == "0":
        return False
    return None


def criterion(ap_value: str, contact_value: str) -> bool | None:
    ap = parse_float(ap_value)
    contact = parse_float(contact_value)
    if ap is None or contact is None:
        return None
    return bool(ap >= AP_THRESHOLD and contact >= CONTACT_THRESHOLD)


def bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "TRUE" if value else "FALSE"


def label_match(label: str, calc: bool | None) -> bool | None:
    archived = parse_label(label)
    if archived is None or calc is None:
        return None
    return archived == calc


def missing_fields(row: dict[str, str], required: Iterable[str]) -> str:
    missing = [field for field in required if str(row.get(field, "")).strip() == ""]
    return ";".join(missing)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def source_files_for_package(package_dir: Path | None, sequence: str) -> dict[str, str]:
    if package_dir is None:
        return {
            "source_AP_SASA_file": "",
            "source_path_AP_contact_file": "",
            "source_aggregate_summary_file": "",
            "trajectory_path": "",
        }
    seq = canonical(sequence)
    ap_file = package_dir / f"{seq}_sasa_AP_SASA.txt"
    contact_file = package_dir / f"{seq}_paper_path_APcontact_last10ns.txt"
    aggregate_file = package_dir / f"{seq}_aggregate_summary.csv"
    trajectories = sorted(package_dir.glob(f"{seq}_*_CG.xtc"), key=lambda p: p.name)
    trajectory = first_existing(trajectories)
    return {
        "source_AP_SASA_file": rel(ap_file) if ap_file.exists() else "",
        "source_path_AP_contact_file": rel(contact_file) if contact_file.exists() else "",
        "source_aggregate_summary_file": rel(aggregate_file) if aggregate_file.exists() else "",
        "trajectory_path": rel(trajectory) if trajectory else "",
    }


def aggregate_diagnostics(aggregate_path: Path | None) -> dict[str, str]:
    empty = {
        "cluster_largest_fraction_200ns": "",
        "cluster_count_200ns": "",
        "cluster_singleton_fraction_200ns": "",
        "cluster_mean_contacts_200ns": "",
        "molecule_count": "",
        "largest_cluster_size_200ns": "",
        "singleton_count_200ns": "",
        "contacted_molecule_count_200ns": "",
        "hard_cutoff_AP_contact_fraction_200ns": "",
    }
    if aggregate_path is None or not aggregate_path.exists():
        return empty
    try:
        rows = read_csv(aggregate_path)
    except Exception:
        return empty
    for row in rows:
        if row.get("frame_ns") == str(AGGREGATE_FRAME_NS) and abs(float(row.get("cutoff_nm", "nan")) - AGGREGATE_CUTOFF_NM) < 1e-9:
            return {
                "cluster_largest_fraction_200ns": row.get("largest_cluster_fraction", ""),
                "cluster_count_200ns": row.get("cluster_count", ""),
                "cluster_singleton_fraction_200ns": row.get("singleton_fraction", ""),
                "cluster_mean_contacts_200ns": row.get("mean_contacts_per_molecule", ""),
                "molecule_count": row.get("molecule_count", ""),
                "largest_cluster_size_200ns": row.get("largest_cluster_size", ""),
                "singleton_count_200ns": row.get("singleton_count", ""),
                "contacted_molecule_count_200ns": row.get("contacted_molecules", ""),
                "hard_cutoff_AP_contact_fraction_200ns": row.get("ap_contact", ""),
            }
    return empty


def phase3_review_files() -> list[Path]:
    files = []
    for branch in BRANCHES:
        for round_id in range(1, 9):
            path = (
                PHASE3_ROOT
                / "branches"
                / branch
                / "rounds"
                / f"round_{round_id:03d}"
                / "review"
                / "md_review.csv"
            )
            files.append(path)
    return files


def build_phase3_rows() -> tuple[list[dict[str, str]], list[str]]:
    output: list[dict[str, str]] = []
    source_files: list[str] = []
    for review_file in phase3_review_files():
        source_files.append(rel(review_file))
        for src in read_csv(review_file):
            package_dir = repo_path_from_text(src.get("metric_source_package_dir", ""))
            file_paths = source_files_for_package(package_dir, src.get("sequence", ""))
            aggregate_path = repo_path_from_text(file_paths["source_aggregate_summary_file"])
            diagnostics = aggregate_diagnostics(aggregate_path)
            calc = criterion(src.get("AP_sasa", ""), src.get("paper_path_APcontact_last10ns", ""))
            match = label_match(src.get("cgmd_label", ""), calc)
            notes: list[str] = []
            if calc is None:
                notes.append("criterion_not_calculable")
            elif match is False:
                notes.append("archived_label_mismatch")
            else:
                notes.append("criterion_matches_archived_label")
            row = {
                "sequence": src.get("sequence", ""),
                "canonical_sequence": canonical(src.get("sequence", "")),
                "acquisition_branch": src.get("source_branch") or src.get("branch_strategy", ""),
                "round_id": src.get("round_id") or src.get("source_round_id", ""),
                "campaign": src.get("campaign_name", ""),
                "simulation_package_directory": rel(package_dir) if package_dir else "",
                "CG_MD_label": src.get("cgmd_label", ""),
                "review_confidence": src.get("label_confidence", ""),
                "review_evidence_summary": src.get("label_evidence_summary", ""),
                "AP_SASA_legacy_200ns": src.get("AP_sasa", ""),
                "AP_contact_path_190_200ns_mean": src.get("paper_path_APcontact_last10ns", ""),
                **diagnostics,
                "criterion_positive": bool_text(calc),
                "label_matches_criterion": bool_text(match),
                "validation_notes": ";".join(notes),
                **file_paths,
                "source_review_file": rel(review_file),
            }
            row["missing_fields"] = missing_fields(
                row,
                [
                    "sequence",
                    "canonical_sequence",
                    "acquisition_branch",
                    "round_id",
                    "campaign",
                    "simulation_package_directory",
                    "CG_MD_label",
                    "review_confidence",
                    "review_evidence_summary",
                    "AP_SASA_legacy_200ns",
                    "AP_contact_path_190_200ns_mean",
                    "source_AP_SASA_file",
                    "source_path_AP_contact_file",
                    "source_aggregate_summary_file",
                ],
            )
            output.append(row)
    return output, source_files


def load_phase4_review(review_path_text: str) -> tuple[dict[str, str], Path | None, Path | None]:
    if not review_path_text:
        return {}, None, None
    review_file = repo_path_from_text(review_path_text)
    if review_file is None or not review_file.exists():
        return {}, review_file, None
    rows = read_csv(review_file)
    review_row = rows[0] if rows else {}
    campaign_root = review_file.parent
    package_text = review_row.get("package_dir", "")
    package_dir = campaign_root / package_text if package_text else None
    return review_row, review_file, package_dir


def phase4_source_path(relative: str, campaign_root: Path | None) -> str:
    if not relative or campaign_root is None:
        return ""
    path = campaign_root / relative
    return rel(path) if path.exists() else ""


def build_phase4_rows() -> tuple[list[dict[str, str]], list[str]]:
    selected_path = PHASE4_HANDOFF / "primary_phase4_selected_peptides_with_cgmd.csv"
    proposal_path = PHASE4_HANDOFF / "primary_phase4_all_selected_peptides.csv"
    proposals = {
        (row["policy"], row["selection_rank"], row["sequence"]): row
        for row in read_csv(proposal_path)
    }
    source_files = [rel(selected_path), rel(proposal_path)]
    output: list[dict[str, str]] = []
    for src in read_csv(selected_path):
        key = (src.get("policy", ""), src.get("selection_rank", ""), src.get("sequence", ""))
        proposal = proposals.get(key, {})
        review_row, review_file, package_dir = load_phase4_review(src.get("source_review_csv", ""))
        if review_file:
            source_files.append(rel(review_file))
        campaign_root = review_file.parent if review_file else None
        file_paths = source_files_for_package(package_dir, src.get("sequence", ""))
        if review_row:
            file_paths["source_AP_SASA_file"] = phase4_source_path(review_row.get("ap_file", ""), campaign_root) or file_paths["source_AP_SASA_file"]
            file_paths["source_path_AP_contact_file"] = phase4_source_path(review_row.get("paper_path_ap_contact_last10ns_file", ""), campaign_root) or file_paths["source_path_AP_contact_file"]
            file_paths["source_aggregate_summary_file"] = phase4_source_path(review_row.get("aggregate_summary_file", ""), campaign_root) or file_paths["source_aggregate_summary_file"]
        aggregate_path = repo_path_from_text(file_paths["source_aggregate_summary_file"])
        diagnostics = aggregate_diagnostics(aggregate_path)
        for target, review_field in [
            ("cluster_largest_fraction_200ns", "cluster_largest_fraction_200ns"),
            ("cluster_count_200ns", "cluster_count_200ns"),
            ("cluster_singleton_fraction_200ns", "cluster_singleton_fraction_200ns"),
            ("cluster_mean_contacts_200ns", "cluster_mean_contacts_200ns"),
        ]:
            if not diagnostics.get(target) and review_row.get(review_field):
                diagnostics[target] = review_row.get(review_field, "")

        archived_status = src.get("simulation_status", "")
        status = archived_status
        exclusion_reason = ""
        if archived_status != "complete":
            status = "excluded_not_simulated"
            if src.get("sequence_length") == "25":
                exclusion_reason = "sequence_length_25_exceeds_supported_cgmd_length_24"
            else:
                exclusion_reason = "not_simulated_in_primary_phase4_archive"

        ap_value = review_row.get("ap_200ns") or src.get("ap_sasa_200ns", "")
        contact_value = review_row.get("paper_path_ap_contact_last10ns_mean") or src.get("paper_path_apcontact_last10ns", "")
        calc = criterion(ap_value, contact_value) if status == "complete" else None
        archived_label = src.get("operational_label", "")
        match = label_match(archived_label, calc) if status == "complete" else None
        notes: list[str] = []
        if status != "complete":
            notes.append("excluded_not_simulated")
        elif calc is None:
            notes.append("criterion_not_calculable")
        elif match is False:
            notes.append("archived_label_mismatch")
        else:
            notes.append("criterion_matches_archived_label")
        output_diagnostics = diagnostics if status == "complete" else {key: "" for key in diagnostics}
        output_file_paths = file_paths if status == "complete" else {
            "source_AP_SASA_file": "",
            "source_path_AP_contact_file": "",
            "source_aggregate_summary_file": "",
            "trajectory_path": "",
        }

        row = {
            "sequence": src.get("sequence", ""),
            "canonical_sequence": canonical(src.get("sequence", "")),
            "policy": src.get("policy", ""),
            "policy_rank": src.get("selection_rank", ""),
            "simulation_status": status,
            "archived_simulation_status": archived_status,
            "exclusion_reason": exclusion_reason,
            "campaign": src.get("source_campaign") or review_row.get("campaign", ""),
            "simulation_package_directory": rel(package_dir) if package_dir else "",
            "CG_MD_label": archived_label,
            "review_confidence": review_row.get("label_confidence", ""),
            "review_evidence_summary": review_row.get("label_evidence_summary") or src.get("operational_rubric", ""),
            "AP_SASA_legacy_200ns": ap_value if status == "complete" else "",
            "AP_contact_path_190_200ns_mean": contact_value if status == "complete" else "",
            **output_diagnostics,
            "criterion_positive": bool_text(calc),
            "label_matches_criterion": bool_text(match),
            "validation_notes": ";".join(notes),
            **output_file_paths,
            "source_review_file": rel(review_file) if review_file else "",
        }
        # Preserve proposal source by using it at least once in code path and validation.
        if proposal and proposal.get("sequence_length") == "25" and status == "excluded_not_simulated":
            row["exclusion_reason"] = "sequence_length_25_exceeds_supported_cgmd_length_24"
        required = [
            "sequence",
            "canonical_sequence",
            "policy",
            "policy_rank",
            "simulation_status",
        ]
        if status == "complete":
            required.extend(
                [
                    "campaign",
                    "simulation_package_directory",
                    "CG_MD_label",
                    "AP_SASA_legacy_200ns",
                    "AP_contact_path_190_200ns_mean",
                    "source_AP_SASA_file",
                    "source_path_AP_contact_file",
                    "source_aggregate_summary_file",
                ]
            )
        else:
            required.append("exclusion_reason")
        row["missing_fields"] = missing_fields(row, required)
        output.append(row)
    return output, sorted(set(source_files))


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt4(value: str) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def label_word(value: str) -> str:
    if str(value) == "1":
        return "positive"
    if str(value) == "0":
        return "negative"
    return ""


def append_latex_table(lines: list[str], caption: str, label: str, rows: list[dict[str, str]], columns: list[tuple[str, str, str]]) -> None:
    lines.extend(
        [
            r"\begin{longtable}{p{0.19\textwidth}p{0.09\textwidth}p{0.17\textwidth}p{0.12\textwidth}p{0.13\textwidth}p{0.11\textwidth}p{0.12\textwidth}}",
            rf"\caption{{{latex_escape(caption)}}}\label{{{label}}}\\",
            r"\toprule",
            " & ".join(latex_escape(header) for _field, header, _kind in columns) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            " & ".join(latex_escape(header) for _field, header, _kind in columns) + r" \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in rows:
        cells = []
        for field, _header, kind in columns:
            value = row.get(field, "")
            if kind == "num":
                cells.append(latex_escape(fmt4(value)))
            elif kind == "label":
                cells.append(latex_escape(label_word(value)))
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])


def write_appendix(phase3_rows: list[dict[str, str]], phase4_rows: list[dict[str, str]]) -> None:
    lines = [
        r"% Auto-generated by cgmd_peptide_evidence/generate_cgmd_peptide_evidence.py.",
        r"% Requires booktabs and longtable.",
        r"\section*{Per-peptide CG--MD evidence tables}",
        "",
        (
            "The operational binary criterion was "
            r"$\mathrm{AP\_SASA}_{200\,\mathrm{ns}}\geq 1.75$ and "
            r"$\mathrm{AP\ contact}_{190-200\,\mathrm{ns}}\geq 0.5$. "
            "Largest-cluster fraction, singleton fraction, cluster count, mean contacts, "
            "and visual inspection were supporting evidence and did not enter the binary "
            "operational conjunction. Phase 4-D candidates were not simulated and are not "
            "included as CG--MD result rows."
        ),
        "",
    ]
    phase3_columns = [
        ("sequence", "Sequence", "text"),
        ("round_id", "Round", "text"),
        ("acquisition_branch", "Branch", "text"),
        ("AP_SASA_legacy_200ns", "AP-SASA 200 ns", "num"),
        ("AP_contact_path_190_200ns_mean", "Path AP-contact 190--200 ns", "num"),
        ("CG_MD_label", "Label", "label"),
        ("review_confidence", "Confidence", "text"),
    ]
    for branch in BRANCHES:
        subset = [row for row in phase3_rows if row["acquisition_branch"] == branch]
        append_latex_table(
            lines,
            f"Phase 3 per-peptide CG--MD evidence for {BRANCH_DISPLAY[branch]}.",
            f"tab:phase3_cgmd_evidence_{branch}",
            subset,
            phase3_columns,
        )

    phase4_columns = [
        ("sequence", "Sequence", "text"),
        ("policy_rank", "Rank", "text"),
        ("policy", "Policy", "text"),
        ("AP_SASA_legacy_200ns", "AP-SASA 200 ns", "num"),
        ("AP_contact_path_190_200ns_mean", "Path AP-contact 190--200 ns", "num"),
        ("CG_MD_label", "Label", "label"),
        ("simulation_status", "Status", "text"),
    ]
    append_latex_table(
        lines,
        "Primary Phase 4 per-peptide CG--MD evidence.",
        "tab:phase4_primary_cgmd_evidence",
        phase4_rows,
        phase4_columns,
    )
    APPENDIX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_by(rows: list[dict[str, str]], *fields: str) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counter[tuple(row.get(field, "") for field in fields)] += 1
    return counter


def positive_count(rows: list[dict[str, str]], key_field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if row.get("CG_MD_label") == "1":
            counter[row.get(key_field, "")] += 1
    return counter


def validation_markdown(
    phase3_rows: list[dict[str, str]],
    phase4_rows: list[dict[str, str]],
    phase3_sources: list[str],
    phase4_sources: list[str],
) -> str:
    phase3_seq_counts = Counter(row["canonical_sequence"] for row in phase3_rows)
    phase3_dupes = {seq: count for seq, count in phase3_seq_counts.items() if count > 1}
    phase4_simulated = [row for row in phase4_rows if row["simulation_status"] == "complete"]
    phase4_excluded = [row for row in phase4_rows if row["simulation_status"] == "excluded_not_simulated"]
    phase3_mismatches = [row for row in phase3_rows if row["label_matches_criterion"] == "FALSE"]
    phase4_mismatches = [row for row in phase4_simulated if row["label_matches_criterion"] == "FALSE"]
    phase3_missing = [row for row in phase3_rows if row["missing_fields"]]
    phase4_missing = [row for row in phase4_rows if row["missing_fields"]]

    lines = [
        "# CG-MD Peptide Evidence Validation",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Operational criterion",
        "",
        "`criterion_positive = AP_SASA_legacy_200ns >= 1.75 AND AP_contact_path_190_200ns_mean >= 0.5`.",
        "",
        "The contact criterion uses the final-window field only: `paper_path_ap_contact_last10ns_mean` / `paper_path_APcontact_last10ns`.",
        "It does not use `ap_contact_200ns`, `paper_ap_contact_200ns`, `ap_contact_same_paper_formula_200ns`, or `paper_path_ap_contact_200ns`.",
        "",
        "## Source files",
        "",
        "### Phase 3 review sources",
        "",
    ]
    lines.extend(f"- `{src}`" for src in sorted(phase3_sources))
    lines.extend(["", "### Primary Phase 4 sources", ""])
    lines.extend(f"- `{src}`" for src in sorted(set(phase4_sources)))
    lines.extend(["", "## Field mappings", ""])
    for phase, mapping in FIELD_MAPPINGS.items():
        lines.append(f"### {phase}")
        lines.append("")
        lines.append("| Output column | Source / calculation |")
        lines.append("|---|---|")
        for column, source in mapping.items():
            lines.append(f"| `{column}` | {source} |")
        lines.append("")

    lines.extend(
        [
            "## Validation summary",
            "",
            f"- Phase 3 row count: `{len(phase3_rows)}`.",
            f"- Phase 3 total unique sequences: `{len(phase3_seq_counts)}`.",
            f"- Phase 3 duplicate sequences: `{len(phase3_dupes)}`.",
            f"- Phase 3 rows with missing required fields: `{len(phase3_missing)}`.",
            f"- Phase 3 label/criterion mismatches: `{len(phase3_mismatches)}`.",
            f"- Primary Phase 4 archived proposal count: `{len(phase4_rows)}`.",
            f"- Primary Phase 4 simulated count: `{len(phase4_simulated)}`.",
            f"- Primary Phase 4 excluded proposal count: `{len(phase4_excluded)}`.",
            f"- Primary Phase 4 positive count: `{sum(1 for row in phase4_simulated if row['CG_MD_label'] == '1')}`.",
            f"- Primary Phase 4 rows with missing required fields: `{len(phase4_missing)}`.",
            f"- Primary Phase 4 label/criterion mismatches: `{len(phase4_mismatches)}`.",
            "",
            "Phase 4-D candidates were not simulated and are intentionally not included in a CG-MD results table.",
            "",
            "## Phase 3 row count by branch",
            "",
            "| Branch | Rows | Positives |",
            "|---|---:|---:|",
        ]
    )
    branch_counts = count_by(phase3_rows, "acquisition_branch")
    branch_pos = positive_count(phase3_rows, "acquisition_branch")
    for branch in BRANCHES:
        lines.append(f"| `{branch}` | {branch_counts[(branch,)]} | {branch_pos[branch]} |")

    lines.extend(["", "## Phase 3 row count by branch and round", "", "| Branch | Round | Rows | Positives |", "|---|---:|---:|---:|"])
    branch_round_counts = count_by(phase3_rows, "acquisition_branch", "round_id")
    for branch in BRANCHES:
        for round_id in range(1, 9):
            key = (branch, str(round_id))
            pos = sum(1 for row in phase3_rows if row["acquisition_branch"] == branch and row["round_id"] == str(round_id) and row["CG_MD_label"] == "1")
            lines.append(f"| `{branch}` | {round_id} | {branch_round_counts[key]} | {pos} |")

    lines.extend(["", "## Primary Phase 4 count by policy", "", "| Policy | Rows | Simulated | Excluded | Positives |", "|---|---:|---:|---:|---:|"])
    for policy in POLICIES:
        rows = [row for row in phase4_rows if row["policy"] == policy]
        simulated = [row for row in rows if row["simulation_status"] == "complete"]
        excluded = [row for row in rows if row["simulation_status"] == "excluded_not_simulated"]
        positives = [row for row in simulated if row["CG_MD_label"] == "1"]
        lines.append(f"| `{policy}` | {len(rows)} | {len(simulated)} | {len(excluded)} | {len(positives)} |")

    lines.extend(["", "## Duplicates and mismatches", ""])
    if phase3_dupes:
        lines.append("Phase 3 duplicate canonical sequences:")
        lines.extend(f"- `{seq}`: {count}" for seq, count in sorted(phase3_dupes.items()))
    else:
        lines.append("No duplicate Phase 3 canonical sequences were found.")
    lines.append("")
    if phase3_mismatches or phase4_mismatches:
        lines.append("Criterion/archive label mismatches:")
        for row in phase3_mismatches:
            lines.append(f"- Phase 3 `{row['sequence']}` branch `{row['acquisition_branch']}` round `{row['round_id']}`.")
        for row in phase4_mismatches:
            lines.append(f"- Phase 4 `{row['sequence']}` policy `{row['policy']}` rank `{row['policy_rank']}`.")
    else:
        lines.append("No disagreement between calculated criterion and archived labels was found.")
    lines.append("")
    if phase3_missing or phase4_missing:
        lines.append("Rows with missing required fields:")
        for row in phase3_missing[:50]:
            lines.append(f"- Phase 3 `{row['sequence']}`: `{row['missing_fields']}`")
        for row in phase4_missing[:50]:
            lines.append(f"- Phase 4 `{row['sequence']}`: `{row['missing_fields']}`")
        if len(phase3_missing) + len(phase4_missing) > 100:
            lines.append("- Additional missing-field rows omitted from this preview; see CSV `missing_fields` column.")
    else:
        lines.append("No required-field gaps were found, except intentionally blank metric/provenance fields for excluded-not-simulated proposals.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    phase3_rows, phase3_sources = build_phase3_rows()
    phase4_rows, phase4_sources = build_phase4_rows()

    write_csv(PHASE3_OUT, phase3_rows, PHASE3_COLUMNS)
    write_csv(PHASE4_OUT, phase4_rows, PHASE4_COLUMNS)
    MAPPING_OUT.write_text(json.dumps(FIELD_MAPPINGS, indent=2), encoding="utf-8")
    VALIDATION_OUT.write_text(
        validation_markdown(phase3_rows, phase4_rows, phase3_sources, phase4_sources),
        encoding="utf-8",
    )
    write_appendix(phase3_rows, phase4_rows)

    summary = {
        "phase3_rows": len(phase3_rows),
        "phase3_unique_sequences": len({row["canonical_sequence"] for row in phase3_rows}),
        "phase4_rows": len(phase4_rows),
        "phase4_simulated_rows": sum(1 for row in phase4_rows if row["simulation_status"] == "complete"),
        "phase4_excluded_rows": sum(1 for row in phase4_rows if row["simulation_status"] == "excluded_not_simulated"),
        "outputs": {
            "phase3": rel(PHASE3_OUT),
            "phase4": rel(PHASE4_OUT),
            "validation": rel(VALIDATION_OUT),
            "appendix": rel(APPENDIX_OUT),
            "mappings": rel(MAPPING_OUT),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
