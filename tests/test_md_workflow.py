from __future__ import annotations

import csv
from pathlib import Path
import shutil
import unittest
import tempfile
import uuid
import warnings

from active_learning_thesis import md_workflow
from active_learning_thesis.config import RunConfig
from active_learning_thesis.md_workflow import (
    MDP_FILENAMES,
    PROFILE_CHOICES,
    build_pdbs,
    make_md_ingest_csv,
    parse_bura_md_benchmark,
    parse_md_results,
    operational_cgmd_threshold_label,
    prepare_bura_md_benchmark,
    prepare_md_campaign,
)


class MdWorkflowTests(unittest.TestCase):
    def test_operational_cgmd_threshold_uses_path_contact_not_contact_fraction(self):
        self.assertEqual(operational_cgmd_threshold_label(1.75, 0.5), 1)
        self.assertEqual(operational_cgmd_threshold_label(1.749999, 0.9), 0)
        self.assertEqual(operational_cgmd_threshold_label(2.0, 0.499999), 0)

    def test_paper_path_contact_weight_and_exact_path_score(self):
        self.assertEqual(md_workflow._ap_contact_same_paper_formula_weight(0.4), 1.0)
        self.assertEqual(md_workflow._ap_contact_same_paper_formula_weight(1.2), 0.0)
        self.assertAlmostEqual(
            md_workflow._ap_contact_same_paper_formula_weight(0.5),
            0.36787944117144233,
        )
        weights = [
            [0.0, 1.0, 0.1],
            [1.0, 0.0, 0.8],
            [0.1, 0.8, 0.0],
        ]
        score, method = md_workflow._max_hamiltonian_path_mean_weight(weights)
        self.assertEqual(method, "exact_dynamic_programming")
        self.assertAlmostEqual(score, 0.9)

    def _scratch_dir(self, name: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"{name}_{uuid.uuid4().hex}_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _write_batch_csv(self, path: Path) -> None:
        rows = [
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.91",
                "pred_std": "0.08",
                "pred_entropy": "0.31",
                "pred_mutual_information": "0.05",
                "acquisition_score": "0.72",
            },
            {
                "sequence": "CCCCC",
                "round_id": "1",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.84",
                "pred_std": "0.06",
                "pred_entropy": "0.28",
                "pred_mutual_information": "0.04",
                "acquisition_score": "0.63",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _read_manifest(self, campaign_dir: Path) -> list[dict[str, str]]:
        with (campaign_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _nsteps(self, path: Path) -> str:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("nsteps"):
                return line.split("=", 1)[1].split(";", 1)[0].strip()
        self.fail(f"No nsteps line found in {path}")

    def test_prepare_md_campaign_allows_duplicate_sequence_replicates(self):
        temp_dir = self._scratch_dir("md_duplicate_campaign")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_duplicate_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        rows = [
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "acquisition_strategy": "manual_replicate",
                "pred_mean": "0.91",
                "pred_std": "0.08",
                "pred_entropy": "0.31",
                "pred_mutual_information": "0.05",
                "acquisition_score": "0.72",
            },
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "acquisition_strategy": "manual_replicate",
                "pred_mean": "0.91",
                "pred_std": "0.08",
                "pred_entropy": "0.31",
                "pred_mutual_information": "0.05",
                "acquisition_score": "0.72",
            },
        ]
        with batch_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            campaign_dir = prepare_md_campaign(run_dir, batch_csv, "duplicate_replicates", md_profile="line_smoke")

        self.assertTrue(any("Duplicate peptide sequence" in str(item.message) for item in caught))
        manifest_rows = self._read_manifest(campaign_dir)
        self.assertEqual([row["sequence"] for row in manifest_rows], ["AAAAA", "AAAAA"])
        self.assertEqual(
            [row["package_dir"].replace("\\", "/") for row in manifest_rows],
            ["packages/AAAAA", "packages/AAAAA__rep02"],
        )
        self.assertTrue((campaign_dir / "packages" / "AAAAA").exists())
        self.assertTrue((campaign_dir / "packages" / "AAAAA__rep02").exists())

        (campaign_dir / "PDBs" / "AAAAA.pdb").write_text("ATOM\n", encoding="utf-8")
        build_pdbs(campaign_dir, validate_only=True)
        self.assertTrue((campaign_dir / "packages" / "AAAAA" / "AAAAA.pdb").exists())
        self.assertTrue((campaign_dir / "packages" / "AAAAA__rep02" / "AAAAA.pdb").exists())

        submit_text = (campaign_dir / "submit_chain.sh").read_text(encoding="utf-8")
        self.assertIn('ALL_PACKAGES=("AAAAA" "AAAAA__rep02")', submit_text)
        self.assertIn('ALL_SEQUENCES=("AAAAA" "AAAAA")', submit_text)
        self.assertIn('Submitting canonical chain for $peptide ($package_name)', submit_text)

        (campaign_dir / "packages" / "AAAAA__rep02" / "AAAAA_CG_150_CG.gro").write_text("gro", encoding="utf-8")
        review_path = parse_md_results(campaign_dir)
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            parsed_rows = list(csv.DictReader(handle))
        self.assertEqual(len(parsed_rows), 2)
        by_package = {row["package_dir"]: row for row in parsed_rows}
        self.assertEqual(by_package[manifest_rows[0]["package_dir"]]["job_root_status"], "package_prepared")
        self.assertEqual(by_package[manifest_rows[1]["package_dir"]]["job_root_status"], "dynamics_complete")

    def test_prepare_md_campaign_creates_profile_specific_chain(self):
        temp_dir = self._scratch_dir("md_campaign")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_campaign_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)

        campaigns = {
            profile: prepare_md_campaign(run_dir, batch_csv, f"campaign_{profile}", md_profile=profile)
            for profile in PROFILE_CHOICES
        }

        for profile, campaign_dir in campaigns.items():
            self.assertTrue((campaign_dir / "manifest.csv").exists())
            self.assertTrue((campaign_dir / "sequences.txt").exists())
            self.assertTrue((campaign_dir / "submit_chain.sh").exists())
            self.assertTrue((campaign_dir / "preflight_bura.sh").exists())
            self.assertTrue((campaign_dir / "README_BURA.md").exists())
            manifest_rows = self._read_manifest(campaign_dir)
            self.assertEqual({row["md_profile"] for row in manifest_rows}, {profile})
            for sequence in ("AAAAA", "CCCCC"):
                package_dir = campaign_dir / "packages" / sequence
                self.assertTrue(package_dir.exists())
                self.assertFalse((package_dir / "13_Array_APcontact.sh").exists())
                prep_text = (package_dir / "0_CG_pol_sysprep.sh").read_text(encoding="utf-8")
                self.assertIn("SLURM_SUBMIT_DIR", prep_text)
                self.assertNotIn("#SBATCH --exclude=", prep_text)
                self.assertIn('basename "$SCRIPT_DIR"', (package_dir / "common.sh").read_text(encoding="utf-8"))

        for profile in ("line_smoke", "production_smoke"):
            campaign_dir = campaigns[profile]
            submit_text = (campaign_dir / "submit_chain.sh").read_text(encoding="utf-8")
            preflight_text = (campaign_dir / "preflight_bura.sh").read_text(encoding="utf-8")
            readme_text = (campaign_dir / "README_BURA.md").read_text(encoding="utf-8")
            for sequence in ("AAAAA", "CCCCC"):
                package_dir = campaign_dir / "packages" / sequence
                self.assertFalse((package_dir / "11_SASA_and_FrameDump.sh").exists())
                self.assertFalse((package_dir / "12_AP_calc.sh").exists())
                self.assertFalse((package_dir / "13_Extract_last10ns_for_paper_APcontact.sh").exists())
            self.assertNotIn("11_SASA_and_FrameDump.sh", submit_text)
            self.assertNotIn("12_AP_calc.sh", submit_text)
            self.assertNotIn("13_Extract_last10ns_for_paper_APcontact.sh", submit_text)
            self.assertIn("--exclude NODELIST", submit_text)
            self.assertIn('DEFAULT_EXCLUDE_NODES=""', submit_text)
            self.assertIn('EXCLUDE_ARGS=()', submit_text)
            self.assertIn('EXCLUDE_ARGS=(--exclude "$DEFAULT_EXCLUDE_NODES,$1")', submit_text)
            self.assertIn("set -eo pipefail", submit_text)
            self.assertNotIn("set -euo pipefail", submit_text)
            self.assertIn('sbatch "${EXCLUDE_ARGS[@]}" --parsable', submit_text)
            self.assertNotIn("11_SASA_and_FrameDump.sh", preflight_text)
            self.assertNotIn("12_AP_calc.sh", preflight_text)
            self.assertNotIn("13_Extract_last10ns_for_paper_APcontact.sh", preflight_text)
            self.assertIn('find . -type f -name "*.sh" -exec dos2unix', preflight_text)
            self.assertIn('module load gromacs/2023.2_g13.1_p3.10.5', preflight_text)
            self.assertIn('stops after `10_Dynamics_b.sh`', readme_text)

        full_campaign = campaigns["full"]
        full_submit = (full_campaign / "submit_chain.sh").read_text(encoding="utf-8")
        full_preflight = (full_campaign / "preflight_bura.sh").read_text(encoding="utf-8")
        full_readme = (full_campaign / "README_BURA.md").read_text(encoding="utf-8")
        for sequence in ("AAAAA", "CCCCC"):
            package_dir = full_campaign / "packages" / sequence
            self.assertTrue((package_dir / "11_SASA_and_FrameDump.sh").exists())
            self.assertTrue((package_dir / "12_AP_calc.sh").exists())
            self.assertTrue((package_dir / "13_Extract_last10ns_for_paper_APcontact.sh").exists())
            env_text = (package_dir / "env_bura.sh").read_text(encoding="utf-8")
            self.assertTrue(env_text.startswith("#!/bin/bash -l\n"))
            self.assertLess(env_text.index("source /etc/profile"), env_text.index("set -euo pipefail"))
            self.assertIn("module load gromacs/2023.2_g13.1_p3.10.5", env_text)
            self.assertIn("/home/.OPT/GROMACS/gromacs-2023.2-gcc13.1-p3.10.5/bin", env_text)
            self.assertIn("/opt/intel/impi/2021.17.1/mpi/2021.17/bin", env_text)
            self.assertIn("BURA_RUNTIME_LIB_DIRS", env_text)
            self.assertIn("LD_LIBRARY_PATH", env_text)
            self.assertIn("/home/.OPT/GCC/gcc-11.5.0/lib64", env_text)
            self.assertIn("/home/.OPT/IntelMKL/2025.3.0.462/mkl/2025.3/lib", env_text)
            self.assertIn("/opt/intel/oneapi/mkl/2025.3/lib", env_text)
            self.assertIn("command -v gmx_mpi", env_text)
            common_text = (package_dir / "common.sh").read_text(encoding="utf-8")
            self.assertTrue(common_text.startswith("#!/bin/bash -l\n"))
            self.assertLess(common_text.index('source "$SCRIPT_DIR/env_bura.sh"'), common_text.index("set -euo pipefail"))
            self.assertIn(
                "Missing SASA time points for AP calculation",
                (package_dir / "12_AP_calc.sh").read_text(encoding="utf-8"),
            )
            last10_script = (package_dir / "13_Extract_last10ns_for_paper_APcontact.sh").read_text(encoding="utf-8")
            self.assertIn("paper_path_last10ns_frames", last10_script)
            self.assertIn("190 191 192 193 194 195 196 197 198 199 200", last10_script)
            self.assertIn("frames_extracted", last10_script)
            self.assertIn('"sequence": "${peptide_name}"', last10_script)
            self.assertIn('"frames_dir": "${OUT_DIR}"', last10_script)
            self.assertNotIn('"sequence": "\'"', last10_script)
            full_dynamics_script = (package_dir / "10_Dynamics_b.sh").read_text(encoding="utf-8")
            self.assertTrue(full_dynamics_script.startswith("#!/bin/bash -l\n"))
            self.assertLess(
                full_dynamics_script.index('source "$SCRIPT_DIR/common.sh"'),
                full_dynamics_script.index("set -euo pipefail"),
            )
            self.assertIn("#SBATCH --nodes=6", full_dynamics_script)
            self.assertIn("#SBATCH --ntasks-per-node=8", full_dynamics_script)
            self.assertIn("#SBATCH --cpus-per-task=6", full_dynamics_script)
            self.assertIn('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"', full_dynamics_script)
            self.assertIn('mpirun -np "${SLURM_NTASKS:-1}" "$GMX_CMD" mdrun', full_dynamics_script)
            full_analysis_script = (package_dir / "11_SASA_and_FrameDump.sh").read_text(encoding="utf-8")
            self.assertIn("contact_frames=(0 5 12 25 50 100 200)", full_analysis_script)
            self.assertIn('PEPTIDE_GROUP="Peptide_non_solvent"', full_analysis_script)
            self.assertIn('printf "%s\\n" "$PEPTIDE_GROUP" | "$GMX_CMD" sasa', full_analysis_script)
            self.assertIn(
                'printf "inserted_initial_noncontact\\n" > "${peptide_name}_paper_initial_sasa_source.txt"',
                full_analysis_script,
            )
            self.assertNotIn('echo -e "1" | "$GMX_CMD" sasa', full_analysis_script)
            prep_script = (package_dir / "0_CG_pol_sysprep.sh").read_text(encoding="utf-8")
            self.assertIn('cp box.gro "${peptide_name}_inserted_initial_noncontact.gro"', prep_script)
            production_smoke_script = (
                campaigns["production_smoke"] / "packages" / sequence / "10_Dynamics_b.sh"
            ).read_text(encoding="utf-8")
            self.assertNotIn("mpirun -np", production_smoke_script)
        self.assertIn("11_SASA_and_FrameDump.sh", full_submit)
        self.assertIn("12_AP_calc.sh", full_submit)
        self.assertIn("13_Extract_last10ns_for_paper_APcontact.sh", full_submit)
        self.assertIn("--exclude NODELIST", full_submit)
        self.assertIn('DEFAULT_EXCLUDE_NODES=""', full_submit)
        self.assertIn('EXCLUDE_ARGS=()', full_submit)
        self.assertIn('EXCLUDE_ARGS=(--exclude "$DEFAULT_EXCLUDE_NODES,$1")', full_submit)
        self.assertIn('sbatch "${EXCLUDE_ARGS[@]}" --parsable', full_submit)
        self.assertIn("11_SASA_and_FrameDump.sh", full_preflight)
        self.assertIn("12_AP_calc.sh", full_preflight)
        self.assertIn("13_Extract_last10ns_for_paper_APcontact.sh", full_preflight)
        self.assertIn('find . -type f -name "*.sh" -exec dos2unix', full_readme)
        self.assertIn('module load gromacs/2023.2_g13.1_p3.10.5', full_readme)
        self.assertIn('1. `line_smoke`', full_readme)
        self.assertIn('2. `production_smoke`', full_readme)
        self.assertIn('3. `full`', full_readme)
        self.assertIn('200 ns production run required for AP targets through 200 ns', full_readme)
        self.assertIn('continues through post-analysis', full_readme)

    def test_prepare_md_campaign_rewrites_nsteps_per_profile(self):
        temp_dir = self._scratch_dir("md_profiles")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_profile_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)

        campaigns = {
            profile: prepare_md_campaign(run_dir, batch_csv, f"campaign_{profile}", md_profile=profile)
            for profile in PROFILE_CHOICES
        }
        sequence = "AAAAA"

        line_package = campaigns["line_smoke"] / "packages" / sequence
        for filename in MDP_FILENAMES:
            self.assertEqual(self._nsteps(line_package / filename), "200")

        production_package = campaigns["production_smoke"] / "packages" / sequence
        self.assertEqual(self._nsteps(production_package / "martini_22P_md.mdp"), "200")
        self.assertEqual(self._nsteps(production_package / "martini_22P_vacuum_mini.mdp"), "50000")
        self.assertEqual(self._nsteps(production_package / "martini_22P_mini_soft.mdp"), "20000")
        self.assertEqual(self._nsteps(production_package / "martini_22P_mini_steep.mdp"), "50000")
        self.assertEqual(self._nsteps(production_package / "martini_22P_equi_Berendsen.mdp"), "1500")
        self.assertEqual(self._nsteps(production_package / "martini_22P_equi_ParrRah.mdp"), "500")

        full_package = campaigns["full"] / "packages" / sequence
        self.assertEqual(self._nsteps(full_package / "martini_22P_md.mdp"), "10000000")
        self.assertEqual(self._nsteps(full_package / "martini_22P_vacuum_mini.mdp"), "50000")

    def test_parse_md_results_marks_smoke_runs_complete_at_dynamics(self):
        temp_dir = self._scratch_dir("md_smoke_parse")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_smoke_parse_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)
        campaign_dir = prepare_md_campaign(run_dir, batch_csv, "campaign_smoke", md_profile="production_smoke")

        for sequence in ("AAAAA", "CCCCC"):
            (campaign_dir / "PDBs" / f"{sequence}.pdb").write_text("ATOM\n", encoding="utf-8")

        pdb_dir = build_pdbs(campaign_dir, validate_only=True)
        self.assertEqual(pdb_dir, campaign_dir / "PDBs")
        self.assertTrue((campaign_dir / "packages" / "AAAAA" / "AAAAA.pdb").exists())

        package_dir = campaign_dir / "packages" / "AAAAA"
        nmol = 1200 // len("AAAAA")
        (package_dir / "AAAAA_CG_150_CG.gro").write_text("gro", encoding="utf-8")

        review_path = parse_md_results(campaign_dir)
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        first = next(row for row in rows if row["sequence"] == "AAAAA")
        self.assertEqual(first["job_root_status"], "dynamics_complete")
        self.assertEqual(first["md_profile"], "production_smoke")
        self.assertEqual(first["sasa_file"], "")
        self.assertEqual(first["ap_file"], "")
        self.assertEqual(first["ap_5ns"], "")
        second = next(row for row in rows if row["sequence"] == "CCCCC")
        self.assertEqual(second["job_root_status"], "package_prepared")
        self.assertEqual(second["md_profile"], "production_smoke")

    def test_parse_full_results_with_empty_ap_file_stays_sasa_complete(self):
        temp_dir = self._scratch_dir("md_full_sasa_only")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_full_sasa_only_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)
        campaign_dir = prepare_md_campaign(run_dir, batch_csv, "campaign_full_sasa_only", md_profile="full")

        for sequence in ("AAAAA", "CCCCC"):
            (campaign_dir / "PDBs" / f"{sequence}.pdb").write_text("ATOM\n", encoding="utf-8")

        build_pdbs(campaign_dir, validate_only=True)

        package_dir = campaign_dir / "packages" / "AAAAA"
        (package_dir / "AAAAA_sasa.xvg").write_text(
            "@ title \"SASA\"\n0.000 10.0\n",
            encoding="utf-8",
        )
        (package_dir / "AAAAA_AP_SASA.txt").write_text("", encoding="utf-8")
        (package_dir / "AAAAA_CG_150_CG.xtc").write_text("xtc", encoding="utf-8")

        review_path = parse_md_results(campaign_dir)
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        first = next(row for row in rows if row["sequence"] == "AAAAA")
        self.assertEqual(first["job_root_status"], "sasa_complete")
        self.assertTrue(first["ap_file"].endswith("AAAAA_AP_SASA.txt"))

    def test_parse_full_results_and_make_ingest(self):
        temp_dir = self._scratch_dir("md_full_parse")
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name="md_full_parse_test", output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)
        campaign_dir = prepare_md_campaign(run_dir, batch_csv, "campaign_full", md_profile="full")

        for sequence in ("AAAAA", "CCCCC"):
            (campaign_dir / "PDBs" / f"{sequence}.pdb").write_text("ATOM\n", encoding="utf-8")

        pdb_dir = build_pdbs(campaign_dir, validate_only=True)
        self.assertEqual(pdb_dir, campaign_dir / "PDBs")
        self.assertTrue((campaign_dir / "packages" / "AAAAA" / "AAAAA.pdb").exists())

        package_dir = campaign_dir / "packages" / "AAAAA"
        (package_dir / "AAAAA_sasa.xvg").write_text(
            "@ title \"SASA\"\n"
            "0.000 10.0\n5.000 5.0\n12.000 4.0\n25.000 2.0\n50.000 1.0\n100.000 0.5\n"
            "190.000 2.0\n195.000 1.0\n200.000 0.5\n",
            encoding="utf-8",
        )
        (package_dir / "AAAAA_paper_initial_sasa.xvg").write_text(
            "@ title \"Initial SASA\"\n0.000 12.0\n",
            encoding="utf-8",
        )
        (package_dir / "AAAAA_paper_initial_sasa_source.txt").write_text(
            "inserted_initial_noncontact\n",
            encoding="utf-8",
        )
        (package_dir / "AAAAA_AP_SASA.txt").write_text(
            "The AP for 5 ns is: 2.0\nThe AP for 12 ns is: 2.5\n",
            encoding="utf-8",
        )
        gro_lines = ["Synthetic contact frame", "20"]
        atom_id = 1
        molecule_origins = [(1.0, 1.0, 1.0), (1.3, 1.0, 1.0), (5.0, 5.0, 5.0), (8.0, 8.0, 8.0)]
        for origin in molecule_origins:
            for residue_id in range(1, 6):
                gro_lines.append(
                    f"{residue_id:5d}{'ALA':<5}{'BB':>5}{atom_id:5d}"
                    f"{origin[0] + residue_id * 0.01:8.3f}{origin[1]:8.3f}{origin[2]:8.3f}"
                )
                atom_id += 1
        gro_lines.append("  10.00000  10.00000  10.00000")
        frame_text = "\n".join(gro_lines) + "\n"
        (package_dir / "AAAAA_100ns.gro").write_text(frame_text, encoding="utf-8")
        (package_dir / "AAAAA_200ns.gro").write_text(frame_text, encoding="utf-8")
        last10_dir = package_dir / "AAAAA_paper_path_last10ns_frames"
        last10_dir.mkdir()
        for frame_ns in range(190, 201):
            (last10_dir / f"AAAAA_{frame_ns}ns.gro").write_text(frame_text, encoding="utf-8")
        nmol = 1200 // len("AAAAA")
        (package_dir / "AAAAA_CG_150_CG.xtc").write_text("xtc", encoding="utf-8")
        (package_dir / "AAAAA_CG_150_CG.tpr").write_text("tpr", encoding="utf-8")

        review_path = parse_md_results(campaign_dir)
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        first = next(row for row in rows if row["sequence"] == "AAAAA")
        self.assertEqual(first["job_root_status"], "analysis_complete")
        self.assertEqual(first["ap_5ns"], "2.0")
        self.assertEqual(first["ap_12ns"], "2.5")
        self.assertEqual(first["paper_ap_sasa_status"], "computed")
        self.assertEqual(first["paper_ap_sasa_initial_source"], "inserted_initial_noncontact")
        self.assertEqual(first["paper_ap_sasa_initial_sasa"], "12.000000")
        self.assertEqual(first["paper_ap_sasa_final10_mean_sasa"], "1.166667")
        self.assertEqual(first["paper_ap_sasa_last10ns_n_frames"], "3")
        self.assertTrue(first["paper_ap_sasa_last10ns_file"].endswith("AAAAA_paper_APSASA_last10ns.txt"))
        self.assertTrue(first["paper_ap_sasa_recompute_script"].endswith("AAAAA_recompute_paper_APSASA.sh"))
        self.assertTrue(first["paper_ap_sasa_status_file"].endswith("AAAAA_paper_APSASA_last10ns_status.json"))
        self.assertEqual(first["paper_ap_sasa_group_selection"], "Peptide_non_solvent")
        self.assertEqual(first["ap_contact_100ns"], "0.500000")
        self.assertEqual(first["ap_contact_200ns"], "0.500000")
        self.assertTrue(first["ap_contact_file"].endswith("AAAAA_AP_contact.txt"))
        self.assertTrue(first["paper_ap_contact_file"].endswith("AAAAA_paper_APcontact.txt"))
        self.assertTrue(0.0 < float(first["paper_ap_contact_200ns"]) < float(first["ap_contact_200ns"]))
        self.assertTrue(first["paper_path_ap_contact_last10ns_file"].endswith("AAAAA_paper_path_APcontact_last10ns.txt"))
        self.assertTrue(first["paper_path_ap_contact_last10ns_script"].endswith("AAAAA_extract_last10ns_for_paper_APcontact.sh"))
        self.assertEqual(first["paper_path_ap_contact_last10ns_status"], "computed")
        self.assertEqual(first["paper_path_ap_contact_last10ns_n_frames"], "11")
        self.assertTrue(0.0 < float(first["paper_path_ap_contact_last10ns_mean"]) <= 1.0)
        self.assertTrue(first["aggregate_summary_file"].endswith("AAAAA_aggregate_summary.csv"))
        self.assertEqual(first["cluster_largest_fraction_200ns"], "0.500000")
        self.assertEqual(first["cluster_count_200ns"], "3")
        self.assertEqual(first["cluster_singleton_fraction_200ns"], "0.500000")
        self.assertEqual(first["cluster_mean_contacts_200ns"], "0.500000")
        self.assertEqual(first["md_profile"], "full")
        paper_contact_path = campaign_dir / first["paper_ap_contact_file"]
        paper_contact_path.write_text(
            "# stale definition\n"
            "# weight(distance_nm)=1/(1+exp(12*(distance_nm-0.6))).\n"
            "The paper_APcontact for 200 ns is: 0.001000 "
            "(peptide_pairs=6; midpoint_nm=0.6; steepness=12; source=AAAAA_200ns.gro)\n",
            encoding="utf-8",
        )
        refreshed_review_path = parse_md_results(campaign_dir)
        with refreshed_review_path.open("r", encoding="utf-8", newline="") as handle:
            refreshed_rows = list(csv.DictReader(handle))
        refreshed_first = next(row for row in refreshed_rows if row["sequence"] == "AAAAA")
        self.assertNotEqual(refreshed_first["paper_ap_contact_200ns"], "0.001000")
        self.assertIn("midpoint_nm=0.4425", (campaign_dir / refreshed_first["paper_ap_contact_file"]).read_text(encoding="utf-8"))
        second = next(row for row in rows if row["sequence"] == "CCCCC")
        self.assertEqual(second["job_root_status"], "package_prepared")
        self.assertEqual(second["md_profile"], "full")

        for row in rows:
            row["cgmd_label"] = "1" if row["sequence"] == "AAAAA" else "0"
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        ingest_path = make_md_ingest_csv(campaign_dir, review_path)
        with ingest_path.open("r", encoding="utf-8", newline="") as handle:
            ingest_rows = list(csv.DictReader(handle))
        self.assertEqual(list(ingest_rows[0].keys()), ["sequence", "round_id", "cgmd_label"])
        self.assertEqual(len(ingest_rows), 2)
        self.assertEqual(ingest_rows[0]["round_id"], "1")

    def test_prepare_and_parse_bura_md_benchmark(self):
        temp_dir = self._scratch_dir("md_bura_benchmark")
        campaign_dir = temp_dir / "campaign"
        source_package = campaign_dir / "packages" / "AAAAA"
        source_package.mkdir(parents=True)
        for filename, content in {
            "common.sh": "#!/bin/bash\nsource \"$SCRIPT_DIR/env_bura.sh\"\n",
            "env_bura.sh": "#!/bin/bash\nexport GMX_CMD=${GMX_CMD:-gmx_mpi}\n",
            "topol.top": "#include \"protein.itp\"\n",
            "protein.itp": "[ moleculetype ]\n",
            "index.ndx": "[ System ]\n",
            "AAAAA.pdb": "ATOM\n",
            "equi2.gro": "gro\n",
            "equi2.cpt": "checkpoint\n",
            "martini_22P_md.mdp": "integrator = md\nnsteps = 10000000\n",
        }.items():
            (source_package / filename).write_text(content, encoding="utf-8")

        benchmark_dir = prepare_bura_md_benchmark(
            campaign_dir,
            "AAAAA",
            benchmark_name="perf_test",
            nsteps=50000,
            layouts=["1n_1mpi_48omp"],
        )

        layout_dir = benchmark_dir / "1n_1mpi_48omp"
        self.assertTrue((layout_dir / "run_benchmark.sh").exists())
        self.assertIn("mpirun -np", (layout_dir / "run_benchmark.sh").read_text(encoding="utf-8"))
        self.assertIn("-s benchmark.tpr", (layout_dir / "run_benchmark.sh").read_text(encoding="utf-8"))
        self.assertEqual(self._nsteps(layout_dir / "martini_22P_md_benchmark.mdp"), "50000")
        self.assertFalse((layout_dir / "benchmark.xtc").exists())

        (layout_dir / "benchmark.log").write_text(
            """
Using 1 MPI process
Using 48 OpenMP threads
               Core t (s)   Wall t (s)        (%)
       Time:     4800.000      100.000     4800.0
                 (ns/day)    (hour/ns)
Performance:      864.000        0.028
Finished mdrun on rank 0 Sun May  3 12:00:00 2026
""",
            encoding="utf-8",
        )

        results_path = parse_bura_md_benchmark(benchmark_dir)
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["layout"], "1n_1mpi_48omp")
        self.assertEqual(rows[0]["status"], "finished")
        self.assertEqual(rows[0]["md_runtime_wall_hms"], "01:40")
        self.assertEqual(rows[0]["md_runtime_ns_per_day"], "864.000")
        self.assertEqual(rows[0]["observed_mpi_processes"], "1")
        self.assertEqual(rows[0]["observed_omp_threads"], "48")


if __name__ == "__main__":
    unittest.main()

