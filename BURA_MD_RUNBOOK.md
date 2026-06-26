# BURA MD Runbook

This file is the practical runbook for using this repository on **BURA** for
the **MD side only**.

Use **Supek** for:
- model training
- replay benchmarking
- active-learning proposal jobs
- discovery jobs

Use **BURA** for:
- coarse-grained MD validation only

The validated workflow for this project is:
- prepare the MD campaign **locally**
- build PDBs **locally**
- upload only the generated campaign folder to BURA
- use **Slurm** on BURA (`sbatch`), not PBS (`qsub`)

## 1. MD Mode Overview

The supported MD profiles are:
- `line_smoke`
- `production_smoke`
- `full`

Recommended order:
1. one peptide, `line_smoke`
2. one peptide, `production_smoke`
3. one peptide, `full`

Current behavior:
- `line_smoke`
  - all generated `.mdp` files use `nsteps = 200`
  - generated Slurm jobs are small smoke-test jobs
  - the automated chain stops after `10_Dynamics_b.sh`
  - `md_review.csv` should normally end at `job_root_status=dynamics_complete`
- `production_smoke`
  - only production `.mdp` uses `nsteps = 200`
  - generated Slurm jobs are moderate-size test jobs
  - the automated chain stops after `10_Dynamics_b.sh`
  - `md_review.csv` should normally end at `job_root_status=dynamics_complete`
- `full`
  - original `.mdp` lengths are preserved
  - the bundled production `.mdp` now runs for 200 ns so SASA/AP targets through 200 ns are reachable
  - generated Slurm jobs use the heavy production resource settings
  - the automated chain continues through SASA/AP post-analysis (`11` and `12`)
  - the primary Phase 3 contact field is
    `paper_path_APcontact_last10ns`: a Hamiltonian-path score using the paper
    piecewise distance weight, averaged across the 11 frames from 190 through
    200 ns. Exact dynamic programming is used for at most 16 peptide copies;
    larger systems use deterministic beam search.
  - the retained Phase 3 operational label rule is
    `AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`.
    A human still reviews and writes `cgmd_label`; parsing does not silently
    assign it.
  - local finalization also computes `AP_contact`, the fraction of peptide
    molecules with at least one inter-peptide bead contact within 0.6 nm. This
    is a separate diagnostic, not the primary label-generating contact metric.
  - local finalization writes an aggregate summary CSV with cutoff sensitivity,
    largest-cluster fraction, cluster count, singleton fraction, and mean
    contacts per peptide so saturated contact-fraction values are not
    over-interpreted.

### 1.1 Frozen full-profile parameter contract

The thesis-ready, source-checked simulation table is maintained in
`THESIS_CGMD_PARAMETER_CONTRACT.md`. In compact form, the full profile uses:

- GROMACS 2023.2 with Martini 2.2P and refined polarizable `PW` water;
- vacuum, soft-core solvated, and final steep minimisation with
  `emtol=20`, `100`, and `10 kJ mol^-1 nm^-1`;
- restrained 9 ps v-rescale/Berendsen equilibration and restrained 12.5 ps
  Nose-Hoover/Parrinello-Rahman equilibration;
- 200 ns production at 303 K and 1 bar with `dt=0.020 ps`;
- reaction-field electrostatics and shifted Coulomb/van der Waals cutoffs of
  1.1 nm;
- compressed trajectory, energy, and log output every 1 ns.

Important: production `constraints=none` does not disable explicit constraints
already present in the Martini peptide and polarizable-water topologies. Those
constraints remain active and are handled by LINCS.

## 2. One-Time Local Setup

### 2.0 Local Windows OpenSSH + VPN setup for dashboard use

If you want to drive BURA from the local dashboard, use native Windows OpenSSH and connect FortiClient first when required:

```powershell
Get-Service ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
ssh-add -l
ssh bura "hostname"
```

If the non-interactive SSH check fails, connect FortiClient VPN and try again. The older Pageant-based helper workflow is deprecated and unsupported by the dashboard.

### 2.1 Use the project conda environment

Open **elevated PowerShell** on Windows if your local drive/path requires it.

```powershell
conda activate ml_peptide_self_assembly
```

### 2.2 Verify PyMOL automation support

This environment is now expected to contain `pymol-open-source`.

```powershell
python -c "import pymol; print('pymol ok')"
python -c "import pymol2; print('pymol2 ok')"
python -m active_learning_thesis --help
```

Expected:
- `pymol ok`
- `pymol2 ok`
- thesis CLI help is shown

## 3. What To Do Every Time You Start A Local MD Session

```powershell
cd "<local_workspace>"
conda activate ml_peptide_self_assembly
```

Optional quick sanity checks:

```powershell
python -c "import pymol2; print('pymol2 ok')"
python -m active_learning_thesis --help
```

## 4. Prepare A Fresh BURA Campaign Locally

Recommended guided command for the day-to-day single-peptide flow:

```powershell
python -m active_learning_thesis prepare-md-stage `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --batch-csv .\active_learning_runs\supek_smoke_init_local\round_001_batch.csv `
  --sequence MFMMMMVVI `
  --campaign first_bura_line_smoke `
  --md-profile line_smoke `
  --cluster bura `
  --exclude-nodes bura201
```

This guided command filters the selected peptide, prepares the one-peptide campaign, builds or reuses the local PDB, and writes `NEXT_BURA_COMMANDS.md` with the exact BURA commands to copy.

The lower-level `prepare-md-campaign` and `build-pdbs` commands below still work and remain supported.

### 4.1 Stage the run configuration and batch CSV locally

At minimum, `prepare-md-campaign` needs:
- a local run directory containing `config.json`
- the batch CSV you want to export

Example staging layout:

```text
active_learning_runs/
  supek_smoke_init_local/
    config.json
    round_001_batch.csv
```

### 4.2 Prepare a campaign

Low-level manual alternative:

```powershell
python -m active_learning_thesis prepare-md-campaign `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --batch-csv .\active_learning_runs\supek_smoke_init_local\round_001_batch.csv `
  --campaign first_bura_line_smoke `
  --cluster bura `
  --md-profile line_smoke
```

This creates:

```text
active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke\
```

### 4.3 Build PDBs locally

```powershell
python -m active_learning_thesis build-pdbs `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

Expected outputs:
- `PDBs\<PEPTIDE>.pdb`
- `packages\<PEPTIDE>\<PEPTIDE>.pdb`

### 4.4 If you already have a PDB manually

Place it into the campaign `PDBs\` folder and then run:

```powershell
python -m active_learning_thesis build-pdbs `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke `
  --validate-only
```

## 5. What To Upload To BURA

Upload **only** the generated campaign folder, for example:

```text
<local_workspace>\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

Do **not** upload the whole repo to BURA for routine MD execution.

## 6. What To Do Every Time You Log Into BURA

Go into the uploaded campaign directory first:

```bash
cd ~/first_bura_line_smoke
```

Then run the standard staging commands:

```bash
find . -type f -name "*.sh" -exec dos2unix {} \+
find . -type f -name "*.sh" -exec chmod u+x {} \+
module load gromacs/2023.2_g13.1_p3.10.5
```

Then run preflight:

```bash
bash ./preflight_bura.sh
```

Important:
- BURA uses **Slurm**
- submit with `sbatch` indirectly through `submit_chain.sh`
- do **not** use `qsub`

## 7. BURA Rules For This Project

On BURA login/access nodes:
- it is okay to upload/stage the prepared campaign
- it is okay to run `dos2unix`
- it is okay to run `chmod` on uploaded shell scripts
- it is okay to inspect modules
- it is okay to run `preflight_bura.sh`
- it is okay to submit jobs through `submit_chain.sh`

On BURA login/access nodes:
- do **not** run `gmx` directly
- do **not** run `martinize.py` directly
- do **not** run `triple-w.py` directly
- do **not** run long Python jobs directly

Heavy MD work must run only through Slurm jobs.

## 8. Submit A Single Peptide

For the first test, submit **one peptide only**:

```bash
bash ./submit_chain.sh MFMMMMVVI
```

If a specific BURA node is misbehaving, you can exclude it explicitly:

```bash
bash ./submit_chain.sh --exclude bura201 MFMMMMVVI
```

You can later submit multiple named peptides or `--all`, but that is **not**
recommended for the first smoke validation.

## 9. Monitor BURA Jobs

Check your jobs:

```bash
squeue -u $USER
```

Check one job in detail:

```bash
scontrol show job JOBID
```

If Slurm gives a predicted start time:

```bash
squeue --start -j JOBID
```

Check node/cluster health:

```bash
sinfo
sinfo -R | head -n 50
```

Cancel jobs if needed:

```bash
scancel JOBID
```

## 10. After The BURA Run Finishes

Recommended guided command:

```powershell
python -m active_learning_thesis finalize-md-stage `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

This reparses the campaign, updates `md_review.csv`, and prints the next recommended ladder step.

The lower-level `parse-md-results` command below still works and remains supported.

Bring the campaign folder back locally and parse it:

```powershell
python -m active_learning_thesis parse-md-results `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

This updates:
- `md_review.csv`

For smoke profiles, empty `sasa_file`, `ap_file`, and AP columns are expected.

Then manually review the campaign and set:
- `review_notes`
- `cgmd_label`

Then convert to ingest format:

```powershell
python -m active_learning_thesis make-md-ingest-csv `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke `
  --review-csv .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke\md_review.csv
```

This generates:
- `cgmd_ingest.csv`

Then import the reviewed labels into the thesis workflow:

```powershell
python -m active_learning_thesis ingest-round `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --import-csv .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke\cgmd_ingest.csv
```

## 11. Useful Commands In One Place

### Local Windows

Prepare a campaign:

```powershell
python -m active_learning_thesis prepare-md-campaign `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --batch-csv .\active_learning_runs\supek_smoke_init_local\round_001_batch.csv `
  --campaign first_bura_line_smoke `
  --cluster bura `
  --md-profile line_smoke
```

Build PDBs:

```powershell
python -m active_learning_thesis build-pdbs `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

Parse results:

```powershell
python -m active_learning_thesis parse-md-results `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke
```

Make ingest CSV:

```powershell
python -m active_learning_thesis make-md-ingest-csv `
  --campaign-dir .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke `
  --review-csv .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke\md_review.csv
```

Import labels:

```powershell
python -m active_learning_thesis ingest-round `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --import-csv .\active_learning_runs\supek_smoke_init_local\md_campaigns\first_bura_line_smoke\cgmd_ingest.csv
```

### BURA

Normalize uploaded scripts:

```bash
find . -type f -name "*.sh" -exec dos2unix {} \+
find . -type f -name "*.sh" -exec chmod u+x {} \+
```

Preflight:

```bash
module load gromacs/2023.2_g13.1_p3.10.5
bash ./preflight_bura.sh
```

Submit:

```bash
bash ./submit_chain.sh MFMMMMVVI
bash ./submit_chain.sh --exclude bura201 MFMMMMVVI
```

Monitor:

```bash
squeue -u $USER
scontrol show job JOBID
sinfo -R | head -n 50
```

Cancel:

```bash
scancel JOBID
```

## 12. Known Troubleshooting Cases

### 12.1 `Permission denied` when running `./preflight_bura.sh`

This is usually Unix execute-bit loss after upload from Windows.

Use:

```bash
find . -type f -name "*.sh" -exec chmod u+x {} \+
bash ./preflight_bura.sh
```

### 12.2 `Required module unavailable: python/Python-2.7.18`

Newly generated campaigns now auto-detect Python 2 more gracefully.

If you are using an **older already-uploaded campaign**, remove the stale hardcoded line:

```bash
sed -i '/python\/Python-2.7.18/d' preflight_bura.sh
sed -i '/python\/Python-2.7.18/d' packages/MFMMMMVVI/env_bura.sh
```

Then rerun:

```bash
bash ./preflight_bura.sh
```

### 12.3 `launch failed requeued held`

This often means the job failed to launch on the cluster rather than failing
inside your MD script.

Check:

```bash
scontrol show job JOBID
sinfo -R | head -n 50
```

If the root job is unhealthy and dependencies are blocked behind it, the cleanest
fix is usually:
- cancel the whole chain
- regenerate a fresh campaign locally
- re-upload and resubmit

### 12.4 Smoke mode still looks too heavy

Freshly generated campaigns now have smaller Slurm requests for:
- `line_smoke`
- `production_smoke`

Freshly generated smoke campaigns also stop at `10_Dynamics_b.sh`, so they do not depend on SASA/AP outputs.

If an older uploaded campaign still requests very large node counts or walltimes,
regenerate it locally from the current code before retrying.

## 13. Current Recommendation

For BURA validation:
1. one peptide, `line_smoke`
2. one peptide, `production_smoke`
3. one peptide, `full`

Keep Supek for ML and BURA for MD.


## 12. Guided Ladder Status

To see where one peptide is in the guided ladder, run locally:

```powershell
python -m active_learning_thesis md-ladder-status `
  --run-dir .\active_learning_runs\supek_smoke_init_local `
  --sequence MFMMMMVVI
```

This reports:
- every guided campaign found for that peptide
- each campaign's parsed terminal status
- the next recommended profile in the ladder
- whether the peptide is ready for human review / ingest prep
