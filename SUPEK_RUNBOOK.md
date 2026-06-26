# Supek ML Runbook

For current thesis status, canonical completed result paths, and the reduced
Phase 5 design, first read
[`THESIS_HANDOFF_FOR_NEXT_MODEL.md`](THESIS_HANDOFF_FOR_NEXT_MODEL.md).
This runbook is operational guidance and is not the authority for whether an
experiment has completed.

This file is the practical runbook for using this repository on **Supek** for
the **ML side only**.

Use **Supek** for:
- model training
- replay benchmarking
- active-learning proposal jobs
- discovery jobs

Use **BURA** for:
- coarse-grained MD validation only

Do **not** use PyCharm Remote Development as the primary workflow on Supek.
The validated setup for this project is:
- plain SSH terminal access
- Git clone on Supek
- Miniforge + Conda environment in your home directory
- PBS jobs for actual execution

## 1. One-Time Supek Setup

### 1.0 Local Windows OpenSSH setup for dashboard use

If you want to drive Supek from the local dashboard, use native Windows OpenSSH on your workstation:

```powershell
Get-Service ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
ssh-add -l
ssh supek "hostname"
```

The dashboard assumes `ssh-agent`, `ssh-add`, and `~/.ssh/config` aliases. The older Pageant-based helper workflow is deprecated and unsupported by the dashboard.

### 1.1 Clone the repo

```bash
mkdir -p ~/projects/ml_peptide_self_assembly
cd ~/projects/ml_peptide_self_assembly
git clone https://github.com/<your-user>/<repo>.git
cd Master-s-thesis---ML-Peptide-Self-Assembly
git checkout codex/active-learning-thesis
```

### 1.2 Create directories for logs and run outputs

Keep:
- code in your home/project area
- large run outputs in scratch

```bash
mkdir -p ~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly/supek_logs
mkdir -p /lustre/scratch/$USER/ml_peptide_self_assembly_runs
```

### 1.3 Install Miniforge in your home directory

If `conda` is not already available:

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda --version
```

### 1.4 Create the project environment

```bash
cd ~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly
source ~/miniforge3/etc/profile.d/conda.sh
conda env create -f ml_peptide_self_assembly.yml
```

### 1.5 Verify the environment lightly on the login node

Do only lightweight checks on the login node.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -c "import sys, site; print('user site:', site.getusersitepackages()); print('has_local:', any('/.local/' in p for p in sys.path))"
python -c "import seqprops, tensorflow as tf; print('seqprops ok'); print(tf.__version__)"
python -m active_learning_thesis --help
```

Expected:
- `has_local: False`
- `seqprops ok`
- TensorFlow imports successfully
- the thesis CLI help is shown

## 2. What To Do Every Time You Log Into Supek

Run this at the start of every new shell session:

```bash
cd ~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Optional quick sanity check:

```bash
python -m active_learning_thesis --help
```

## 3. Cluster Rules For This Project

On Supek login nodes:
- it is okay to clone the repo
- it is okay to create/activate the Conda environment
- it is okay to do import/help checks
- it is okay to create PBS job scripts
- it is okay to submit jobs with `qsub`

On Supek login nodes:
- do **not** run model training directly
- do **not** run replay directly
- do **not** run long discovery/proposal jobs directly

Actual ML work must run through PBS jobs.

## 4. Validated PBS Job Pattern

All validated Supek jobs for this repo follow this structure:

```bash
#!/bin/bash
#PBS -N example_name
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=40GB
#PBS -l walltime=01:00:00
#PBS -o /lustre/home/$USER/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly/supek_logs/example.out
#PBS -e /lustre/home/$USER/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly/supek_logs/example.err

cd $PBS_O_WORKDIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m active_learning_thesis ...
```

Notes:
- `#PBS -P ...` was **not** needed in the validated smoke runs because the
  account used the default project `_pbs_project_default`
- if PBS later rejects a job because of account/project rules, add the correct
  project code then

## 5. Validated Smoke Workflow

The following Supek jobs were successfully validated:

### 5.1 Initialize a run

Command used inside PBS:

```bash
python -m active_learning_thesis init-run \
  --run-name supek_smoke_init \
  --output-root /lustre/scratch/$USER/ml_peptide_self_assembly_runs \
  --epochs 1 \
  --batch-size 3 \
  --candidate-pool-min 20 \
  --max-rounds 1
```

### 5.2 Propose the first AL batch

```bash
python -m active_learning_thesis propose-round \
  --run-dir /lustre/scratch/$USER/ml_peptide_self_assembly_runs/supek_smoke_init
```

### 5.3 Run discovery

This invokes the legacy/generic `run-discovery` support path. It is separate
from the thesis Phase 4 fixed-surrogate comparison and must not be used to
generate Phase 4 thesis evidence.

```bash
python -m active_learning_thesis run-discovery \
  --run-dir /lustre/scratch/$USER/ml_peptide_self_assembly_runs/supek_smoke_init
```

### 5.4 Run replay smoke

```bash
python -m active_learning_thesis run-replay \
  --run-dir /lustre/scratch/$USER/ml_peptide_self_assembly_runs/supek_smoke_init \
  --strategies random ensemble_mi
```

## 6. Where Outputs Go

Code and PBS scripts:
- `~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly`

Logs:
- `~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly/supek_logs`

Run outputs:
- `/lustre/scratch/$USER/ml_peptide_self_assembly_runs`

Example validated run:
- `/lustre/scratch/$USER/ml_peptide_self_assembly_runs/supek_smoke_init`

## 7. Useful Commands

Check jobs:

```bash
qstat -u $USER
```

Detailed job info:

```bash
qstat -fxw JOBID
```

Delete a queued/running duplicate job:

```bash
qdel JOBID
```

Inspect logs:

```bash
cat supek_logs/<jobname>.out
cat supek_logs/<jobname>.err
```

Inspect run outputs:

```bash
ls /lustre/scratch/$USER/ml_peptide_self_assembly_runs
ls /lustre/scratch/$USER/ml_peptide_self_assembly_runs/supek_smoke_init
```

## 8. Known Non-Blocking Warnings

The following warnings appeared in successful Supek runs and were not blocking:
- TensorRT libraries such as `libnvinfer.so.7` missing
- `ptxas` not found
- TensorFlow retracing warnings

These warnings did **not** prevent:
- GPU detection
- `init-run`
- `propose-round`
- `run-discovery`
- `run-replay` smoke

## 9. What To Do If You Change Profiles Or Start Fresh

If you log in from another machine/profile, repeat:

```bash
cd ~/projects/ml_peptide_self_assembly/Master-s-thesis---ML-Peptide-Self-Assembly
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

If the repo is missing, reclone it.

If the environment is missing, recreate it from:

```bash
conda env create -f ml_peptide_self_assembly.yml
```

If you are unsure whether the environment is healthy, rerun:

```bash
python -c "import seqprops, tensorflow as tf; print('seqprops ok'); print(tf.__version__)"
python -m active_learning_thesis --help
```

## 10. Current Recommendation

Supek setup is validated for the ML side. Phases 1-5 now have completed
scientific artefacts. Phase 5 completed all 12 replay jobs and aggregation; its
canonical tables and figures are under
`thesis_results/05_self_paced_active_learning/`.

The remaining work is thesis synthesis and the manual Phase 4/4-D simulation
slate decision. BURA remains the MD platform, and no simulation batch should
be inferred automatically from the Phase 4 comparisons.
