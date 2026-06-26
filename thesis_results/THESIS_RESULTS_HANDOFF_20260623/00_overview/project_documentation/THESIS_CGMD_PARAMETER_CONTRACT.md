# Thesis CG-MD Parameter Contract

## Scope and provenance

This document records the exact full-profile coarse-grained molecular-dynamics
contract used for the BURA peptide simulations. It is intended to replace
placeholder text in the thesis CG-MD parameter table.

The values were checked against:

- the canonical templates under
  `MD/CG_sims_BURA/CG_sims_BURA/`;
- generated full-profile packages under
  `active_learning_runs/thesis_main_supek_clean_20260502_original/md_campaigns/`;
- generated `0_CG_pol_sysprep.sh` through
  `13_Extract_last10ns_for_paper_APcontact.sh`;
- generated `topol.top` and `Protein_A.itp`;
- completed production `.tpr` settings printed in `*_CG.log`.

All 150 completed production logs found in the canonical campaign tree report
GROMACS 2023.2 and the same realized production timing and output intervals.
The six full-profile `.mdp` templates were identical across 173 of 174
packages; the single alternate copy belongs to a smoke-profile package and is
not part of the full production contract.

## Thesis-ready table

| Protocol element | Exact implemented value |
|---|---|
| Software version | GROMACS 2023.2, mixed precision, MPI build with OpenMP enabled. Completed production used `gmx_mpi`; the recorded BURA runtime used Intel MPI 2021.17.1. |
| Force field and water model | Peptides were converted by `martinize.py` 2.6 with `-ff martini22p`. The topology included the refined Martini 2.2 polarizable force field file `martini_v2.2refP.itp`; solvent residue `PW` is the three-site polarizable water model (`W`, `WP`, `WM`). |
| Minimisation threshold | Three steepest-descent stages were used: vacuum minimisation, `emtol=20 kJ mol^-1 nm^-1`, at most 50,000 steps; solvated soft-core minimisation, `emtol=100 kJ mol^-1 nm^-1`, at most 20,000 steps; final solvated steep minimisation, `emtol=10 kJ mol^-1 nm^-1`, at most 50,000 steps. A stage stopped when its force criterion was reached or its step limit was exhausted. |
| Equilibration stages and durations | Restrained Berendsen stage: `dt=0.006 ps`, 1,500 steps, total 9 ps. Restrained Parrinello-Rahman stage: `dt=0.025 ps`, 500 steps, total 12.5 ps. Production: `dt=0.020 ps`, 10,000,000 steps, total 200 ns. |
| Thermostat and coupling constant | Berendsen equilibration used stochastic velocity rescaling (`v-rescale`) at 303 K with separate `Protein` and `PW` groups and `tau_t=1.0 ps` for both. The second equilibration used Nose-Hoover at 303 K with the same groups and `tau_t=1.0 ps`. Production used `v-rescale`, 303 K, `Protein` and `PW`, `tau_t=1.0 ps`; the realized thermostat update interval was 10 steps (0.2 ps). |
| Barostat, coupling constant, and compressibility | First equilibration: isotropic Berendsen, `ref_p=1.0 bar`, `tau_p=5.0 ps`, compressibility `4.5e-5 bar^-1`. Second equilibration: semi-isotropic Parrinello-Rahman, `ref_p=1.0 1.0 bar`, `tau_p=12.0 ps`, compressibility `3e-4 3e-4 bar^-1`. Production: isotropic Parrinello-Rahman, `ref_p=1.0 bar`, `tau_p=12.0 ps`, compressibility `4.5e-5 bar^-1`; the realized pressure-coupling interval was 25 steps (0.5 ps). |
| Constraints | The production `.mdp` used `constraints=none`, meaning GROMACS did not convert additional bonds into constraints. Explicit constraints already present in the Martini peptide and polarizable-water topologies remained active and were solved with LINCS. The realized production settings were LINCS order 4, one iteration, and warning angle 30 degrees. The final steep-minimisation template specified LINCS order 8 and two iterations. Backbone position restraints with `POSRES_FC=4000 kJ mol^-1 nm^-2` were active during solvated minimisation and both equilibration stages, but not during production. |
| Neighbour-list update rule | Production used the Verlet cutoff scheme, grid neighbour searching, periodic boundaries in `xyz`, `nstlist=20` (0.4 ps), and `verlet-buffer-tolerance=0.005 kJ mol^-1 ps^-1`. GROMACS selected the realized list radius automatically; a representative completed production `.tpr` reported `rlist=1.376 nm`. Vacuum minimisation used `nstlist=1`; the other solvated stages used `nstlist=20`. |
| Electrostatic settings | Reaction-field electrostatics with `rcoulomb=1.1 nm`, relative dielectric constant `epsilon_r=2.5`, and input `epsilon_rf=0`, realized by GROMACS as an infinite reaction-field dielectric. The realized Coulomb modifier was potential shift. The initial vacuum minimisation was the exception, using `epsilon_r=15`. |
| Van der Waals settings | Cutoff van der Waals interactions with potential shifting, `rvdw=1.1 nm`, and no long-range dispersion correction (`DispCorr=No`). |
| Trajectory-output intervals | Production wrote uncompressed coordinates, velocities, and forces every 500,000 steps = 10 ns (`nstxout=nstvout=nstfout=500000`). The compressed trajectory was written every 50,000 steps = 1 ns (`nstxout-compressed=50000`) with `compressed-x-precision=50000`. The log was written every 50,000 steps = 1 ns. No compressed output group restriction was set, so the system was written. |
| Energy-output intervals | The realized production `.tpr` calculated energies every 100 steps = 2 ps (`nstcalcenergy=100`) and wrote energies every 50,000 steps = 1 ns (`nstenergy=50000`). |
| Terminal and protonation treatment | `martinize.py` reported charged chain termini. No explicit pH or alternative protonation option was passed, so standard `martinize.py` 2.6 Martini residue assignments were retained. Side-chain and terminal charges were taken from the generated topology. |
| Secondary-structure assumptions | Every residue was supplied to `martinize.py` as extended (`-ss EEEE...`, one `E` per residue). The generated topology therefore used the Martini extended-state backbone parameters and local short- and long-range elastic bonds for extended regions. Secondary structure was fixed by this input rather than predicted separately for each peptide. |

## Additional implemented system setup

- The number of peptide copies was
  `floor(1200 / sequence_length)`.
- Coarse-grained copies were inserted into a cubic
  `20 x 20 x 20 nm^3` box.
- The initial inserted structure was retained before solvation as the preferred
  non-contact reference for paper-style SASA analysis.
- Solvation used `water.gro`, a solute-solvent exclusion radius of `0.21 nm`,
  and `triple-w.py` to convert the solvent representation to polarizable `PW`.
- One 200 ns trajectory was run per selected peptide; independent MD replicas
  were not part of the operational protocol.

## Interpretation cautions

1. Do not write that the simulation had "no constraints." The `.mdp` did not
   request conversion of extra bonds, but explicit topology constraints were
   active and the completed production run reported them.
2. Do not describe `compressed-x-precision=50000` as an interval. It is the
   compressed-coordinate precision parameter; the interval is
   `nstxout-compressed=50000`.
3. The 9 ps and 12.5 ps equilibration stages are short and position-restrained.
   Report them exactly rather than describing an unspecified long
   equilibration.
4. These parameters describe the full production profile, not the line-smoke or
   production-smoke profiles, whose step counts are intentionally shortened.
