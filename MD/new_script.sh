#!/bin/bash
#SBATCH --job-name=AKCPQPtest
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20 
#SBATCH --cpus-per-task=2
#SBATCH --time=320:00:00

files=()

# Iterate over all files with the .pdb extension in the current directory
for file in *.pdb; do
  # Check if the file name matches the specific regex
  if [[ $file =~ ^[A-Z]+\.pdb$ ]]; then
    filename="${file%.pdb}"
    files+=("$filename")
    # Exit the loop after the first match
    break
  fi
done

export peptide_name="${files[@]}"
letter_count=${#peptide_name}
export nmol=$((1200 / letter_count))

python2 martinize.py -f ${peptide_name}.pdb -x ${peptide_name}_CG.pdb -o topol.top -p backbone -ff martini22p -ss EEEEEEEEEE
sed -i -e 's/martini\.itp/martini_v2.2refP.itp/' topol.top

# Insert molecules into the box
gmx_mpi insert-molecules -ci ${peptide_name}_CG.pdb -nmol ${nmol} -box 20 20 20 -o box.gro
sed -i -e "s/1/${nmol}/" topol.top

#Minimize in softcore only the peptides
gmx_mpi grompp -f martini_22P_vacuum_mini.mdp -c box.gro -p topol.top -o box.tpr -r box.gro -maxwarn 2
gmx_mpi mdrun -v -s box.tpr -deffnm box

gmx_mpi solvate -cp box.gro -cs water.gro -radius 0.21 -p topol.top -o solvbox.gro

# Change the water to polarizable water:
python triple-w.py solvbox.gro
sed -i -e "s/${nmol}W/${nmol}\nPW/" topol.top

echo -e "name 13 PW\nq" | gmx_mpi make_ndx -f solvbox_PW.gro -o index.ndx

export GMX_MAXCONSTRWARN=-1

#Softcore minimisation
gmx_mpi grompp -f martini_22P_mini_soft.mdp -c solvbox_PW.gro -p topol.top -o em0.tpr -r solvbox_PW.gro -n index.ndx -maxwarn 2
gmx_mpi mdrun -v -deffnm em0 -s em0.tpr

#Steepest descent minimisation
gmx_mpi grompp -f martini_22P_mini_steep.mdp -c em0.gro -p topol.top -o em1.tpr -r em0.gro -n index.ndx -maxwarn 2
gmx_mpi mdrun -v -deffnm em1

#Production
gmx_mpi grompp -f martini_22P_md.mdp -c em1.gro -r em1.gro -p topol.top -o ${peptide_name}_${nmol}_CG.tpr
gmx_mpi mdrun -v -deffnm ${peptide_name}_${nmol}_CG

### SASA ###
echo -e "1" | gmx sasa -f ${peptide_name}_${nmol}_CG.xtc -s ${peptide_name}_${nmol}_CG.tpr -n index_image.ndx -tu ns -o ${peptide_name}_sasa.xvg

