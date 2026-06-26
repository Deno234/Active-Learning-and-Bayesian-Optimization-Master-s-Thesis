import os
import shutil

#################################################################################################################
# USE: Write the names of the PDB files you generated and placed into the PDBs folder in the sequences.txt. Run this code.
# Each of the files in the master folder this code is in will be copied to the daughter folders as-is, just modified
# to have the peptide names in the job name for easier recognition
# To run it simply upload the files, run GROMACS module, add the master folder path to the execute_sims.sh and run the script
# MAKE SURE NOT TO HAVE ANY SPACES IN THE sequences.txt FILE
#################################################################################################################
# Path to the directory containing the PDB files
pdb_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PDBs')

# Replace 'sequences.txt' with the path to your text file
with open('sequences.txt', 'r') as f:
    peptides = f.read().splitlines()

# Get the current working directory
cwd = os.getcwd()

# Get a list of all files in the current working directory
files = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

for peptide in peptides:
    # Create the peptide directory if it doesn't already exist
    os.makedirs(peptide, exist_ok=True)
    # Check if a PDB file with the same name as the peptide exists in the PDB directory
    pdb_file = os.path.join(pdb_dir, peptide + '.pdb')
    if os.path.isfile(pdb_file):
        # If it does, copy it to the peptide directory
        shutil.copy2(pdb_file, os.path.join(cwd, peptide, peptide + '.pdb'))

    # Copy all files (excluding directories) to the peptide directory
    for file in files:
        if file != 'PDBs':  # Exclude the 'PDBs' directory
            shutil.copy2(file, os.path.join(peptide, file))

with open('sequences.txt', 'r') as f:
    peptides = f.read().splitlines()

for peptide in peptides:
    # Get the path to the peptide directory
    peptide_dir = os.path.join(os.getcwd(), peptide)

    # Get a list of all .sh files in the peptide directory
    sh_files = [f for f in os.listdir(peptide_dir) if f.endswith('.sh')]

    for sh_file in sh_files:
        # Get the path to the .sh file
        sh_file_path = os.path.join(peptide_dir, sh_file)

        # Read the contents of the .sh file
        with open(sh_file_path, 'r') as f:
            contents = f.read()

        # Replace the job name with the name of the peptide
        contents = contents.replace('#SBATCH --job-name=', '#SBATCH --job-name={}'.format(peptide))

        # Write the modified contents back to the .sh file
        with open(sh_file_path, 'w') as f:
            f.write(contents)