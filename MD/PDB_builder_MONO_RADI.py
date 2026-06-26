import sys

sys.path.append('/C:/Users/Marko/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/PyMOL (Anaconda3 (64-bit))')

from pymol import cmd

# The 'sequences.txt' has to contain the one letter amino acid sequences you want to build
# Make sure no excess spaces or newlines are added to the txt list
# Only works on 20 canonical amino acids
with open('sequences.txt', 'r') as f:
    for line in f:
        sequence = line.strip()
        name = sequence
        cmd.fab(sequence, name)
        cmd.alter(name, 'chain="A"')
        cmd.save(f'{name}.pdb', name)

