# Adapt for more TTM

The `fff` codebase uses eV/atom for everything to be consistent with ASE's tooling. 
Our model was originally trained using kcal/mol.
All we should need to do is change the units of the output layer.
