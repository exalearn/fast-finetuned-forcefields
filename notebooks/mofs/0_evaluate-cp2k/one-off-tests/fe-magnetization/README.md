# Exploring Magnetization

We know that the Fe ions in some MOFs should have non-zero spin states.
Allowing CP2K to express ions in these magnetic states can require:

1. Running [in unrestricted KS](https://manual.cp2k.org/cp2k-2023_1-branch/CP2K_INPUT/FORCE_EVAL/DFT.html#UKS)
1. Defining the [initial spin states](https://manual.cp2k.org/cp2k-2023_1-branch/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND.html#MAGNETIZATION)
1. Increasing [the multiplicity](https://manual.cp2k.org/cp2k-2023_1-branch/CP2K_INPUT/FORCE_EVAL/DFT.html#MULTIPLICITY) to account for the unpaired electrons

