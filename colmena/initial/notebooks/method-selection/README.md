# Method Selection

These notebooks evaluate different adjustable parameters of our active learning run, including:

1. Error Threshold: The desired level of error between forcefield and target method for new training points.
1. Sampling Method: What method (e.g., Molecular Dynamics) we use to produce new training points.
1. Sampling Temperature: Once we identified MD as best, tuning which temperature we use for dynamics.
1. Run Length: How many new training points we gather.

We test each method by comparing its performance against a hold-out set produced by running MD with TTM,
and evaluating the quality of structures produced after running Monte Carlo optimization with the forcefield.

## TL;DR

We select Molecular Dynamics at 500K to produce new structures, and adjust the length of the MD runs to produce structures with a 1 meV/atom error.
We still see large improvements in forcefield quality at 20000 training points.
