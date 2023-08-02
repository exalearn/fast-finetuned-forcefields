# Make a Test Set via Global Optimization

The goal in this notebook is to produce a test set for our forcefield full of new structures generated with global optimization.

We will create a set of test structures by running Monte Carlo optimization at different training set sizes. 
Each optimization run will produce a set of a structures that we will pool before creating a test set with
diverse cluster sizes and energies.
