# Fast Finetuned Forcefields
[![CI](https://github.com/exalearn/fast-finetuned-forcefields/actions/workflows/python-app.yml/badge.svg)](https://github.com/exalearn/fast-finetuned-forcefields/actions/workflows/python-app.yml)
[![Coverage Status](https://coveralls.io/repos/github/exalearn/fast-finetuned-forcefields/badge.svg?branch=main)](https://coveralls.io/github/exalearn/fast-finetuned-forcefields?branch=main)

Toolkit for rapidly assembling forcefields on HPC

## Installation

The easiest route to installation is to use the environment files provided in `envs`. 
For example, installing a CPU-only environment is: 

`conda env create --file envs/environment-cpu.yml --force`

You will also need to install TTM for some features. See [python-ttm](https://github.com/exalearn/python-ttm) for details.
