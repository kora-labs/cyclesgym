# Cyclesgym

## Overview

This repository contains an [OpenAI gym](https://gym.openai.com/) interface to the [Cycles 
agricultural simulator](https://plantscience.psu.edu/research/labs/kemanian/models-and-tools/cycles).

## Installation

We recommend Python 3.8+ installation using [Anaconda](https://www.anaconda.com/products/individual#downloads).

First, create and activate a virtual environment using Anaconda:

```bash
conda create -yn cyclesgym python=3.8
conda activate cyclesgym
```

Then, clone the repo and change working directory

```bash
git clone https://gitlab.inf.ethz.ch/matteotu/cyclesgym.git
cd cyclesgym
```

Subsequently, install the library according to your needs.
If you only need the managers to manipulate cycles files, run:

To install, run:

```bash
pip install -e .
```

If you further want to use some basic libraries to train reinforcement learning agents of the cyclesgym environments use:
```bash
pip install -e .SOLVERS
```

Or, if you are using zsh:
```bash
pip install -e .\[SOLVERS\]
```


### Pre-defined environments
For an explanation of our pre-defined environments and how to use them, see [here](documents/predefined_envs.md).

### Reproduce the experiments  
To reproduce the experiments described in the paper, see [here](documents/experiments.md).

### Custom experiments
For a custom experiment, please refer to the jupyter notebooks available in the `notebooks` folder.