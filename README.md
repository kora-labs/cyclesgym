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
git clone https://github.com/zuzuba/cyclesgym.git
cd cyclesgym
```

Subsequently, install the library according to your needs.
If you only need the managers to manipulate cycles files, run:

To install, run:

```bash
pip install -e .
```

If you want to use the OpenAI gym environment based on cycles, run:
```bash
pip install -e .ENV
```

Or, if you are using zsh:
```bash
pip install -e .\[ENV\]
```

If you also want to install some basic libraries to solve the environment, substitute .ENV with .ENV_SOLVERS

**Note**: The [pygmo library](https://esa.github.io/pygmo2/) installed with the ENV_SOLVERS option may give problems 
when installed via pip on Mac or Windows, see [here](https://esa.github.io/pygmo2/install.html). 