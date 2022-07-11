# Deep Memory Update
This repository contains implementation of the Deep Memory Update architecture and
experiments performed in the original paper.

## Environment installation
**Note : Use Python 3.7 or newer**

* Install [poetry](https://python-poetry.org/)
* Install environment by issuing `poetry install` in the root of the repository

## Running experiments with guild
* Activate the environment by `poetry shell`
* Run the experiment, for example: `guild run dmu:order min-length=100 recurrent-cells="2 4" max_epochs=100`

## Reproducing experiments from the paper
To reproduce a single run of an experiment for all recurrent modules run a shell script
from the `experiments_repro` directory. For example to run NatLang experiment issue
`./experiments_repro/nat_lang.sh`.
