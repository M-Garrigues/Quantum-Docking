# Quantum-Docking

As part of the Blaise Pascal's [Re]Generative Quantum Challenge, we attempt to use neutral atoms to simulate molecular docking.

It features code to go from two molecule files to their binding interaction graph. It then has a quantum solver using VQAA to get the maximum clique out of the graph, representing the most probable binding conformations.



---



## Installation

This repo is designed for python >3.10.

To install the environment, run

```bash
pip install -r requirements.txt
```




---

## DEV

Installation and usage of pre-commit:

Eventhough time is of essence, our code needs to have a minimum of coherence between its different pieces if we want to avoid troubles. Pre-commit applies a uniform style on everyting and will check for basic problems in the code. To install and use it, please follow these steps:

* pip install pre-commit
* or brew install pre-commit
* pre-commit install
* pre-commit run -a

It should then run each time you make a commit. When it fails, it sometimes auto-fixes the problems, mainly if they are style related. You then just need to add the changes and commit again. If they are not auto-fixed, fix them, add them and commit. When everything is green you can push !
