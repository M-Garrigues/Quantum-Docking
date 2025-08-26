# Towards molecular docking with neutral atoms

By Mathieu GARRIGUES, Victor ONOFRE and Noé BOSC-HADDAD.

## Abstract

New computational strategies that can speed up the drug discovery process are emerging, such as molecular docking. This method predicts the activity of molecules at the binding site of proteins, helping to select the ones that exhibit desirable behavior and rejecting the rest. However, for large chemical libraries, it's essential to search and score configurations using fewer computational resources while maintaining high precision.

In this work, we map the molecular docking problem to a graph problem, a maximum-weight independent set problem on a unit-disk graph in a Physical Neutral Atom Quantum Processor. Here, each vertex represents an atom trapped by optical tweezers. The Variational Quantum Adiabatic Algorithm (VQAA) approach is used to solve the generic graph problem with two optimization methods, Scipy and Hyperopt. Additionally, a machine learning method is explored using the Adiabatic Algorithm. Results for multiple graphs are presented, and a small instance of the molecular docking problem is solved, demonstrating the potential for future near-term quantum applications.

## Contributions

In this work, we suggest adapting the Neutral Atoms VQAA algorithm to identify the Maximum Independent Set of a graph in the context of molecular docking. Furthermore, we propose an enhancement to this algorithm by leveraging a graph machine learning approach to significantly accelerate its computational efficiency.


## Origin

This work was originally sarted as part of the Blaise Pascal's [Re]Generative Quantum Challenge, in an attempt to use neutral atoms to simulate molecular docking.
Following a 3rd place finish, we continued the work in order to make it complete.


## Cite

Please cite our work as:

@misc{garrigues2024molecular,
      title={Towards molecular docking with neutral atoms},
      author={Mathieu Garrigues and Victor Onofre and Noé Bosc-Haddad},
      year={2024},
      eprint={2402.06770},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}

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
