# Quantum-Docking

As part of the Blaise Pascal's [Re]Generative Quantum Challenge, we attempt to use neutral atoms to speed up molecular docking.


Installation and usage of pre-commit:

Eventhough time is of essence, our code needs to have a minimum of coherence between its different pieces if we want to avoid troubles. Pre-commit applies a uniform style on everyting and will check for basic problems in the code. To install and use it, please follow these steps:

* pip install pre-commit
* or brew install pre-commit
* pre-commit install
* pre-commit run -a

It should then run each time you make a commit. When it fails, it sometimes auto-fixes the problems, mainly if they are style related. You then just need to add the changes and commit again. If they are not auto-fixed, fix them, add them and commit. When everything is green you can push !
