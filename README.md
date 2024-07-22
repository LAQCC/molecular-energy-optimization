# A brief overview of VQE

::: {.meta}
:property=\"og:description\": Find the ground state of a Hamiltonian
using the variational quantum eigensolver algorithm.
:property=\"og:image\":
<https://pennylane.ai/qml/_static/demonstration_assets//pes_h2.png>
:::

::: {.related}
tutorial_quantum_chemistry Building molecular Hamiltonians
vqe_parallel VQE with parallel QPUs with Rigetti tutorial_vqe_qng
Accelerating VQE with quantum natural gradient
tutorial_vqe_spin_sectors VQE in different spin sectors tutorial_vqt
Variational quantum thermalizer
:::

_Author: Alain Delgado --- Posted: 08 February 2020. Last updated: 29
August 2023._

The Variational Quantum Eigensolver (VQE) is a flagship algorithm for
quantum chemistry using near-term quantum computers. It is an
application of the [Ritz variational
principle](https://en.wikipedia.org/wiki/Ritz_method), where a quantum
computer is trained to prepare the ground state of a given molecule.

The inputs to the VQE algorithm are a molecular Hamiltonian and a
parametrized circuit preparing the quantum state of the molecule. Within
VQE, the cost function is defined as the expectation value of the
Hamiltonian computed in the trial state. The ground state of the target
Hamiltonian is obtained by performing an iterative minimization of the
cost function. The optimization is carried out by a classical optimizer
which leverages a quantum computer to evaluate the cost function and
calculate its gradient at each optimization step.

In this tutorial you will learn how to implement the VQE algorithm in a
few lines of code. As an illustrative example, we use it to find the
ground state of the hydrogen molecule, $\mathrm{H}_2$. First, we build
the molecular Hamiltonian using a minimal basis set approximation. Next,
we design the quantum circuit preparing the trial state of the molecule,
and the cost function to evaluate the expectation value of the
Hamiltonian. Finally, we select a classical optimizer, initialize the
circuit parameters, and run the VQE algorithm using a PennyLane
simulator.



## Referência

Sennane, W., Piquemal, J.-P., & Rančić, M. J. (2022). Calculating the Ground State Energy of Benzene Under Spatial Deformations with Noisy Quantum Computing. *TotalEnergies* e *Laboratoire de Chimie Théorique, Sorbonne Université, UMR7616 CNRS*. [arXiv:2203.05275v2](https://arxiv.org/abs/2203.05275v2)

