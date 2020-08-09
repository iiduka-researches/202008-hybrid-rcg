# An implementation of the Hybrid Riemannian conjugate gradient method

This repository provides a solver of [pymanopt](https://github.com/pymanopt/pymanopt),
which is an implementation of the Hybrid Riemannian conjugate gradient method.
This implementation is based on the treatise,

> H. Sakai, H. Iiduka: Hybrid Riemannian Conjugate Gradient Methods with Global ConvergenceProperties, (submitted)

All numerical examples presented in the paper are used this implementation.
Our proposed algorithm, which is the modification of `pymanopt.solver.ConjugateGradient` is implemented in `algorithms.py`.

# Acknowledgements
This work was supported by JSPS KAKENHI Grant Number JP18K11184.

# Authors
  * Hiroyuki SAKAI
  * [Hideaki IIDUKA](https://iiduka.net)