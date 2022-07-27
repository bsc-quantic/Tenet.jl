# Tenet.jl

A Proof-of-Concept Julia library for **Ten**sor **Net**works, heavily inspired by [quimb](https://github.com/jcmgray/quimb). `Tenet` is thought to be executed both at local environments for fast iterations and on large supercomputers for big simulations.

## Features

- [ ] Representation of generic tensor networks
- [ ] Implementation of Tensor Network States/Operators
  - [ ] Matrix Product States (MPS)
  - [ ] Tree Tensor Networks (TTN)
  - [ ] Projected Entangled Pair States (PEPS)
  - [ ] Multiscale Entangled Renormalization Ansatz (MERA)
- [ ] Representation of quantum circuits
  - [ ] Load from/Dump to OpenQASM
  - [ ] Gate decompositions
  - [ ] ZX-calculus transformations
- [ ] Abstract interface for contraction path solvers (like `opt_einsum`)
  - [ ] Exhaustive
  - [ ] Greedy
  - [ ] Hypergraph partitioning (like `cotengra`)
- [ ] Distributed execution
- [ ] Advance tensor network techniques
  - [ ] Tensor cuttings/slicing heuristics
  - [ ] Forest ...
  - [ ] Tensor Renormalization Group (TRG)
  - [ ] Density Matrix Renormalization Group (DMRG)

## Differences with `quimb`

`Tenet.jl` is heavily inspired by `quimb` but there are some major differences:

- Obviously, `Tenet.jl` is written in Julia while `quimb` is written in Python.
- In `Tenet.jl`, a quantum circuit is just a symbolic graph of gates; gates are not eagerly contracted to a state. In `quimb`, the actual behaviour is a lil diffuse.