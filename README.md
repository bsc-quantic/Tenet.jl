# Tenǝʇ

[![CI](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/bsc-quantic/Tenet.jl/branch/master/graph/badge.svg?token=011276A85K)](https://codecov.io/github/bsc-quantic/Tenet.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Registry](https://badgen.net/badge/registry/bsc-quantic/purple)](https://github.com/bsc-quantic/Registry)

A Proof-of-Concept Julia library for **Ten**sor **Net**works, heavily inspired by [quimb](https://github.com/jcmgray/quimb). `Tenet` can be executed both at local environments and on large supercomputers.

## Features

- [x] Contraction of generic tensor networks
- [x] Tensor Network slicing
- [ ] Ansatz Tensor Network (States/Operators)
  - [x] Matrix Product States (MPS)
  - [ ] Matrix Product Operators (MPO)
  - [ ] Tree Tensor Networks (TTN)
  - [ ] Projected Entangled Pair States (PEPS)
  - [ ] Multiscale Entangled Renormalization Ansatz (MERA)
- [ ] Automatic Differentiation
- [ ] Advance tensor network techniques
  - [ ] Tensor Renormalization Group (TRG)
  - [ ] Density Matrix Renormalization Group (DMRG)
- [ ] Distributed contraction
