# Tenǝʇ.jl

[![CI](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/bsc-quantic/Tenet.jl/branch/master/graph/badge.svg?token=011276A85K)](https://codecov.io/github/bsc-quantic/Tenet.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Registry](https://badgen.net/badge/registry/bsc-quantic/purple)](https://github.com/bsc-quantic/Registry)
[![Documentation: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bsc-quantic.github.io/Tenet.jl/)

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressibility** _Simple to use._ 👶
- **Flexibility** _Extend it to your needs._ 🔧
- **Performance** _Goes brr... fast._ 🏎️

`Tenet` can be executed both at local environments and on large supercomputers.

## Features

- [x] Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- [x] Tensor Network slicing/cuttings
- [x] Automatic Differentiation, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [ ] Distributed contraction
- [ ] Quantum Tensor Networks
  - [x] Matrix Product States (MPS)
  - [x] Matrix Product Operators (MPO)
  - [ ] Tree Tensor Networks (TTN)
  - [ ] Projected Entangled Pair States (PEPS)
  - [ ] Multiscale Entanglement Renormalization Ansatz (MERA)
- [ ] Numerical Tensor Network algorithms
  - [ ] Tensor Renormalization Group (TRG)
  - [ ] Density Matrix Renormalization Group (DMRG)
