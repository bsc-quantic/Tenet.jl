# Tenǝʇ.jl

[![CI](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/bsc-quantic/Tenet.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/bsc-quantic/Tenet.jl/branch/master/graph/badge.svg?token=011276A85K)](https://codecov.io/github/bsc-quantic/Tenet.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Registry](https://badgen.net/badge/registry/bsc-quantic/purple)](https://github.com/bsc-quantic/Registry)
[![Documentation: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bsc-quantic.github.io/Tenet.jl/)
[![Documentation: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bsc-quantic.github.io/Tenet.jl/dev/)

A Julia library for **Ten**sor **Net**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressiveness** _Simple to use._ 👶
- **Flexibility** _Extend it to your needs._ 🔧
- **Performance** _Goes brr... fast._ 🏎️

A video of its presentation at JuliaCon 2023 can be seen here:

[![Watch the video](https://img.youtube.com/vi/8BHGtm6FRMk/maxresdefault.jpg)](https://youtu.be/8BHGtm6FRMk)

## Features

- [x] Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- [x] Tensor Network slicing/cuttings
- [x] Automatic Differentiation of TN contraction
- [ ] Distributed contraction
- [x] Local Tensor Network transformations
  - [x] Hyperindex converter
  - [x] Rank simplification
  - [x] Diagonal reduction
  - [x] Anti-diagonal gauging
  - [x] Column reduction
  - [x] Split simplification
- [x] 2D & 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
