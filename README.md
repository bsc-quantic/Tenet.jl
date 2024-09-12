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

## Features

- Optimized Tensor Network contraction order, powered by [EinExprs.jl](https://github.com/bsc-quantic/EinExprs.jl)
- Tensor Network slicing/cuttings
- Automatic Differentiation of TN contraction
- Distributed contraction
- Local Tensor Network transformations/simplifications
- 2D & 3D visualization of large networks, powered by [Makie.jl](https://github.com/MakieOrg/Makie.jl)
- Quantum Tensor Networks
- Conversion from/to [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl)

## Preview

A video of its presentation at JuliaCon 2023 can be seen here:

[![Watch the video](https://img.youtube.com/vi/8BHGtm6FRMk/maxresdefault.jpg)](https://youtu.be/8BHGtm6FRMk)
