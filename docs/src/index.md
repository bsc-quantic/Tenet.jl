# Tenet.jl

!!! warning "Status: Alpha stage 🚧"
    `Tenet` is currently in a design stage and thus, the public API might change without notification. We do follow Julia's SemVer specification, so until we arrive to v1.0, patch version upgrades won't break the interface but minor version upgrades might.

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressiveness** _Simple to use_ 👶
- **Flexibility** _Extend it to your needs_ 🔧
- **Performance** _Goes brr... fast_ 🏎️

!!! info "BSC-Quantic's Registry"
    `Tenet` and some of its dependencies are located in our [own Julia registry](https://github.com/bsc-quantic/Registry).
    In order to download `Tenet`, add our registry to your Julia installation by using the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) in a REPL session,

    ```
    ]registry add https://github.com/bsc-quantic/Registry
    ```
    or using the `Pkg` package directly,

    ```julia
    using Pkg
    pkg"registry add https://github.com/bsc-quantic/Registry"
    ```

## Features

- Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- Tensor Network slicing/cuttings
- Automatic Differentiation of TN contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
- Quantum Tensor Networks
  - Matrix Product States (MPS)
  - Matrix Product Operators (MPO)
- 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
- Translation from quantum circuits, powered by [`Quac`](https://github.com/bsc-quantic/Quac.jl)

### Roadmap

The following feature are not yet implemented but are work in progress or are thought to be implemented in the near-mid future:

- Distributed contraction
- Quantum Tensor Networks
  - Tree Tensor Networks (TTN)
  - Projected Entangled Pair States (PEPS)
  - Multiscale Entanglement Renormalization Ansatz (MERA)
- Numerical Tensor Network algorithms
  - Tensor Renormalization Group (TRG)
  - Density Matrix Renormalization Group (DMRG)
