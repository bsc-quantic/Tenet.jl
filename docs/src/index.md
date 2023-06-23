# Tenet.jl

!!! warn "Status: Alpha stage 🚧"
    `Tenet` is currently in a design stage and thus, the public API might change without notification. We do follow Julia's SemVer specification, so until we arrive to v1.0, patch version upgrades won't break the interface but minor version upgrades might.

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressibility** _Simple to use._ 👶
- **Flexibility** _Extend it to your needs._ 🔧
- **Performance** _Goes brr... fast._ 🏎️

!!! info "Registry"
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

- [x] Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- [x] Tensor Network slicing/cuttings
- [x] Automatic Differentiation of TN contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
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
- [x] 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
- [x] Translation from quantum circuits, powered by [`Quac`](https://github.com/bsc-quantic/Quac.jl)