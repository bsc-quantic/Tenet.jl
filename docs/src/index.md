# Tenet.jl

!!! info "BSC-Quantic's Registry"
    `Tenet` and some of its dependencies are located in our [own Julia registry](https://github.com/bsc-quantic/Registry).
    In order to download `Tenet`, add our registry to your Julia installation by using the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) in a REPL session,

    ```julia
    using Pkg
    pkg"registry add https://github.com/bsc-quantic/Registry"
    ```

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressiveness** _Simple to use_ 👶
- **Flexibility** _Extend it to your needs_ 🔧
- **Performance** _Goes brr... fast_ 🏎️

A video of its presentation at JuliaCon 2023 can be seen here:

```@raw html
<div class="youtube-video">
<iframe width="560" style="height='315'" src="https://www.youtube-nocookie.com/embed/8BHGtm6FRMk?si=bPXB6bPtK695HFIR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
```

## Features

- Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- Tensor Network slicing/cuttings
- Automatic Differentiation of TN contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
- Quantum Tensor Networks
  - Matrix Product States (MPS)
  - Matrix Product Operators (MPO)
  - Projected Entangled Pair States (PEPS)
- 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
- Translation from quantum circuits, powered by [`Quac`](https://github.com/bsc-quantic/Quac.jl)

### Roadmap

The following feature are not yet implemented but are work in progress or are thought to be implemented in the near-mid future:

- Distributed contraction
- Quantum Tensor Networks
  - Tree Tensor Networks (TTN)
  - Multiscale Entanglement Renormalization Ansatz (MERA)
- Numerical Tensor Network algorithms
  - Tensor Renormalization Group (TRG)
  - Density Matrix Renormalization Group (DMRG)
