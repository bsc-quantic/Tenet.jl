# Tenet.jl

!!! danger "Status: Alpha stage 🚧"
    `Tenet` is currently in alpha stage and thus, the public API might change without notification.

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressibility** _Simple to use._
- **Flexibility** _Extend it to your needs._
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

- Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- Automatic Differentiation of TN contraction
- Support for arbitrary network structures
- Local transformations and simplifications
- 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
- High-level interface
- Translation from quantum circuits
