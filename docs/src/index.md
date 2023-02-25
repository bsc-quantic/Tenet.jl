# Tenet.jl

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. `Tenet` is,

- **Expressible** _It is simple to use._
- **Flexible** _It can be extended to your own needs._
- **Performant** _It goes fast._

!!! info "Registry"
    `Tenet` and other supporting libraries are located in our own Julia registry.
    In order to download `Tenet`, add our registry to your Julia installation by using the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) in a REPL session,
    ```
    registry add https://github.com/bsc-quantic/Registry
    ```
    or using the `Pkg` package directly,
    ```julia
    using Pkg
    pkg"registry add https://github.com/bsc-quantic/Registry"
    ```

## Features

### Arbitrary Tensor Network Contraction

Tenet can represent `TensorNetwork`s of `Arbitrary` form. Thanks to [`OptimizedEinsum`](https://github.com/bsc-quantic/OptimizedEinsum.jl), it can find contraction paths (using the `contractpath` function) which can then be computed by [`OMEinsum`](https://github.com/under-Peter/OMEinsum.jl) (using the `contract` function).

### Visualization

Thanks to [`Makie`](https://github.com/MakieOrg/Makie.jl), TNs can be visualized in 2D and 3D on different backends (OpenGL, WebGL and Cairo).

```@setup plot
# using WGLMakie
# using JSServe
# Page(exportable=true, offline=true)
using CairoMakie
CairoMakie.activate!(type = "svg")
```

```@example plot
using Tenet

tn = rand(TensorNetwork, 10, 3)
plot(tn, labels=true)
```

### (Quantum) Tensor Network Ansatzes

Tenet provides some popular Tensor Network Ansatzes. Currently implemented are:

- `MatrixProductState`

## Contents

```@contents
Pages = [
    "tensor.md",
    "tensor-network.md",
    "ansatz.md",
    "transformations.md",
    "alternatives.md",
]
```
