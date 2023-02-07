# Tenet.jl

A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers.

It aims for,

- **Expressibility** _It should be simple to use._
- **Flexibility** _It can be extended to your own needs._
- **Performance** _It goes fast._

## Features

### Arbitrary Tensor Network Contraction

Tenet can represent `TensorNetwork`s of `Arbitrary` form. Thanks to [`OptimizedEinsum`](https://github.com/bsc-quantic/OptimizedEinsum.jl), it can find contraction paths (using the `contractpath` function) which can then be computed by [`OMEinsum`](https://github.com/under-Peter/OMEinsum.jl) (using the `contract` function).

### Visualization

Thanks to [`Makie`](https://github.com/MakieOrg/Makie.jl), TNs can be visualized in 2D and 3D on different backends (OpenGL, WebGL and Cairo).

```@example
using Tenet # hide
using WGLMakie

tn = rand(TensorNetwork, 10, 3)
plot(tn)
```

### (Quantum) Tensor Network Ansatzes

Tenet provides some popular Tensor Network Ansatzes. Currently implemented are:

- `MatrixProductState`

## Contents

```@contents
Pages = [
    "tensor.md",
    "alternatives.md",
]
```
