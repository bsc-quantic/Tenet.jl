# Visualization

```@setup viz
using Makie
Makie.inline!(true)

using Tenet
```

`Tenet` provides a Package Extension for `Makie` support. You can just import a `Makie` backend and call [`Makie.plot`](@ref) on a [`TensorNetwork`](@ref).

```@docs
Makie.plot(::Tenet.TensorNetwork)
```

```@example viz
using CairoMakie # hide
tn = rand(TensorNetwork, 20, 4, seed=0) # hide
plot(tn, labels=true)
```
