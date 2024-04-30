# Visualization

```@setup viz
using Makie
using GraphMakie
Makie.inline!(true)
set_theme!(resolution=(800,400))

using CairoMakie
CairoMakie.activate!(type = "svg")

using Tenet
```

`Tenet` provides a Package Extension for `Makie` support. You can just import a `Makie` backend and call [`Makie.plot`](@ref) on a [`TensorNetwork`](@ref).

```@docs
GraphMakie.graphplot(::Tenet.TensorNetwork)
```

```@example viz
tn = rand(TensorNetwork, 14, 4, seed=0) # hide
graphplot(tn, labels=true)
```
