# Tensor Networks

```@setup plot
using Tenet
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))
using GraphMakie
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

Tensor Networks (TN) are a graphical notation for representing complex multi-linear functions. For example, the following equation

```math
\sum_{ijklmnop} A_{im} B_{ijp} C_{njk} D_{pkl} E_{mno} F_{ol}
```

can be represented visually as

The graph's nodes represent tensors and edges represent tensor indices.

In `Tenet`, these objects are represented by the [`TensorNetwork`](@ref) type.

Information about a `TensorNetwork` can be queried with the following functions.

Contraction path optimization and execution is delegated to the [`EinExprs`](https://github.com/bsc-quantic/EinExprs) library. A `EinExpr` is a lower-level form of a Tensor Network, in which the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. `Expr`) but in which every node represents an Einstein summation expression (aka `einsum`).

## Query information

## Modification

### Add/Remove tensors

### Replace existing elements

## Slicing

## Visualization

`Tenet` provides a Package Extension for `Makie` support. You can just import a `Makie` backend and call [`GraphMakie.graphplot`](@ref) on a [`TensorNetwork`](@ref).

```@example plot
tn = rand(TensorNetwork, 14, 4, seed=0) # hide
graphplot(tn, layout=Stress(), labels=true)
```

## Transformations

In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network's structure locally by permuting, contracting, factoring or truncating tensors.

A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.

Our approach is based in [gray2021hyper](@cite), which can also be found in [quimb](https://quimb.readthedocs.io/).

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform`/`transform!` functions.

### Hyperindex converter

### Contraction simplification

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2, 2), (:m, :n, :o)) #hide
E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j)) #hide

tn = TensorNetwork([A, B, C, E]) #hide
reduced = transform(tn, Tenet.ContractSimplification) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Diagonal reduction

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = zeros(Float64, 2, 2, 2, 2) #hide
for i in 1:2 #hide
    for j in 1:2 #hide
        for k in 1:2 #hide
            data[i, i, j, k] = k #hide
        end #hide
    end #hide
end #hide

A = Tensor(data, (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2), (:j, :n)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.DiagonalReduction) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Anti-diagonal reduction

### Dimension truncation

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = rand(3, 3, 3) #hide
data[:, 1:2, :] .= 0 #hide

A = Tensor(data, (:i, :j, :k)) #hide
B = Tensor(rand(3, 3), (:j, :l)) #hide
C = Tensor(rand(3, 3), (:l, :m)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.Truncate) #hide

graphplot!(fig[1, 1], tn; layout=Spring(C=10), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Spring(C=10), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Split simplification

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

v1 = Tensor([1, 2, 3], (:i,)) #hide
v2 = Tensor([4, 5, 6], (:j,)) #hide
m1 = Tensor(rand(3, 3), (:k, :l)) #hide

t1 = contract(v1, v2) #hide
tensor = contract(t1, m1)  #hide

tn = TensorNetwork([ #hide
    tensor, #hide
    Tensor(rand(3, 3, 3), (:k, :m, :n)), #hide
    Tensor(rand(3, 3, 3), (:l, :n, :o)) #hide
]) #hide
reduced = transform(tn, Tenet.SplitSimplification) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced, layout=Spring(C=11); labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```
