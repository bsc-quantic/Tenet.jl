# Tensor Networks

```@setup plot
using Tenet
using EinExprs
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))
using GraphMakie
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

When the number of tensors in some einsum expression starts to grow, the traditional written mathematical notation starts being inadecuate and it's prone to errors.
Physicists noticed about this and developed[^1] a graphical notation called _Tensor Networks_, in which tensors of a einsum are represented by the vertices of a graph and the edges are the tensor indices connecting tensors.

[^1]: This manual is no place for history but first developments trace back to Penrose.

For example, the following equation

```math
\sum_{ijklmnop} A_{im} B_{ijp} C_{njk} D_{pkl} E_{mno} F_{ol}
```

can be represented visually as

```@raw html
<img class="light-only" src="/assets/tn-sketch-light.svg" alt="Sketch of a Tensor Network"/>
<img class="dark-only" src="/assets/tn-sketch-dark.svg" alt="Sketch of a Tensor Network (dark mode)"/>
```

Not exclusively, but much of the research on Tensor Networks comes from the physics fields, so it's to be expected that the majority of Tensor Network libraries are written from the physics point of view.
This has some consequences on how the abstractions are implemented and what interface is offered to the user.
For example, some libraries only offer access to certain structured Tensor Networks like MPS or PEPS, forbiding modification of the graph topology.
This is completely fine, but it's not the design philosophy of Tenet.

Instead, Tenet constructs abstractions layer by layer, starting from the most essential and adding more and more details for sofistification.
The most essential of these layers in Tenet is the [`TensorNetwork`](@ref) type.

## The `TensorNetwork` type

In `Tenet`, Tensor Networks are represented by the [`TensorNetwork`](@ref) type.
In order to fit all posible use-cases of [`TensorNetwork`](@ref) implements a **hypergraph**[^2] of [`Tensor`](@ref) objects, with support for open-indices.

[^2]: A hypergraph is the generalization of a graph but where edges are not restricted to connect 2 vertices, but any number of vertices.

For example, the example above can be constructed as follows:

```@repl plot
tn = TensorNetwork([
    Tensor(zeros(2,2), (:i, :m)), # A
    Tensor(zeros(2,2,2), (:i, :j, :p)), # B
    Tensor(zeros(2,2,2), (:n, :j, :k)), # C
    Tensor(zeros(2,2,2), (:p, :k, :l)), # D
    Tensor(zeros(2,2,2), (:m, :n, :o)), # E
    Tensor(zeros(2,2), (:o, :l)), # F
])
```

[`Tensor`](@ref)s can be added or removed after construction using [`push!`](@ref), [`pop!`](@ref), [`delete!`](@ref) and [`append!`](@ref) methods.

```@repl plot
A = only(pop!(tn, [:i, :m]))
tn
push!(tn, A)
```

## Query information

### Replace existing elements

## Contraction

When contracting two tensors in a Tensor Network, diagrammatically it is equivalent to fusing the two vertices of the involved tensors.

```@raw html
<figure>
<img class="light-only" src="/assets/tensor-matmul-light.svg" alt="Matrix Multiplication using Tensor Network notation"/>
<img class="dark-only" src="/assets/tensor-matmul-dark.svg" alt="Matrix Multiplication using Tensor Network notation (dark mode)"/>
<figcaption>Matrix Multiplication using Tensor Network notation</figcaption>
</figure>
```

The ultimate goal of Tensor Networks is to compose tensor contractions until you get the final result tensor.
Tensor contraction is associative, so mathematically the order in which you perform the contractions doesn't matter, but the computational cost depends (and a lot) on the order.
Actually, finding the optimal contraction path is a NP-complete problem and general tensor network contraction is #P-complete.

But don't fear! Optimal contraction paths can be computed for small tensor networks (i.e. in the order of of up to 40 indices) in a laptop, and several good heuristics and approximate algorithms are known for solving such problem.
In Tenet, contraction path optimization is delegated to the [`EinExprs`](https://github.com/bsc-quantic/EinExprs) library.
A `EinExpr` is a lower-level form of a Tensor Network, in which the contents of the arrays have been left out and the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. `Expr`) but in which every node represents an Einstein summation expression (aka `einsum`). You can get the `EinExpr` (which again, represents the contraction path) by calling [`einexpr`](@ref).

```@repl plot
einexpr(tn; optimizer=Exhaustive())
```

## Visualization

`Tenet` provides visualization support with [`GraphMakie`](https://github.com/MakieOrg/GraphMakie.jl). You can just import a [`Makie`](https://docs.makie.org/) backend and call [`GraphMakie.graphplot`](@ref) on a [`TensorNetwork`](@ref).

```@example plot
graphplot(tn, layout=Stress(), labels=true)
```

## Slicing

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
