# The `TensorNetwork` class

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

In `Tenet`, Tensor Networks are represented by the [`TensorNetwork`](@ref) type.
In order to fit all posible use-cases of [`TensorNetwork`](@ref) implements a **hypergraph**[^2] of [`Tensor`](@ref) objects, with support for open-indices and multiple shared indices between two tensors.

[^2]: A hypergraph is the generalization of a graph but where edges are not restricted to connect 2 vertices, but any number of vertices.

For example, this Tensor Network...

```@raw html
<img class="light-only" width="70%" src="/assets/tn-sketch-light.svg" alt="Sketch of a Tensor Network"/>
<img class="dark-only" width="70%" src="/assets/tn-sketch-dark.svg" alt="Sketch of a Tensor Network (dark mode)"/>
```

... can be constructed as follows:

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

You can also replace existing tensors and indices with [`replace`](@ref) and [`replace!`](@ref).

```@repl plot
:i ∈ tn
replace!(tn, :i => :my_index)
:i ∈ tn
:my_index ∈ tn
replace!(tn, :my_index => :i) # hide
```

!!! warning
    Note that although it is a bit unusual but completely legal to have more than one tensor with the same indices, there can be problems when deciding which tensor to be replaced.
    Because of that, you **must** pass the exact tensor you want to replace. A copy of it won't be valid.

## The `AbstractTensorNetwork` interface

Subclasses of [`TensorNetwork`](@ref) inherit from the [`AbstractTensorNetwork`](@ref Tenet.AbstractTensorNetwork) abstract type.
Subtypes of it are required to implement a [`TensorNetwork`](@ref) method that returns the composed [`TensorNetwork`](@ref) object.
In exchange, [`AbstracTensorNetwork`](@ref Tenet.AbstractTensorNetwork) automatically implements [`tensors`](@ref) and [`inds`](@ref) methods for any interface-fulfilling subtype.

As the names suggest, [`tensors`](@ref) returns tensors and [`inds`](@ref) returns indices.

```@repl plot
tensors(tn)
inds(tn)
```

What is interesting about them is that they implement a small query system based on keyword dispatching. For example, you can get the tensors that contain or intersect with a subset of indices using the `contains` or `intersects` keyword arguments:

!!! note
    Keyword dispatching doesn't work with multiple unrelated keywords. Checkout [Keyword dispatch](@ref) for more information.

```@repl plot
tensors(tn; contains=[:i,:m]) # A
tensors(tn; intersects=[:i,:m]) # A, B, E
```

Or get the list of open indices (which in this case is none):

```@repl plot
inds(tn; set = :open)
```

The list of available keywords depends on the layer, so don't forget to check the 🧭 API reference!

## Contraction

When contracting two tensors in a Tensor Network, diagrammatically it is equivalent to fusing the two vertices of the involved tensors.

```@raw html
<figure>
<img class="light-only" width="70%" src="/assets/tensor-matmul-light.svg" alt="Matrix Multiplication using Tensor Network notation"/>
<img class="dark-only" width="70%" src="/assets/tensor-matmul-dark.svg" alt="Matrix Multiplication using Tensor Network notation (dark mode)"/>
<figcaption>Matrix Multiplication using Tensor Network notation</figcaption>
</figure>
```

The ultimate goal of Tensor Networks is to compose tensor contractions until you get the final result tensor.
Tensor contraction is associative, so mathematically the order in which you perform the contractions doesn't matter, but the computational cost depends (and a lot) on the order (which is also known as _contractio path_).
Actually, finding the optimal contraction path is a NP-complete problem and general tensor network contraction is #P-complete.

But don't fear! Optimal contraction paths can be found for small tensor networks (i.e. in the order of of up to 40 indices) in a laptop, and several approximate algorithms are known for obtaining quasi-optimal contraction paths.
In Tenet, contraction path optimization is delegated to the [`EinExprs`](https://github.com/bsc-quantic/EinExprs) library.
A `EinExpr` is a lower-level form of a Tensor Network, in which the contents of the arrays have been left out and the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. `Expr`) but in which every node represents an Einstein summation expression (aka `einsum`). You can get the `EinExpr` (which again, represents the contraction path) by calling [`einexpr`](@ref).

```@repl plot
path = einexpr(tn; optimizer=Exhaustive())
```

Once a contraction path is found, you can pass it to the [`contract`](@ref) method. Note that if no contraction `path` is provided, then Tenet will choose an optimizer based on the characteristics of the Tensor Network which will be used for finding the contraction path.

```@repl plot
contract(tn; path=path)
contract(tn)
```

If you want to manually perform the contractions, then you can indicate which index to contract by just passing the index. If you call [`contract!`](@ref), the Tensor Network will be modified in-place and if [`contract`](@ref) is called, a mutated copy will be returned.

```@repl plot
contract(tn, :i)
```

## Visualization

`Tenet` provides visualization support with [`GraphMakie`](https://github.com/MakieOrg/GraphMakie.jl). Import a [`Makie`](https://docs.makie.org/) backend and call [`GraphMakie.graphplot`](@ref) on a [`TensorNetwork`](@ref).

```@example plot
graphplot(tn, layout=Stress(), labels=true)
```

## Transformations

In Tensor Network computations, it is good practice transform before in order to prepare the network for further operations.
In the case of exact Tensor Network contraction, a crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search.
This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself (althoug it can), but more importantly, it reduces the size of the (hyper)graph.
These transformations can modify the network's structure locally by permuting, contracting, factoring or truncating tensors.
Our approach is based in [gray2021hyper](@cite), which can also be found in [quimb](https://quimb.readthedocs.io/).

Tenet provides a set of predefined transformations which you can apply to your `TensorNetwork` using the [`transform`](@ref)/[`transform!`](@ref) functions.

### HyperFlatten

The [`HyperFlatten`](@ref Tenet.HyperFlatten) transformation converts hyperindices to COPY-tensors (i.e. [Kronecker delta](https://wikipedia.org/wiki/Kronecker_delta)s).
It is useful when some method requires the Tensor Network to be represented as a graph and not as a hypergraph.
The opposite transformation is [`HyperGroup`](@ref Tenet.HyperGroup).

```@example plot
fig = Figure() # hide

A = Tensor(rand(2,2), [:i,:j])
B = Tensor(rand(2,2), [:i,:k])
C = Tensor(rand(2,2), [:i,:l])

tn = TensorNetwork([A, B, C])
transformed = transform(tn, Tenet.HyperFlatten())

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], transformed; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Contraction simplification

The [`ContractionSimplification`](@ref Tenet.ContractSimplification) transformation contracts greedily tensors whose resulting tensor is smaller (in number of elements or in rank, it's configurable). These preemptive contractions don't affect the result of the contraction path but reduce the search space.

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l))
B = Tensor(rand(2, 2), (:i, :m))
C = Tensor(rand(2, 2, 2), (:m, :n, :o))
E = Tensor(rand(2, 2, 2), (:o, :p, :j))

tn = TensorNetwork([A, B, C, E])
transformed = transform(tn, Tenet.ContractSimplification)

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], transformed; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Diagonal reduction

The [`DiagonalReduction`](@ref Tenet.DiagonalReduction) transformation tries to reduce the rank of tensors by looking up tensors that have pairs of indices with a diagonal structure between them.

```math
A_{ijkl} = \begin{cases}
A'_{i k l} &\text{for } i = j \\
0 \quad &\text{for } i \neq j
\end{cases} \quad \mapsto A_{ijkl} = A'_{\alpha k l} \delta_{\alpha i j}
```

In such cases, the diagonal structure between the indices can be extracted into a COPY-tensor and the two indices of the tensor are fused into one.

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = zeros(Float64, 2, 2, 2, 2)
for i in 1:2
    for j in 1:2
        for k in 1:2
            data[i, i, j, k] = k
        end
    end
end

A = Tensor(data, (:i, :j, :k, :l))

tn = TensorNetwork([A])
transformed = transform(tn, Tenet.DiagonalReduction)

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], transformed; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Truncation

The [`Truncate`](@ref Tenet.Truncate) transformation truncates the dimension of a [`Tensor`](@ref) if it founds slices of it where all elements are smaller than a given threshold.

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = rand(3, 3, 3)
data[:, 1:2, :] .= 0

A = Tensor(data, (:i, :j, :k))
B = Tensor(rand(3, 3), (:j, :l))
C = Tensor(rand(3, 3), (:l, :m))

tn = TensorNetwork([A, B, C])
transformed = transform(tn, Tenet.Truncate)

graphplot!(fig[1, 1], tn; layout=Spring(C=10), labels=true) #hide
graphplot!(fig[1, 2], transformed; layout=Spring(C=10), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Split simplification

The [`SplitSimplification`](@ref Tenet.SplitSimplification) transformation decomposes a [`Tensor`](@ref) using the Singular Value Decomposition (SVD) if the rank of the decomposition is smaller than the original; i.e. there are singular values which can be truncated.

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

# outer product has rank=1
v1 = Tensor([1, 2, 3], (:i,))
v2 = Tensor([4, 5, 6], (:j,))
t1 = contract(v1, v2)

m1 = Tensor(rand(3, 3), (:k, :l))
tensor = contract(t1, m1) 

tn = TensorNetwork([
    tensor,
    Tensor(rand(3, 3, 3), (:k, :m, :n)),
    Tensor(rand(3, 3, 3), (:l, :n, :o))
])
transformed = transform(tn, Tenet.SplitSimplification)

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], transformed, layout=Spring(C=10.0, iterations=200); labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```
