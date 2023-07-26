# Transformations

In tensor network computations, it is common to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations can modify the network's structure in a variety of ways such as reducing dimensions or ranks of tensors, converting certain indices, or even reordering tensor indices.

A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.

Our approach has been significantly inspired by the ideas presented in the [Quimb](https://quimb.readthedocs.io/) library, explained in [this paper](https://arxiv.org/pdf/2002.01935.pdf).

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform`/`transform!` functions.

```@docs
transform
transform!
```

# Example
Here we show how can we reduce the complexity of the tensor network by applying a tranformation to it:
```julia-repl
julia> tn = TensorNetwork(...)
julia> transformed_tn = transform(tn, Tenet.RankSimplification)

julia> fig = Figure()
julia> ax1 = Axis(fig[1, 1]; title="Original TensorNetwork")
julia> p1 = plot!(ax1, tn; node_size=5.)
julia> ax2 = Axis(fig[1, 2], title="Transformed TensorNetwork")
julia> p2 = plot!(ax2, tn2; node_size=5.)
julia> ax1.titlesize=20
julia> ax2.titlesize=20
```
```@raw html
<figure>
<img width=500 src="../assets/transformation.svg" alt="Before and after transformation in a Tensor Network">
</figure>
```

# Transformations

```@docs
Tenet.HyperindConverter
Tenet.DiagonalReduction
Tenet.RankSimplification
Tenet.AntiDiagonalGauging
Tenet.ColumnReduction
Tenet.SplitSimplification
```