# Transformations

In tensor network computations, it is common to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations can modify the network's structure in a variety of ways such as reducing dimensions or ranks of tensors, converting certain indices, or even reordering tensor indices.

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform` and `transform!` functions. The `transform` function is non-mutating, meaning it will return a new `TensorNetwork` where the transformation has been applied, leaving the original network unchanged. For in-place transformation, you can use the mutating version, `transform!`.

```@docs
transform
transform!
```

# Examples
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
```

```@docs
Tenet.DiagonalReduction
```
```@docs
Tenet.RankSimplification
```

```@docs
Tenet.AntiDiagonalGauging
```

```@docs
Tenet.ColumnReduction
```

```@docs
Tenet.SplitSimplification
```
