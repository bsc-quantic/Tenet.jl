# Transformations

In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network's structure locally by permuting, contracting, factoring or truncating tensors.

A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.

Our approach has been significantly inspired by the ideas presented in the [Quimb](https://quimb.readthedocs.io/) library, explained in [this paper](https://arxiv.org/pdf/2002.01935.pdf).

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform`/`transform!` functions.

```@docs
transform
transform!
```

# Example
Here we show how can we reduce the complexity of the tensor network by applying a tranformation to it:
```@example
tn = TensorNetwork(...)
transformed_tn = transform(tn, Tenet.RankSimplification)

fig = Figure() # hide
ax1 = Axis(fig[1, 1]; title="Original TensorNetwork") # hide
p1 = plot!(ax1, tn; node_size=5.) # hide
ax2 = Axis(fig[1, 2], title="Transformed TensorNetwork") # hide
p2 = plot!(ax2, tn2; node_size=5.) # hide
ax1.titlesize=20 # hide
ax2.titlesize=20 # hide
fig # hide
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