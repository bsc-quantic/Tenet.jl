# Tensor

```@meta
DocTestSetup = quote
    using Tenet
end
```

A tensor $T$ of order[^1] $n$ is a multilinear[^2] application between $n$ vector spaces.

[^1]: The _order_ of a tensor may also be known as _rank_ or _dimensionality_ in other fields, althought these can be missleading, since it has nothing to do with the _rank_ of linear algebra nor with the _dimensionality_ of a vector space. We prefer to use _order_.
[^2]: Meaning that the relationships between the output and the inputs, and the inputs between them, are linear.

```math
T(\mathbf{v}^{(1)}, \dots, \mathbf{v}^{(n)}) = c \qquad\qquad \mathbf{v}^{(i)} \in V^{(i)}, \forall i
```

It is a higher-dimensional generalization of linear algebra, where scalar number can be viewed as _order-0 tensors_, vectors as _order-1 tensors_, matrices as _order-2 tensors_, ...

Letters are used to identify each of the vector spaces the tensor relates to. In computer science, you would think of tensors as "_n-dimensional arrays with named dimensions_".

```math
T_{ijk}
```

You can create a `Tensor` by passing an array and a list of `Symbol`s that name indices.

```@example tensor
using Tenet
T = Tensor(rand(5,3,2), (:i,:j,:k))
```

The _dimensionality_ or size of each index can be consulted using the `size` function.

```@example tensor
@assert size(T, :i) == 5
@assert size(T, :j) == 3
@assert size(T, :k) == 2
@assert length(T) == prod(i -> size(T,i), [:i,:j,:k])
```

```@docs
Base.size(::Tensor)
Base.size(::Tensor,i)
```
