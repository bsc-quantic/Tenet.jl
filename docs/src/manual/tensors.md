# Tensors

```@setup tensor
using Tenet
```

If you have reached here, you probably know what a tensor is, and probably have heard many jokes about _what a tensor is_[^1]. Nevertheless, we are gonna give a brief remainder.

[^1]: For example, recursive definitions like _a tensor is whatever that transforms as a tensor_.

A tensor $T$ of order[^2] $n$ is a multilinear[^3] function between $n$ vector spaces over a field $\mathcal{F}$.

[^2]: The _order_ of a tensor may also be known as _rank_ or _dimensionality_ in other fields. However, these can be missleading, since it has nothing to do with the _rank_ of linear algebra nor with the _dimensionality_ of a vector space. Thus we prefer to use the word _order_.
[^3]: Meaning that the relationships between the output and the inputs, and the inputs between them, are linear.

```math
T : \mathcal{F}^{\dim(1)} \times \dots \times \mathcal{F}^{\dim(n)} \mapsto \mathcal{F}
```

In layman's terms, it is a linear function that maps a set of vectors to a scalar.

```math
T(\mathbf{v}^{(1)}, \dots, \mathbf{v}^{(n)}) = c \in \mathcal{F} \qquad\qquad \forall i, \mathbf{v}^{(i)} \in \mathcal{F}^{\dim(i)}
```

Tensor algebra is a higher-order generalization of linear algebra, where scalar numbers can be viewed as _order-0 tensors_, vectors as _order-1 tensors_, matrices as _order-2 tensors_, ...

Letters are used to identify each of the vector spaces the tensor relates to.
In computer science, you would intuitively think of tensors as "_n-dimensional arrays with named dimensions_".

```math
T_{ijk} \iff \mathtt{T[i,j,k]}
```

## The `Tensor` type

In `Tenet`, a tensor is represented by the `Tensor` type, which wraps an array and a list of index names. As it subtypes `AbstractArray`, many array operations are automatically dispatched.

You can create a `Tensor` by passing an `AbstractArray` and a `Vector` or `Tuple` of `Symbol`s.

```@repl tensor
Tᵢⱼₖ = Tensor(rand(3,5,2), (:i,:j,:k))
```

Use `parent` and `inds` to access the underlying array and indices respectively.

```@repl tensor
parent(Tᵢⱼₖ)
inds(Tᵢⱼₖ)
```

The _dimensionality_ or size of each index can be consulted using the `size` function.

```@repl tensor
size(Tᵢⱼₖ)
size(Tᵢⱼₖ, :j)
length(Tᵢⱼₖ)
```

Note that these indices are the ones that really define the dimensions of the tensor and not the order of the array dimensions. This is key for interacting with other tensors.

```@repl tensor
a = Tensor([1 0; 1 0], (:i, :j))
b = Tensor([1 1; 0 0], (:j, :i))
c = a + b
parent(a) + parent(b)
```

As such [`adjoint`](@ref) doesn't permute the dimensions; it just conjugates the array.

```@repl tensor
d = Tensor([1im 2im; 3im 4im], (:i, :j))
d'
conj(d)
```
