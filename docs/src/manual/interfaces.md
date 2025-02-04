# Interfaces

Julia doesn't have a formal definition of interface built into the language. Instead it relies on [duck typing](https://wikipedia.org/wiki/Duck_typing).
Any declaration of a formal interface is then the documentation written for it.

## [TensorNetwork](@id man-interface-tensornetwork)

A [`TensorNetwork` (interface)](@ref man-interface-tensornetwork) is a collection of [`Tensor`](@ref)s forming a graph structure.

| Required method                      | Brief description                                     |
| :----------------------------------- | :---------------------------------------------------- |
| [`tensors(tn)`](@ref tensors)        | Returns the list of [`Tensor`](@ref)s present in `tn` |
| `inds_set_all`                       |                                                       |
| `inds_set_open`                      |                                                       |
| `inds_set_inner`                     |                                                       |
| `inds_set_hyper`                     |                                                       |
| `replace!(tn, index => new_index)`   |                                                       |
| `replace!(tn, tensor => new_tensor)` |                                                       |

!!! todo
    - Are `contract` and `contract!` required methods for the interface?
    - Is conversion to `TensorNetwork` a required method method for the interface?

The `inds_set_all`, `inds_set_inner`, `inds_set_open` and `inds_set_hyper` are the underlying functions used by `inds(tn; set)`, so they are required for it to work.

!!! todo
    We are looking for a better way to add value-dispatch to [`inds`](@ref) (specifically, the `inds(tn; set)` method) without incurring in huge dynamic dispatch overhead (like the `Val`-dispatch method).
    We might be interested in studying the internal mechanism used in [ValSplit.jl](https://github.com/ztangent/ValSplit.jl).

The following methods are optional but you might be interested on implementing them for performance purposes.

| Method           | Default definition                                           | Brief description                                                 |
| :--------------- | :----------------------------------------------------------- | :---------------------------------------------------------------- |
| `size(tn)`       | Get index sizes from `tensors(tn)`                           | Returns a `Dict` that maps indices to their sizes                 |
| `size(tn, i)`    | Get first matching tensor from `tensors(tn)` and query to it | Returns the size of the given index `i`                           |
| `arrays`         | `parent.(tensors(tn))`                                       | Returns the arrays wrapped by the [`Tensor`](@ref)s in `tn`       |
| `ninds`          | `length(inds(tn))`                                           | Returns the number of indices in `tn`                             |
| `ntensors`       | `length(tensors(tn))`                                        | Returns the number of tensors contained in `tn`                   |
| `in(index, tn)`  | `in(index, inds(tn))`                                        | Returns `true` if `index` is a existing index in `tn`             |
| `in(tensor, tn)` | `in(tensor, tensors(tn))`                                    | Returns `true` if `tensor` is a existing [`Tensor`](@ref) in `tn` |

## [Pluggable](@id man-interface-pluggable)

A [`Pluggable`](@ref man-interface-pluggable) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Site`](@ref)s and open indices.

| Required method           | Brief description                                                                       |
| :------------------------ | :-------------------------------------------------------------------------------------- |
| [`sites(tn)`](@ref sites) | Returns the list of [`Site`](@ref)s present in `tn`                                     |
| `ind_at(tn, at)`          | Return the index linked to the `at` `Symbol`                                            |
| `site_at(tn, at)`         | Return the [`Site`](@ref) linked to the index `at`                                      |
| `inds_set_physical(tn)`   | Return the indices linked to [`Site`](@ref); i.e. the ones behaving as physical indices |

| Method                  | Default definition             | Brief description                                     |
| :---------------------- | :----------------------------- | :---------------------------------------------------- |
| `nsites(tn; kwargs...)` | `length(sites(tn; kwargs...))` | Returns the number of [`Site`](@ref)s present in `tn` |
| `in(site, tn)`          | `in(site, sites(tn))`          | Returns `true` if `site` exists in `tn`               |

!!! danger
    Do not just forward calls to `replace!(tn, index => new_index)` because it would break the mapping between [`Site`](@ref)s and indices when a mapped index is replaced.

## [Ansatz](@id man-interface-ansatz)

A [`Ansatz`](@ref man-interface-ansatz) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Lane`](@ref)s and [`Tensor`](@ref)s.

| Required method           | Brief description                                                                                          |
| :------------------------ | :--------------------------------------------------------------------------------------------------------- |
| [`lanes(tn)`](@ref lanes) | Returns the list of [`Lane`](@ref)s present in `tn`                                                        |
| `tensor_at(tn, at)`       | Returns the [`Tensor`](@ref) linked to the `at` [`Lane`](@ref). Dispatched through `tensors(tn; at::Lane)` |
| `lattice(tn)`             | Returns the [`Lattice`](@ref) associated to `tn`                                                           |

| Method         | Default definition    | Brief description                                     |
| :------------- | :-------------------- | :---------------------------------------------------- |
| `nlanes(tn)`   | `length(lanes(tn))`   | Returns the number of [`Lane`](@ref)s present in `tn` |
| `in(lane, tn)` | `in(lane, lanes(tn))` | Returns `true` if `lane` exists in `tn`               |

!!! danger
    Do not just forward calls to `replace!(tn, index => new_index)` nor `replace!(tn, tensor => new_tensor)` because it would break the mapping between [`Lane`](@ref)s and [`Tensor`](@ref)s when a mapped [`Tensor`] is replaced.
