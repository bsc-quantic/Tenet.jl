# Interfaces

Julia doesn't have a formal definition of interface built into the language. Instead it relies on [duck typing](https://wikipedia.org/wiki/Duck_typing).
Any declaration of a formal interface is then the documentation written for it.

## [TensorNetwork](@id man-interface-tensornetwork) interface

A [TensorNetwork (interface)](@ref man-interface-tensornetwork) is a collection of [`Tensor`](@ref)s forming a graph structure.

| Required method                      | Brief description                                              |
| :----------------------------------- | :------------------------------------------------------------- |
| `tensors(tn; kwargs...)`             | Returns a list of [`Tensor`](@ref)s present in `tn`            |
| `copy(tn)`                           | Returns a shallow copy of `tn`                                 |
| `replace!(tn, index => new_index)`   | Renames the `index` with `new_index`, if `index` is in `tn`    |
| `replace!(tn, tensor => new_tensor)` | Replace the `tensor` with `new_tensor`, if `tensor` is in `tn` |

The following methods are optional but you might be interested on implementing them for performance reasons.

| Method                    | Default definition                                           | Brief description                                                 |
| :------------------------ | :----------------------------------------------------------- | :---------------------------------------------------------------- |
| `inds(tn; kwargs...)`     | `mapreduce(inds, ∪, tensors(tn))`                            | Returns a list of indices present in `tn`                         |
| `hasind(tn, ind)`         | `in(index, inds(tn))`                                        | Returns `true` if `index` is a existing index in `tn`             |
| `hastensor(tn, tensor)`   | `in(tensor, tensors(tn))`                                    | Returns `true` if `tensor` is a existing [`Tensor`](@ref) in `tn` |
| `size(tn)`                | Get index sizes from `tensors(tn)`                           | Returns a `Dict` that maps indices to their sizes                 |
| `size(tn, i)`             | Get first matching tensor from `tensors(tn)` and query to it | Returns the size of the given index `i`                           |
| `ntensors(tn; kwargs...)` | `length(tensors(tn; kwargs...))`                             | Returns the number of tensors contained in `tn`                   |
| `ninds(tn; kwargs...)`    | `length(inds(tn; kwargs...))`                                | Returns the number of indices in `tn`                             |

### [WrapsTensorNetwork](@id man-interface-wrapstensornetwork) trait

Many of the types in Tenet work by composing [`TensorNetwork` (type)](@ref TensorNetwork) and all of the optional methods above have a faster implementation for it.
By just forwarding to their [`TensorNetwork` (type)](@ref TensorNetwork) field, wrapper types can accelerate their [TensorNetwork (interface)](@ref man-interface-tensornetwork) methods.

| Required method                    | Default definition | Brief description                                                |
| :--------------------------------- | :----------------- | :--------------------------------------------------------------- |
| `Wraps(::Type{TensorNetwork}, tn)` | `No()`             | Return `Yes()` if `tn` contains a [`TensorNetwork`](@ref) object |
| `TensorNetwork(tn)`                | (_undefined_)      | Return the [`TensorNetwork`](@ref) object wrapped by `tn`        |

### `tensors` keyword methods

| Method                       | Default implementation                | Default Brief description                                         |
| :--------------------------- | :------------------------------------ | :---------------------------------------------------------------- |
| `tensors(tn; contains=is)`   | `filter(⊇(is), tensors(tn))`          | Returns the [`Tensor`](@ref)s containing at all the indices `is`. |
| `tensors(tn; intersects=is)` | `filter(isdisjoint(is), tensors(tn))` | Returns the [`Tensor`](@ref)s intersecting with the indices `is`. |

### `inds` keyword methods

| Method                 | Brief description                                                                                          |
| :--------------------- | :--------------------------------------------------------------------------------------------------------- |
| `inds(tn; set)`        | Return a subset of the indices present in `tn`. `set` can be one of `:all`, `:open`, `:inner` or `:hyper`. |
| `inds(tn; parallelto)` | Return the indices parallel to the index `parallelto`.                                                     |

## [Pluggable](@id man-interface-pluggable) interface

A [`Pluggable`](@ref man-interface-pluggable) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Site`](@ref)s and open indices.

| Required method  | Brief description                                   |
| :--------------- | :-------------------------------------------------- |
| `sites(tn)`      | Returns the list of [`Site`](@ref)s present in `tn` |
| `indat(tn, at)`  | Return the index linked to the `at` `Symbol`        |
| `siteat(tn, at)` | Return the [`Site`](@ref) linked to the index `at`  |

| Method                  | Default definition                      | Brief description                                                                       |
| :---------------------- | :-------------------------------------- | :-------------------------------------------------------------------------------------- |
| `inds_set_physical(tn)` | `map(at -> site_at(tn, at), sites(tn))` | Return the indices linked to [`Site`](@ref); i.e. the ones behaving as physical indices |
| `nsites(tn; kwargs...)` | `length(sites(tn; kwargs...))`          | Returns the number of [`Site`](@ref)s present in `tn`                                   |
| `hassite(site, tn)`     | `in(site, sites(tn))`                   | Returns `true` if `site` exists in `tn`                                                 |

!!! danger
    Do not just forward calls to `replace!(tn, index => new_index)` because it would break the mapping between [`Site`](@ref)s and indices when a mapped index is replaced.

## [Ansatz](@id man-interface-ansatz) interface

A [`Ansatz`](@ref man-interface-ansatz) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Lane`](@ref)s and [`Tensor`](@ref)s.

| Required method    | Brief description                                                                                          |
| :----------------- | :--------------------------------------------------------------------------------------------------------- |
| `lanes(tn)`        | Returns the list of [`Lane`](@ref)s present in `tn`                                                        |
| `tensorat(tn, at)` | Returns the [`Tensor`](@ref) linked to the `at` [`Lane`](@ref). Dispatched through `tensors(tn; at::Lane)` |
| `lattice(tn)`      | Returns the [`Lattice`](@ref) associated to `tn`                                                           |

| Method       | Default definition  | Brief description                                     |
| :----------- | :------------------ | :---------------------------------------------------- |
| `nlanes(tn)` | `length(lanes(tn))` | Returns the number of [`Lane`](@ref)s present in `tn` |
| `haslane`    | ...                 | Returns `true` if `lane` exists in `tn`               |

!!! danger
    Do not just forward calls to `replace!(tn, index => new_index)` nor `replace!(tn, tensor => new_tensor)` because it would break the mapping between [`Lane`](@ref)s and [`Tensor`](@ref)s when a mapped [`Tensor`] is replaced.
