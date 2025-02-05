# Interfaces

Julia doesn't have a formal definition of interface built into the language. Instead it relies on [duck typing](https://wikipedia.org/wiki/Duck_typing).
Any declaration of a formal interface is then the documentation written for it.

## [TensorNetwork](@id man-interface-tensornetwork) interface

A [TensorNetwork (interface)](@ref man-interface-tensornetwork) is a collection of [`Tensor`](@ref)s forming a graph structure.

| Required method          | Brief description                                   |
| :----------------------- | :-------------------------------------------------- |
| `tensors(tn; kwargs...)` | Returns a list of [`Tensor`](@ref)s present in `tn` |
| `copy(tn)`               | Returns a shallow copy of `tn`                      |

The following methods are optional but you might be interested on implementing them for performance reasons.

| Method                    | Default definition                                           | Brief description                                                 |
| :------------------------ | :----------------------------------------------------------- | :---------------------------------------------------------------- |
| `inds(tn; kwargs...)`     | `mapreduce(inds, ∪, tensors(tn))`                            | Returns a list of indices present in `tn`                         |
| `hasind(tn, ind)`         | `index in inds(tn)`                                          | Returns `true` if `index` is a existing index in `tn`             |
| `hastensor(tn, tensor)`   | `tensor in tensors(tn)`                                      | Returns `true` if `tensor` is a existing [`Tensor`](@ref) in `tn` |
| `size(tn)`                | Get index sizes from `tensors(tn)`                           | Returns a `Dict` that maps indices to their sizes                 |
| `size(tn, i)`             | Get first matching tensor from `tensors(tn)` and query to it | Returns the size of the given index `i`                           |
| `ntensors(tn; kwargs...)` | `length(tensors(tn; kwargs...))`                             | Returns the number of tensors contained in `tn`                   |
| `ninds(tn; kwargs...)`    | `length(inds(tn; kwargs...))`                                | Returns the number of indices in `tn`                             |

### Mutating methods

Tensor Networks are not static entitities. They change.
In order to support mutation, the Tensor Network type needs to implement the following methods.

| Method                               | Brief description                                              |
| :----------------------------------- | :------------------------------------------------------------- |
| `push!(tn, tensor)`                  | Adds a new [`Tensor`](@ref) to `tn`                            |
| `pop!(tn, tensor)`                   | Removes a specific [`Tensor`](@ref) from `tn`                  |
| `replace!(tn, index => new_index)`   | Renames the `index` with `new_index`, if `index` is in `tn`    |
| `replace!(tn, tensor => new_tensor)` | Replace the `tensor` with `new_tensor`, if `tensor` is in `tn` |

!!! warning
    These methods are not forwarded because mutating the `TensorNetwork` can break mappings of composed objects in [Pluggable](@ref man-interface-pluggable) and [Ansatz](@ref man-interface-ansatz).

!!! todo
    We are considering moving to a _effect handling_ system, which would ease tracking mutation on subtypes. In particular the effects we are currently considering are:

    - `ContractIndexEffect` called on `contract!(tn, i)`
    - `ReplaceIndexEffect` called on `replace!(tn, old_index => new_index)`
    - `ReplaceTensorEffect` called on `replace!(tn, old_tensor => new_tensor)`

### Keyword methods

#### `tensors` keyword methods

| Method                       | Default implementation                | Default Brief description                                        |
| :--------------------------- | :------------------------------------ | :--------------------------------------------------------------- |
| `tensors(tn; contains=is)`   | `filter(⊇(is), tensors(tn))`          | Returns the [`Tensor`](@ref)s containing all the indices `is`    |
| `tensors(tn; intersects=is)` | `filter(isdisjoint(is), tensors(tn))` | Returns the [`Tensor`](@ref)s intersecting with the indices `is` |

#### `inds` keyword methods

| Method                 | Brief description                                                                                               |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `inds(tn; set)`        | Return a subset of the indices present in `tn`, where `set` can be one of `:all`, `:open`, `:inner` or `:hyper` |
| `inds(tn; parallelto)` | Return the indices parallel to the index `parallelto`                                                           |

### [WrapsTensorNetwork](@id man-interface-wrapstensornetwork) trait

Many of the types in Tenet work by composing [`TensorNetwork` (type)](@ref TensorNetwork) and all of the optional methods above have a faster implementation for it.
By just forwarding to their [`TensorNetwork` (type)](@ref TensorNetwork) field, wrapper types can accelerate their [TensorNetwork (interface)](@ref man-interface-tensornetwork) methods.

| Required method                    | Default definition | Brief description                                                |
| :--------------------------------- | :----------------- | :--------------------------------------------------------------- |
| `Wraps(::Type{TensorNetwork}, tn)` | `No()`             | Return `Yes()` if `tn` contains a [`TensorNetwork`](@ref) object |
| `TensorNetwork(tn)`                | (_undefined_)      | Return the [`TensorNetwork`](@ref) object wrapped by `tn`        |

## [Pluggable](@id man-interface-pluggable) interface

A [`Pluggable`](@ref man-interface-pluggable) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Site`](@ref)s and open indices.

| Required method | Brief description                                   |
| :-------------- | :-------------------------------------------------- |
| `sites(tn)`     | Returns the list of [`Site`](@ref)s present in `tn` |
| `inds(tn; at)`  | Return the index linked to `at` index               |
| `sites(tn; at)` | Return the [`Site`](@ref) linked to the index `at`  |

As with the interface above, there are optional methods with default implementations that you might be interested in overriding for performance reasons.

| Method                  | Default definition             | Brief description                                                                            |
| :---------------------- | :----------------------------- | :------------------------------------------------------------------------------------------- |
| `inds(tn; plugs)`       | ...                            | Return the indices linked to some [`Site`](@ref); i.e. the ones behaving as physical indices |
| `nsites(tn; kwargs...)` | `length(sites(tn; kwargs...))` | Returns the number of [`Site`](@ref)s present in `tn`                                        |
| `hassite(site, tn)`     | `site in sites(tn))`           | Returns `true` if `site` exists in `tn`                                                      |

### Mutating methods

!!! warning
    If you use directly these methods, you are responsible for leaving the Tensor Network in a coherent state.

| Mutating methods            | Brief description                |
| :-------------------------- | :------------------------------- |
| `addsite!(tn, site => ind)` | Register mapping `site` to `ind` |
| `rmsite!(tn, site)`         | Unregister `site`                |

## [Ansatz](@id man-interface-ansatz) interface

A [`Ansatz`](@ref man-interface-ansatz) is a [`TensorNetwork`](@ref man-interface-tensornetwork) together with a mapping between [`Lane`](@ref)s and [`Tensor`](@ref)s.

| Required method   | Brief description                                                                                          |
| :---------------- | :--------------------------------------------------------------------------------------------------------- |
| `lanes(tn)`       | Returns the list of [`Lane`](@ref)s present in `tn`                                                        |
| `tensors(tn; at)` | Returns the [`Tensor`](@ref) linked to the `at` [`Lane`](@ref). Dispatched through `tensors(tn; at::Lane)` |
| `lattice(tn)`     | Returns the [`Lattice`](@ref) associated to `tn`                                                           |

As in the interfaces defined above, there are optional methods with default definitions which you might be interested on overriding for performance reasons.

| Method             | Default definition  | Brief description                                     |
| :----------------- | :------------------ | :---------------------------------------------------- |
| `nlanes(tn)`       | `length(lanes(tn))` | Returns the number of [`Lane`](@ref)s present in `tn` |
| `haslane(tn,lane)` | `lane in lanes(tn)` | Returns `true` if `lane` exists in `tn`               |

### Mutating methods

!!! warning
    If you use directly these methods, you are responsible for leaving the Tensor Network in a coherent state.

| Method                         | Brief description                           |
| :----------------------------- | :------------------------------------------ |
| `addlane!(tn, lane => tensor)` | Registers the mapping of `lane` to `tensor` |
| `rmlane!(tn, lane)`            | Unregister `lane` from mapping              |
