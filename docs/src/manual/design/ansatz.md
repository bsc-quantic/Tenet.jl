# The `Ansatz` class

```@setup examples
using Tenet
```

## The `Lattice` type

A [`Lattice`](@ref) is a graph that reprents the connectivity pattern between different physical sites.
Its vertices are [`Lane`](@ref)s and wherever an edge exists between two [`Lane`](@ref)s, it means that those two sites can interact locally.
A [`Lattice`](@ref) is constructed by passing a `Graph` and a dictionary mapping [`Lane`](@ref)s to the original graph's vertices:

```@repl examples
using Graphs
graph = path_graph(4)
lattice = Lattice(graph, Dict([lane"1" => 1, lane"2" => 2, lane"3" => 3, lane"4" => lane"4"]))
```

Because the kind of graph topologies we use are quite common, we provide some shortcuts:

```@repl examples
lattice = Lattice(Val(:chain), 4)
lattice = Lattice(Val(:grid), 3, 5)
lattice = Lattice(Val(:lieb), 2, 2)
```

## The `Ansatz` type

A [`Ansatz`](@ref) is a [`Quantum`](@ref) Tensor Network with a fixed graph structure, represented by a [`Lattice`](@ref).

```@repl examples
lattice = Lattice(Val(:chain), 4)
tn = TensorNetwork([
    Tensor(rand(2,2), (:p1, :v12)),
    Tensor(rand(2,2,4), (:p2, :v12, :v23)),
    Tensor(rand(2,4,2), (:p3, :v23, :v34)),
    Tensor(rand(2,2), (:p4, :v34)),
])
qtn = Quantum(tn, Dict([site"1" => :p1, site"2" => :p2, site"3" => :p3, site"4" => :p4]))
ansatz = Ansatz(qtn, lattice)
```

Just as [`Quantum`](@ref) adds information to a [`TensorNetwork`](@ref) on how to map open indices to input/output physical sites, a [`Ansatz`](@ref) teaches a [`Quantum`](@ref) on how to map [`Lattice`](@ref) [`Lane`](@ref)s to [`Tensor`](@ref)s.

```@repl examples
tensors(ansatz; at=lane"1")
```

Because ...

```@repl examples
inds(ansatz; bond=(lane"1", lane"2"))
```

## Time Evolution

[`evolve!`] is a high-level wrapper for different methods used for time-evolution.

!!! note
    In other Tensor Network and Quantum Computing libraries, you may know [`evolve!`](@ref) by the name of `apply!` or `apply`. Given that the word "apply" has many semantic acceptions, we believe that "evolve" fits better to its purpose.

```@repl examples
gate = Gate([0 1; 1 0], [site"1", site"1'"])
evolve!(tn, gate)
```

When applying a multi-site gate, there are different numerical methods that can be used for approximate evolution of states.
As of the time of writing, only the "Simple Update" algorithm is implemented in [`simple_update!`](@ref) but we plan to implement other methods like "Full Update" algorithm in the future.

!!! tip
    [`evolve!`](@ref) is just a wrapper over different numerical methods for evolving states. You're free to call [`simple_update!`](@ref) directly if you want.

!!! warning
    Currently, only 2-site gates are supported.

For example, this is how you would evolve / apply a two-site local operator using both [`evolve!](@ref) and [`simple_update!](@ref):

```@repl examples
evolve!(tn, Gate(rand(2,2,2,2), [site"1", site"2", site"1'", site"2'"]))
simple_update!(tn, Gate(rand(2,2,2,2), [site"1", site"2", site"1'", site"2'"]))
```

As you operate on a [`Ansatz`](@ref), it will make sure that the [`Lattice`](@ref) topology is preserve through different operations. So a two-site [`Gate`](@ref) between non-connected [`Lane`](@ref)s is forbidden:

```@repl examples
evolve!(tn, Gate(rand(2,2,2,2), [site"1", site"4", site"1'", site"4'"])) # this errors
```
