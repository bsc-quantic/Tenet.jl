using Interfaces

@interface TensorNetworkInterface AbstractTensorNetwork (
    mandatory=(
        inds=(
            "`inds` returns a list of the indices in the Tensor Network" => x -> inds(x) isa AbstractVector{Symbol},
            "`inds(; set = :all)` is equal to naive `inds`" => x -> inds(x; set=:all) == inds(x),
            "`inds(; set = :open)` returns a list of indices of the Tensor Network" =>
                x -> inds(x; set=:open) isa AbstractVector{Symbol},
            "`inds(; set = :inner)` returns a list of indices of the Tensor Network" =>
                x -> inds(x; set=:inner) isa AbstractVector{Symbol},
            "`inds(; set = :hyper)` returns a list of indices of the Tensor Network" =>
                x -> inds(x; set=:hyper) isa AbstractVector{Symbol},
            "`inds(; parallelto)` returns a list of indices parallel to `i` in the graph" =>
                x -> inds(x; parallelto=first(inds(x))) isa AbstractVector{Symbol},
        ),
        tensors=(
            "`tensors` returns a list of the tensors in the Tensor Network" =>
                x -> tensors(x) isa AbstractVector{<:Tensor},
            "`tensors(; contains = i)` returns a list of tensors containing index `i`" =>
                x -> tensors(x; contains=first(inds(x))) isa AbstractVector{<:Tensor},
            "`tensors(; intersects = i)` returns a list of tensors intersecting index `i`" =>
                x -> tensors(x; intersects=first(inds(x))) isa AbstractVector{<:Tensor},
        ),
        size=(
            "`size` returns a mapping from indices to their dimensionalities" =>
                x -> size(x) isa AbstractDict{Symbol,Int},
            "`size` on Symbol returns the dimensionality of that index" => x -> size(x, first(inds(x))) isa Int,
        ),
        # default implementations work but these check if the user has overridden them and work
        ninds=(
            "`ninds` returns the number of indices in the Tensor Network" => x -> ninds(x) == length(inds(x)),
            "`ninds(; set = :all)` is equal to naive `ninds`" => x -> ninds(x; set=:all) == ninds(x),
            "`ninds(; set = :open)` returns the number of open indices in the Tensor Network" =>
                x -> ninds(x; set=:open) == length(inds(x; set=:open)),
            "`ninds(; set = :inner)` returns the number of inner indices in the Tensor Network" =>
                x -> ninds(x; set=:inner) == length(inds(x; set=:inner)),
            "`ninds(; set = :hyper)` returns the number of hyper indices in the Tensor Network" =>
                x -> ninds(x; set=:hyper) == length(inds(x; set=:hyper)),
        ),
        ntensors=(
            "`ntensors` returns the number of tensors in the Tensor Network" => x -> ntensors(x) == length(tensors(x)),
            "`ntensors(; contains = i)` returns the number of tensors containing index `i`" =>
                x -> ntensors(x; contains=first(inds(x))) == length(tensors(x; contains=first(inds(x)))),
            "`ntensors(; intersects = i)` returns the number of tensors intersecting index `i`" =>
                x -> ntensors(x; intersects=first(inds(x))) == length(tensors(x; contains=first(inds(x)))),
        ),
        arrays=(
            "`arrays` returns a list of the arrays in the Tensor Network" => x -> arrays(x) == parent.(tensors(x)),
            "`arrays(; contains = i)` returns a list of arrays containing index `i`" =>
                x -> arrays(x; contains=first(inds(x))) == parent.(tensors(x; contains=first(inds(x)))),
            "`arrays(; intersects = i)` returns a list of arrays intersecting index `i`" =>
                x -> arrays(x; intersects=first(inds(x))) == parent.(tensors(x; contains=first(inds(x)))),
        ),
        inclusion=(
            "`in` on `Symbol` returns if the index is present in the Tensor Network" =>
                x -> in(first(inds(x)), x) == true,
            "`in` on `Tensor` returns if that exact object is present in the Tensor Network" =>
                x -> in(first(tensors(x)), x) == true,
            "`in` on copied `Tensor` is never included" => x -> in(copy(first(tensors(x))), x) == false,
        ),
    ),
    optional=(collect=(), (push!)=(), (pop!)=(), (replace!)=(), contract=(), (contract!)=()),
) "AbstractTensorNetwork interface"

@interface PluggableInterface AbstractTensorNetwork (
    mandatory=(
        sites=(
            "`sites` returns a list of the sites of the Tensor Network" => x -> sites(x) isa AbstractVector{<:Site},
            "`sites(; set=:all)` is equal to naive `sites`" => x -> sites(x; set=:all) == sites(x),
            "`sites(; set=:inputs)` returns a list of the input sites of the Tensor Network" =>
                x -> sites(x; set=:inputs) isa AbstractVector{<:Site} && all(isdual, sites(x; set=:inputs)),
            "`sites(; set=:outputs)` returns a list of the output sites of the Tensor Network" =>
                x -> sites(x; set=:outputs) isa AbstractVector{<:Site} && all(!isdual, sites(x; set=:outputs)),
            "`sites(; at::Symbol)` returns the `Site` associated to a index (if any)" =>
                x -> sites(x; at=inds(x; at=first(sites(x)))) == first(sites(x)),
        ),
        socket="`socket` returns the socket of the Tensor Network" => x -> socket(x) isa Union{State,Operator},
        inds_at="`inds(; at::Site)` returns the index associated with a `Site`" =>
            x -> inds(x; at=first(sites(x))) isa Symbol,
        # default implementations work but these check if the user has overridden them and work
        nsites=("`nsites` returns the number of sites in the Tensor Network" => x -> nsites(x) == length(sites(x)),),
    ),
    optional=(),
) "The Pluggable Tensor Network interface"

@interface AnsatzInterface AbstractTensorNetwork (
    mandatory=(
        lanes="`lanes` returns a list of the lanes of the Tensor Network" => x -> lanes(x) isa AbstractVector{<:Lane},
        lattice="`lattice` returns the lattice of the Tensor Network" => x -> lattice(x) isa Lattice,
        tensors_at="`tensors(; at::Lane)` returns a list of the tensors at the lane" =>
            x -> tensors(x; at=first(lanes(x))) isa AbstractVector{Tensor},
    ),
    optional=(),
) "The Ansatz Tensor Network interface"

@interface QuantumInterface AbstractTensorNetwork (
    mandatory=(
        pluggable="The Quantum interface is a Pluggable interface" => x -> Interfaces.test(PluggableInterface, x),
        ansatz="The Quantum interface is an Ansatz interface" => x -> Interfaces.test(AnsatzInterface, x),
    ),
    optional=(),
) "The Quantum Tensor Network interface"

@implements TensorNetworkInterface TensorNetwork [
    TensorNetwork([
        Tensor(zeros(2, 2), (:i, :j)), Tensor(zeros(2, 2), (:j, :k)), Tensor(zeros(2), (:k,)), Tensor(zeros(2), (:k,))
    ]),
]

@implements AnsatzInterface Ansatz []

@implements PluggableInterface Quantum []
@implements PluggableInterface Circuit []

@implements QuantumInterface Product []
@implements QuantumInterface MPS []
@implements QuantumInterface MPO []
