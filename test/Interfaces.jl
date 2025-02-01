using Test
using Tenet
using Tenet: ninds, ntensors

# TensorNetwork interface
function test_tensornetwork(tn)
    test_tensornetwork_inds(tn)
    test_tensornetwork_ninds(tn)
    test_tensornetwork_tensors(tn)
    test_tensornetwork_ntensors(tn)
    test_tensornetwork_arrays(tn)
    test_tensornetwork_size(tn)
    test_tensornetwork_in(tn)
    test_tensornetwork_replace!(tn)
    test_tensornetwork_contract(tn)
    test_tensornetwork_contract!(tn)
end

function test_tensornetwork_inds(tn)
    # `inds` returns a list of the indices in the Tensor Network
    @testimpl inds(tn) isa AbstractVector{Symbol}

    # `inds(; set = :all)` is equal to naive `inds`
    @testimpl inds(tn; set=:all) == inds(tn)

    # `inds(; set = :open)` returns a list of indices of the Tensor Network
    @testimpl inds(tn; set=:open) isa AbstractVector{Symbol}

    # `inds(; set = :inner)` returns a list of indices of the Tensor Network
    @testimpl inds(tn; set=:inner) isa AbstractVector{Symbol}

    # `inds(; set = :hyper)` returns a list of indices of the Tensor Network
    @testimpl inds(tn; set=:hyper) isa AbstractVector{Symbol}

    # `inds(; parallelto)` returns a list of indices parallel to `i` in the graph
    @testimpl inds(tn; parallelto=first(inds(tn))) isa AbstractVector{Symbol}
end

function test_tensornetwork_ninds(tn)
    # `ninds` returns the number of indices in the Tensor Network
    @testimpl ninds(tn) == length(inds(tn))

    # `ninds(; set = :all)` is equal to naive `ninds`
    @testimpl ninds(tn; set=:all) == ninds(tn)

    # `ninds(; set = :open)` returns the number of open indices in the Tensor Network
    @testimpl ninds(tn; set=:open) == length(inds(tn; set=:open))

    # `ninds(; set = :inner)` returns the number of inner indices in the Tensor Network
    @testimpl ninds(tn; set=:inner) == length(inds(tn; set=:inner))

    # `ninds(; set = :hyper)` returns the number of hyper indices in the Tensor Network
    @testimpl ninds(tn; set=:hyper) == length(inds(tn; set=:hyper))
end

function test_tensornetwork_tensors(tn)
    # `tensors` returns a list of the tensors in the Tensor Network
    @testimpl tensors(tn) isa AbstractVector{<:Tensor}

    # `tensors(; contains = i)` returns a list of tensors containing index `i`
    @testimpl tensors(tn; contains=first(inds(tn))) isa AbstractVector{<:Tensor}

    # `tensors(; intersects = i)` returns a list of tensors intersecting index `i`
    @testimpl tensors(tn; intersects=first(inds(tn))) isa AbstractVector{<:Tensor}
end

function test_tensornetwork_ntensors(tn)
    #`ntensors` returns the number of tensors in the Tensor Network
    @testimpl ntensors(tn) == length(tensors(tn))

    #`ntensors(; contains = i)` returns the number of tensors containing index `i`
    @testimpl ntensors(tn; contains=first(inds(tn))) == length(tensors(tn; contains=first(inds(tn))))

    #`ntensors(; intersects = i)` returns the number of tensors intersecting index `i`
    @testimpl ntensors(tn; intersects=first(inds(tn))) == length(tensors(tn; contains=first(inds(tn))))
end

function test_tensornetwork_arrays(tn)
    # `arrays` returns a list of the arrays in the Tensor Network
    @testimpl arrays(tn) == parent.(tensors(tn))

    # `arrays(; contains = i)` returns a list of arrays containing index `i`
    @testimpl arrays(tn; contains=first(inds(tn))) == parent.(tensors(tn; contains=first(inds(tn))))

    # `arrays(; intersects = i)` returns a list of arrays intersecting index `i`
    @testimpl arrays(tn; intersects=first(inds(tn))) == parent.(tensors(tn; contains=first(inds(tn))))
end

function test_tensornetwork_size(tn)
    # `size` returns a mapping from indices to their dimensionalities
    @testimpl size(tn) isa AbstractDict{Symbol,Int}

    # `size` on Symbol returns the dimensionality of that index
    @testimpl size(tn, first(inds(tn))) isa Int
end

function test_tensornetwork_in(tn)
    # `in` on `Symbol` returns if the index is present in the Tensor Network
    @testimpl in(first(inds(tn)), tn) == true

    # `in` on `Tensor` returns if that exact object is present in the Tensor Network
    @testimpl in(first(tensors(tn)), tn) == true

    # `in` on copied `Tensor` is never included
    @testimpl in(copy(first(tensors(tn))), tn) == false
end

function test_tensornetwork_replace!(tn)
    # `replace!` on `Symbol` replaces an index in the Tensor Network
    @testimpl let tn = deepcopy(tn)
        ind = first(inds(tn))
        new_ind = gensym(:new)
        replace!(tn, ind => new_ind)
        new_ind ∈ tn
    end

    # `replace!` on `Tensor` replaces a tensor in the Tensor Network
    @testimpl let tn = deepcopy(tn)
        tensor = first(tensors(tn))
        new_tensor = copy(tensor)
        replace!(tn, tensor => new_tensor)
        new_tensor ∈ tn
    end
end

function test_tensornetwork_contract(tn)
    # `contract` returns a `Tensor`
    @testimpl contract(tn) isa Tensor
end

function test_tensornetwork_contract!(tn)
    # `contract!` on `Symbol` contracts an index in-place
    @testimpl let tn = deepcopy(tn)
        ind = first(inds(tn))
        contract!(tn, ind)
        ind ∉ tn
    end
end

# Pluggable interface
function test_pluggable(tn)
    test_pluggable_sites(tn)
    test_pluggable_socket(tn)
    test_pluggable_inds(tn)
    test_pluggable_ninds(tn)
end

function test_pluggable_sites(tn)
    # `sites` returns a list of the sites in the Tensor Network
    @testimpl sites(tn) isa AbstractVector{<:Site}

    # `sites(; set = :all)` is equal to naive `sites`
    @testimpl sites(tn; set=:all) == sites(tn)

    # `sites(; set = :inputs)` returns a list of input sites (i.e. dual) in the Tensor Network
    @testimpl sites(tn; set=:inputs) isa AbstractVector{<:Site} && all(isdual, sites(tn; set=:inputs))

    # `sites(; set = :outputs)` returns a list of output sites (i.e. non-dual) in the Tensor Network
    @testimpl sites(tn; set=:outputs) isa AbstractVector{<:Site} && all(!isdual, sites(tn; set=:outputs))

    # `sites(; at::Symbol)` returns the site linked to the index
    @testimpl sites(tn; at=first(inds(tn))) isa Site
end

function test_pluggable_socket(tn)
    # `socket` returns the socket of the Tensor Network
    @testimpl socket(tn) isa Socket
end

function test_pluggable_inds(tn)
    # `inds` returns a list of the indices in the Tensor Network
    @testimpl inds(tn; at=first(sites(tn))) isa Site
end

function test_pluggable_ninds(tn)
    # `ninds` returns the number of sites in the Tensor Network
    @testimpl nsites(tn) == length(sites(tn))

    # `ninds(; set = :all)` is equal to naive `ninds`
    @testimpl nsites(tn; set=:all) == nsites(tn)

    # `ninds(; set = :inputs)` returns the number of input sites in the Tensor Network
    @testimpl nsites(tn; set=:inputs) == length(sites(tn; set=:inputs))

    # `ninds(; set = :outputs)` returns the number of output sites in the Tensor Network
    @testimpl nsites(tn; set=:outputs) == length(sites(tn; set=:inputs))
end

# Ansatz interface
function test_ansatz(tn)
    test_ansatz_lanes(tn)
    test_ansatz_lattice(tn)
    test_ansatz_tensors(tn)
end

function test_ansatz_lanes(tn)
    # `lanes` returns a list of the lanes in the Tensor Network
    @testimpl lanes(tn) isa AbstractVector{<:Lane}
end

function test_ansatz_lattice(tn)
    # `lattice` returns the lattice of the Tensor Network
    @testimpl lattice(tn) isa Lattice
end

function test_ansatz_tensors(tn)
    # `tensors(; at::Lane)` returns the `Tensor` linked to a `Lane`
    @testimpl tensors(tn; at=first(lanes(tn))) isa Tensor
end
