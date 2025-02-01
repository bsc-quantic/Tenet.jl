using Test
using Tenet
using Tenet: ninds, ntensors

# TensorNetwork interface
function test_tensornetwork(
    tn;
    inds=true,
    ninds=true,
    tensors=true,
    ntensors=true,
    arrays=true,
    size=true,
    inclusion=true,
    replace=true,
    contract=true,
)
    inds && test_tensornetwork_inds(tn)
    ninds && test_tensornetwork_ninds(tn)
    tensors && test_tensornetwork_tensors(tn)
    ntensors && test_tensornetwork_ntensors(tn)
    arrays && test_tensornetwork_arrays(tn)
    size && test_tensornetwork_size(tn)
    inclusion && test_tensornetwork_in(tn)
    replace && test_tensornetwork_replace!(tn)
    contract && test_tensornetwork_contract(tn)
    contract && test_tensornetwork_contract!(tn)
end

function test_tensornetwork_inds(tn)
    # `inds` returns a list of the indices in the Tensor Network
    @test inds(tn) isa AbstractVector{Symbol}

    # `inds(; set = :all)` is equal to naive `inds`
    @test inds(tn; set=:all) == inds(tn)

    # `inds(; set = :open)` returns a list of indices of the Tensor Network
    @test inds(tn; set=:open) isa AbstractVector{Symbol}

    # `inds(; set = :inner)` returns a list of indices of the Tensor Network
    @test inds(tn; set=:inner) isa AbstractVector{Symbol}

    # `inds(; set = :hyper)` returns a list of indices of the Tensor Network
    @test inds(tn; set=:hyper) isa AbstractVector{Symbol}

    # `inds(; parallelto)` returns a list of indices parallel to `i` in the graph
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        inds(tn; parallelto=first(_inds)) isa AbstractVector{Symbol}
    end
end

function test_tensornetwork_ninds(tn)
    # `ninds` returns the number of indices in the Tensor Network
    @test ninds(tn) == length(inds(tn))

    # `ninds(; set = :all)` is equal to naive `ninds`
    @test ninds(tn; set=:all) == ninds(tn)

    # `ninds(; set = :open)` returns the number of open indices in the Tensor Network
    @test ninds(tn; set=:open) == length(inds(tn; set=:open))

    # `ninds(; set = :inner)` returns the number of inner indices in the Tensor Network
    @test ninds(tn; set=:inner) == length(inds(tn; set=:inner))

    # `ninds(; set = :hyper)` returns the number of hyper indices in the Tensor Network
    @test ninds(tn; set=:hyper) == length(inds(tn; set=:hyper))
end

function test_tensornetwork_tensors(tn)
    # `tensors` returns a list of the tensors in the Tensor Network
    @test tensors(tn) isa AbstractVector{<:Tensor}

    # `tensors(; contains = i)` returns a list of tensors containing index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        tensors(tn; contains=first(_inds)) isa AbstractVector{<:Tensor}
    end

    # `tensors(; intersects = i)` returns a list of tensors intersecting index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        tensors(tn; intersects=first(_inds)) isa AbstractVector{<:Tensor}
    end
end

function test_tensornetwork_ntensors(tn)
    #`ntensors` returns the number of tensors in the Tensor Network
    @test ntensors(tn) == length(tensors(tn))

    #`ntensors(; contains = i)` returns the number of tensors containing index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        ntensors(tn; contains=first(_inds)) == length(tensors(tn; contains=first(_inds)))
    end

    #`ntensors(; intersects = i)` returns the number of tensors intersecting index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        ntensors(tn; intersects=first(_inds)) == length(tensors(tn; contains=first(_inds)))
    end
end

function test_tensornetwork_arrays(tn)
    # `arrays` returns a list of the arrays in the Tensor Network
    @test arrays(tn) == parent.(tensors(tn))

    # `arrays(; contains = i)` returns a list of arrays containing index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        arrays(tn; contains=first(_inds)) == parent.(tensors(tn; contains=first(_inds)))
    end

    # `arrays(; intersects = i)` returns a list of arrays intersecting index `i`
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        arrays(tn; intersects=first(_inds)) == parent.(tensors(tn; contains=first(_inds)))
    end
end

function test_tensornetwork_size(tn)
    # `size` returns a mapping from indices to their dimensionalities
    @test size(tn) isa AbstractDict{Symbol,Int}

    # `size` on Symbol returns the dimensionality of that index
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        size(tn, first(_inds)) isa Int
    end
end

function test_tensornetwork_in(tn)
    # `in` on `Symbol` returns if the index is present in the Tensor Network
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        in(first(_inds), tn) == true
    end

    # `in` on `Tensor` returns if that exact object is present in the Tensor Network
    @test let tn = tn
        _tensors = tensors(tn)
        if isempty(_tensors)
            # TODO it should skip here but let's just return true for now
            return true
        end
        in(first(_tensors), tn) == true
    end

    # `in` on copied `Tensor` is never included
    @test let tn = tn
        _tensors = tensors(tn)
        if isempty(_tensors)
            # TODO it should skip here but let's just return true for now
            return true
        end
        in(copy(first(_tensors)), tn) == false
    end
end

function test_tensornetwork_replace!(tn)
    # `replace!` on `Symbol` replaces an index in the Tensor Network
    @test let tn = deepcopy(tn)
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        ind = first(_inds)
        new_ind = gensym(:new)
        replace!(tn, ind => new_ind)
        new_ind ∈ tn
    end

    # `replace!` on `Tensor` replaces a tensor in the Tensor Network
    @test let tn = deepcopy(tn)
        _tensors = tensors(tn)
        if isempty(_tensors)
            # TODO it should skip here but let's just return true for now
            return true
        end
        tensor = first(_tensors)
        new_tensor = copy(tensor)
        replace!(tn, tensor => new_tensor)
        new_tensor ∈ tn
    end
end

function test_tensornetwork_contract(tn)
    # `contract` returns a `Tensor`
    @test contract(tn) isa Tensor
end

function test_tensornetwork_contract!(tn)
    # `contract!` on `Symbol` contracts an index in-place
    @test let tn = deepcopy(tn)
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        ind = first(_inds)
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
    @test sites(tn) isa AbstractVector{<:Site}

    # `sites(; set = :all)` is equal to naive `sites`
    @test sites(tn; set=:all) == sites(tn)

    # `sites(; set = :inputs)` returns a list of input sites (i.e. dual) in the Tensor Network
    @test sites(tn; set=:inputs) isa AbstractVector{<:Site} && all(isdual, sites(tn; set=:inputs))

    # `sites(; set = :outputs)` returns a list of output sites (i.e. non-dual) in the Tensor Network
    @test sites(tn; set=:outputs) isa AbstractVector{<:Site} && all(!isdual, sites(tn; set=:outputs))

    # `sites(; at::Symbol)` returns the site linked to the index
    @test let tn = tn
        _inds = inds(tn)
        if isempty(_inds)
            # TODO it should skip here but let's just return true for now
            return true
        end
        sites(tn; at=first(_inds)) isa Site
    end
end

function test_pluggable_socket(tn)
    # `socket` returns the socket of the Tensor Network
    @test socket(tn) isa Socket
end

function test_pluggable_inds(tn)
    # `inds` returns a list of the indices in the Tensor Network
    @test let tn = tn
        _sites = sites(tn)
        if isempty(_sites)
            # TODO it should skip here but let's just return true for now
            return true
        end
        inds(tn; at=first(_sites)) isa Site
    end
end

function test_pluggable_ninds(tn)
    # `ninds` returns the number of sites in the Tensor Network
    @test nsites(tn) == length(sites(tn))

    # `ninds(; set = :all)` is equal to naive `ninds`
    @test nsites(tn; set=:all) == nsites(tn)

    # `ninds(; set = :inputs)` returns the number of input sites in the Tensor Network
    @test nsites(tn; set=:inputs) == length(sites(tn; set=:inputs))

    # `ninds(; set = :outputs)` returns the number of output sites in the Tensor Network
    @test nsites(tn; set=:outputs) == length(sites(tn; set=:inputs))
end

# Ansatz interface
function test_ansatz(tn)
    test_ansatz_lanes(tn)
    test_ansatz_lattice(tn)
    test_ansatz_tensors(tn)
end

function test_ansatz_lanes(tn)
    # `lanes` returns a list of the lanes in the Tensor Network
    @test lanes(tn) isa AbstractVector{<:Lane}
end

function test_ansatz_lattice(tn)
    # `lattice` returns the lattice of the Tensor Network
    @test lattice(tn) isa Lattice
end

function test_ansatz_tensors(tn)
    # `tensors(; at::Lane)` returns the `Tensor` linked to a `Lane`
    @test tensors(tn; at=first(lanes(tn))) isa Tensor
end
