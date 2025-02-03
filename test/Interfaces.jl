using Test
using Tenet
using Tenet: ninds, ntensors, nsites, lattice

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
    contract_mut=true,
)
    @testset "TensorNetwork interface" begin
        inds && test_tensornetwork_inds(tn)
        ninds && test_tensornetwork_ninds(tn)
        tensors && test_tensornetwork_tensors(tn)
        ntensors && test_tensornetwork_ntensors(tn)
        arrays && test_tensornetwork_arrays(tn)
        size && test_tensornetwork_size(tn)
        inclusion && test_tensornetwork_in(tn)
        replace && test_tensornetwork_replace!(tn)
        contract && test_tensornetwork_contract(tn)
        contract_mut && test_tensornetwork_contract!(tn)
    end
end

function test_tensornetwork_inds(tn)
    @testset "`inds` returns a list of the indices in the Tensor Network" begin
        @test inds(tn) isa Base.AbstractVecOrTuple{Symbol}
    end

    @testset "`inds(; set = :all)` is equal to naive `inds`" begin
        @test inds(tn; set=:all) == inds(tn)
    end

    @testset "`inds(; set = :open)` returns a list of indices of the Tensor Network" begin
        @test inds(tn; set=:open) isa Base.AbstractVecOrTuple{Symbol}
    end

    @testset "`inds(; set = :inner)` returns a list of indices of the Tensor Network" begin
        @test inds(tn; set=:inner) isa Base.AbstractVecOrTuple{Symbol}
    end

    @testset "`inds(; set = :hyper)` returns a list of indices of the Tensor Network" begin
        @test inds(tn; set=:hyper) isa Base.AbstractVecOrTuple{Symbol}
    end

    @testset "`inds(; parallelto)` returns a list of indices parallel to `i` in the graph" begin
        @testif pred = !isempty(inds(tn)) inds(tn; parallelto=first(inds(tn))) isa Base.AbstractVecOrTuple{Symbol}
    end
end

function test_tensornetwork_ninds(tn)
    @testset "`ninds` returns the number of indices in the Tensor Network" begin
        @test ninds(tn) == length(inds(tn))
    end

    @testset "`ninds(; set = :all)` is equal to naive `ninds`" begin
        @test ninds(tn; set=:all) == ninds(tn)
    end

    @testset "`ninds(; set = :open)` returns the number of open indices in the Tensor Network" begin
        @test ninds(tn; set=:open) == length(inds(tn; set=:open))
    end

    @testset "`ninds(; set = :inner)` returns the number of inner indices in the Tensor Network" begin
        @test ninds(tn; set=:inner) == length(inds(tn; set=:inner))
    end

    @testset "`ninds(; set = :hyper)` returns the number of hyper indices in the Tensor Network" begin
        @test ninds(tn; set=:hyper) == length(inds(tn; set=:hyper))
    end
end

function test_tensornetwork_tensors(tn)
    @testset "`tensors` returns a list of the tensors in the Tensor Network" begin
        @test tensors(tn) isa Base.AbstractVecOrTuple{<:Tensor}
    end

    @testset "`tensors(; contains = i)` returns a list of tensors containing index `i`" begin
        @testif pred = !isempty(inds(tn)) tensors(tn; contains=first(inds(tn))) isa Base.AbstractVecOrTuple{<:Tensor}
    end

    @testset "`tensors(; intersects = i)` returns a list of tensors intersecting index `i`" begin
        @testif pred = !isempty(inds(tn)) tensors(tn; intersects=first(inds(tn))) isa Base.AbstractVecOrTuple{<:Tensor}
    end
end

function test_tensornetwork_ntensors(tn)
    @testset "`ntensors` returns the number of tensors in the Tensor Network" begin
        @test ntensors(tn) == length(tensors(tn))
    end

    @testset "`ntensors(; contains = i)` returns the number of tensors containing index `i`" begin
        @testif pred = !isempty(inds(tn)) ntensors(tn; contains=first(inds(tn))) ==
            length(tensors(tn; contains=first(inds(tn))))
    end

    @testset "`ntensors(; intersects = i)` returns the number of tensors intersecting index `i`" begin
        @testif pred = !isempty(inds(tn)) ntensors(tn; intersects=first(inds(tn))) ==
            length(tensors(tn; contains=first(inds(tn))))
    end
end

function test_tensornetwork_arrays(tn)
    @testset "`arrays` returns a list of the arrays in the Tensor Network" begin
        @test arrays(tn) == parent.(tensors(tn))
    end

    @testset "`arrays(; contains = i)` returns a list of arrays containing index `i`" begin
        @testif pred = !isempty(inds(tn)) arrays(tn; contains=first(inds(tn))) ==
            parent.(tensors(tn; contains=first(inds(tn))))
    end

    @testset "`arrays(; intersects = i)` returns a list of arrays intersecting index `i`" begin
        @testif pred = !isempty(inds(tn)) arrays(tn; intersects=first(inds(tn))) ==
            parent.(tensors(tn; contains=first(inds(tn))))
    end
end

function test_tensornetwork_size(tn)
    @testset "`size` returns a mapping from indices to their dimensionalities" begin
        @test size(tn) isa AbstractDict{Symbol,Int}
    end

    @testset "`size` on Symbol returns the dimensionality of that index" begin
        @testif pred = !isempty(inds(tn)) size(tn, first(inds(tn))) isa Int
    end
end

function test_tensornetwork_in(tn)
    @testset "`in` on `Symbol` returns if the index is present in the Tensor Network" begin
        @testif pred = !isempty(inds(tn)) first(inds(tn)) ∈ tn
    end

    @testset "`in` on `Tensor` returns if that exact object is present in the Tensor Network" begin
        @testif pred = !isempty(tensors(tn)) first(tensors(tn)) ∈ tn
    end

    @testset "`in` on copied `Tensor` is never included" begin
        @testif pred = !isempty(tensors(tn)) copy(first(tensors(tn))) ∉ tn
    end
end

function test_tensornetwork_replace!(tn)
    @testset "`replace!` on `Symbol` replaces an index in the Tensor Network" begin
        @testif pred = !isempty(inds(tn)) let tn = deepcopy(tn)
            ind = first(inds(tn))
            new_ind = gensym(:new)
            replace!(tn, ind => new_ind)
            new_ind ∈ tn
        end
    end

    @testset "`replace!` on `Tensor` replaces a tensor in the Tensor Network" begin
        @testif pred = !isempty(tensors(tn)) let tn = deepcopy(tn)
            tensor = first(tensors(tn))
            new_tensor = copy(tensor)
            replace!(tn, tensor => new_tensor)
            new_tensor ∈ tn
        end
    end
end

function test_tensornetwork_contract(tn)
    @testset "`contract` returns a `Tensor`" begin
        @testif pred = !isempty(tensors(tn)) contract(tn) isa Tensor
    end
end

function test_tensornetwork_contract!(tn)
    @testset "`contract!` on `Symbol` contracts an index in-place" begin
        @testif pred = !isempty(inds(tn; set=:inner)) let tn = deepcopy(tn)
            ind = first(inds(tn; set=:inner))
            contract!(tn, ind)
            ind ∉ tn
        end
    end
end

# Pluggable interface
function test_pluggable(tn; sites=true, socket=true, inds=true, ninds=true)
    @testset "Pluggable interface" begin
        sites && test_pluggable_sites(tn)
        socket && test_pluggable_socket(tn)
        inds && test_pluggable_inds(tn)
        ninds && test_pluggable_nsites(tn)
    end
end

function test_pluggable_sites(tn)
    @testset "`sites` returns a list of the sites in the Tensor Network" begin
        @test sites(tn) isa Base.AbstractVecOrTuple{<:Site}
    end

    @testset "`sites(; set = :all)` is equal to naive `sites`" begin
        @test sites(tn; set=:all) == sites(tn)
    end

    @testset "`sites(; set = :inputs)` returns a list of input sites (i.e. dual) in the Tensor Network" begin
        @test sites(tn; set=:inputs) isa Base.AbstractVecOrTuple{<:Site} && all(isdual, sites(tn; set=:inputs))
    end

    @testset "`sites(; set = :outputs)` returns a list of output sites (i.e. non-dual) in the Tensor Network" begin
        @test sites(tn; set=:outputs) isa Base.AbstractVecOrTuple{<:Site} && all(!isdual, sites(tn; set=:outputs))
    end

    @testset "`sites(; at::Symbol)` returns the site linked to the index" begin
        @testif pred = !isempty(inds(tn)) && !isempty(sites(tn)) sites(tn; at=first(inds(tn))) isa Union{Nothing,Site}
    end
end

function test_pluggable_socket(tn)
    @testset "`socket` returns the socket of the Tensor Network" begin
        @test socket(tn) isa Tenet.Socket
    end
end

function test_pluggable_inds(tn)
    @testset "`inds(; at::Site)` returns the index linked to the `Site`" begin
        @testif pred = !isempty(sites(tn)) inds(tn; at=first(sites(tn))) isa Symbol
    end
end

function test_pluggable_nsites(tn)
    @testset "`nsites` returns the number of sites in the Tensor Network" begin
        @test nsites(tn) == length(sites(tn))
    end

    @testset "`nsites(; set = :all)` is equal to naive `nsites`" begin
        @test nsites(tn; set=:all) == nsites(tn)
    end

    @testset "`nsites(; set = :inputs)` returns the number of input sites in the Tensor Network" begin
        @test nsites(tn; set=:inputs) == length(sites(tn; set=:inputs))
    end

    @testset "`nsites(; set = :outputs)` returns the number of output sites in the Tensor Network" begin
        @test nsites(tn; set=:outputs) == length(sites(tn; set=:outputs))
    end
end

# Ansatz interface
function test_ansatz(tn; lanes=true, lattice=true, tensors=true)
    @testset "Ansatz interface" begin
        lanes && test_ansatz_lanes(tn)
        lattice && test_ansatz_lattice(tn)
        tensors && test_ansatz_tensors(tn)
    end
end

function test_ansatz_lanes(tn)
    @testset "`lanes` returns a list of the lanes in the Tensor Network" begin
        @test lanes(tn) isa Base.AbstractVecOrTuple{<:Lane}
    end
end

function test_ansatz_lattice(tn)
    @testset "`lattice` returns the lattice of the Tensor Network" begin
        @test lattice(tn) isa Lattice
    end
end

function test_ansatz_tensors(tn)
    @testset "`tensors(; at::Lane)` returns the `Tensor` linked to a `Lane`" begin
        @test tensors(tn; at=first(lanes(tn))) isa Tensor
    end
end
