using Graphs: Graphs

# TODO should we use `PartitionedGraph` here?
struct AnsatzMixin
    lattice::Lattice
    lanemap::Dict{Lane,Tensor}
    bondmap::Dict{Bond,Symbol}
end

function AnsatzMixin(lanemap::Dict{Lane,Tensor}, bondmap::Dict{Bond,Symbol})
    graph = Lattice()
    for lane in keys(lanemap)
        Graphs.add_vertex!(graph, lane)
    end
    for bond in keys(bondmap)
        Graphs.add_edge!(graph, bond)
    end
    return AnsatzMixin(graph, lanemap, bondmap)
end

Base.copy(mixin::AnsatzMixin) = AnsatzMixin(copy(mixin.lattice), copy(mixin.lanemap), copy(mixin.bondmap))

trait(::AnsatzInterface, ::AnsatzMixin) = IsAnsatz()

# required methods
lanes(mixin::AnsatzMixin) = sort!(collect(keys(mixin.lanemap)))
bonds(mixin::AnsatzMixin) = collect(keys(mixin.bondmap))

tensors(kwargs::@NamedTuple{at::L}, mixin::AnsatzMixin) where {L<:Lane} = mixin.lanemap[kwargs.at]
inds(kwargs::@NamedTuple{bond::B}, mixin::AnsatzMixin) where {B<:Bond} = mixin.bondmap[kwargs.bond]

# TODO check if renaming `inds(; bond)` to this
inds(kwargs::@NamedTuple{at::B}, tn::AbstractTensorNetwork) where {B<:Bond} = inds((; bond=kwargs.at), tn)

# optional methods
nlanes(mixin::AnsatzMixin) = length(mixin.lanemap)
haslane(mixin::AnsatzMixin, lane) = haskey(mixin.lanemap, lane)

nbonds(mixin::AnsatzMixin) = length(mixin.bondmap)
hasbond(mixin::AnsatzMixin, bond) = haskey(mixin.bondmap, bond)

Graphs.neighbors(mixin::AnsatzMixin, lane::Lane) = Graphs.neighbors(mixin.lattice, lane)
# TODO better way for `neighbors` method on `Bond`?

# mutating methods
function addlane!(mixin::AnsatzMixin, lane::Lane, tensor::Tensor)
    Graphs.add_vertex!(mixin.lattice, lane)
    mixin.lanemap[lane] = tensor
end

function rmlane!(mixin::AnsatzMixin, lane::Lane)
    Graphs.delete_vertex!(mixin.lattice, lane)
    delete!(mixin.lanemap, lane)
end

function addbond!(mixin::AnsatzMixin, bond::Bond, ind::Symbol)
    Graphs.add_edge!(mixin.lattice, bond)
    mixin.bondmap[bond] = ind
end

function rmbond!(mixin::AnsatzMixin, bond::Bond)
    Graphs.delete_edge!(mixin.lattice, bond)
    delete!(mixin.bondmap, bond)
end

# effect handlers
function handle!(mixin::AnsatzMixin, effect::DeleteEffect{Tensor})
    rmlane!(mixin, findfirst(==(effect.f), tn.lanemap))
end

# TODO `findfirst` has a big overhead, specially on `merge` => add a inverse mapping from Tensor to Lane?
function handle!(mixin::AnsatzMixin, effect::ReplaceEffect{Pair{Tensor,Tensor}})
    lane = findfirst(==(effect.f.first), mixin.lanemap)
    !isnothing(lane) || return nothing
    mixin.lanemap[lane] = effect.f.second
end

# other useful methods
Lattice(x::AnsatzMixin) = x.lattice
hastensor(mixin::AnsatzMixin, tensor) = any(x -> x === tensor, values(mixin.lanemap))
