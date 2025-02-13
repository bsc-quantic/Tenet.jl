using Graphs: add_edge!, add_vertex!

# TODO should we use `PartitionedGraph` here?
struct AnsatzMixin
    lattice::Latice
    lanemap::Dict{Lane,Tensor}
    bondmap::Dict{Bond,Symbol}
end

Lattice(x::AnsatzMixin) = x.lattice

function AnsatzMixin(lanemap::Dict{Lane,Tensor}, bondmap::Dict{Bond,Symbol})
    graph = Lattice()
    for lane in keys(mixin.lanemap)
        add_vertex!(graph, lane)
    end
    for bond in keys(mixin.bondmap)
        add_edge!(graph, bond)
    end
    return AnsatzMixin(graph, lanemap, bondmap)
end

hastensor(mixin::AnsatzMixin, tensor) = tensor ∈ values(mixin.lanemap)

# required methods
lanes(mixin::AnsatzMixin) = sort!(collect(keys(mixin.lanemap)))
bonds(mixin::AnsatzMixin) = collect(keys(mixin.bondmap))

tensors(kwargs::@NamedTuple{at::L}, mixin::AnsatzMixin) where {L<:Lane} = mixin.lanemap[at]
inds(kwargs::@NamedTuple{bond::B}, mixin::AnsatzMixin) where {B<:Bond} = mixin.bondmap[bond]

# TODO check if renaming `inds(; bond)` to this
inds(kwargs::@NamedTuple{at::B}, tn::AbstractTensorNetwork) where {B<:Bond} = inds((; bond=kwargs.at), tn)

# optional methods
nlanes(mixin::AnsatzMixin) = length(mixin.lanemap)
haslane(mixin::AnsatzMixin, lane) = haskey(mixin.lanemap, lane)

nbonds(mixin::AnsatzMixin) = length(mixin.bondmap)
hasbond(mixin::AnsatzMixin, bond) = haskey(mixin.bondmap, bond)

Base.neighbors(mixin::AnsatzMixin, lane::Lane) = neighbors(mixin.lattice, lane)
# TODO better way for `neighbors` method on `Bond`?

# mutating methods
function addlane!(mixin::AnsatzMixin, p::Pair{<:Lane,Tensor})
    add_vertex!(mixin.lattice, p.first)
    mixin.lanemap[p.first] = p.second
end

function rmlane!(mixin::AnsatzMixin, lane::Lane)
    delete_vertex!(mixin.lattice, lane)
    delete!(mixin.lanemap, lane)
end

function addbond!(mixin::AnsatzMixin, p::Pair{<:Bond,Symbol})
    add_edge!(mixin.lattice, p.first)
    mixin.bondmap[p.first] = p.second
end

function rmbond!(mixin::AnsatzMixin, bond::Bond)
    delete_edge!(mixin.lattice, bond)
    delete!(mixin.bondmap, bond)
end

# effect handlers
function handle!(mixin::AnsatzMixin, effect::DeleteEffect{Tensor})
    rmlane!(mixin, findfirst(==(effect.f), tn.lanemap))
end

function handle!(mixin::AnsatzMixin, effect::ReplaceEffect{Pair{Tensor,Tensor}})
    lane = findfirst(==(effect.f.first), mixin.lanemap)
    !isnothing(lane) || return nothing
    mixin.lanemap[lane] = effect.f.second
end
