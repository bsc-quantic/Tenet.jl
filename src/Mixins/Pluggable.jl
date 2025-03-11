"""
    PluggableMixin

A mixin for objects that can be plugged.
"""
struct PluggableMixin
    sitemap::Dict{Site,Symbol}
end

trait(::PluggableInterface, ::PluggableMixin) = IsPluggable()

# required methods
sites(::@NamedTuple{}, mixin::PluggableMixin) = sort!(collect(keys(mixin.sitemap)))
inds(::@NamedTuple{at::S}, mixin::PluggableMixin) where {S<:Site} = mixin.sitemap[at]
sites(::@NamedTuple{at::Symbol}, mixin::PluggableMixin) = findfirsrt(==(at), mixin.sitemap)

# optional methods
nsites(::@NamedTuple{}, mixin::PluggableMixin) = length(keys(mixin.sitemap))
hassite(mixin::PluggableMixin, site::Site) = haskey(mixin.sitemap, site)

# mutating methods
addsite!(mixin::PluggableMixin, p::Pair{<:Site,Symbol}) = mixin.sitemap[p.first] = p.second
rmsite!(mixin::PluggableMixin, site::Site) = delete!(mixin.sitemap, site)

# effect handlers
function handle!(mixin::PluggableMixin, effect::ReplaceEffect{Pair{Symbol,Symbol}})
    site = findfirst(==(effect.f.first), mixin.sitemap)
    isnothing(site) && return nothing
    mixin.sitemap[site] = effect.f.second
end
