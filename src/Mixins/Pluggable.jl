"""
    PluggableMixin

A mixin for objects that can be plugged.
"""
struct PluggableMixin
    sitemap::Dict{Site,Symbol}
end

Base.copy(mixin::PluggableMixin) = PluggableMixin(copy(mixin.sitemap))

trait(::PluggableInterface, ::PluggableMixin) = IsPluggable()

# required methods
sites(::@NamedTuple{}, mixin::PluggableMixin) = sort!(collect(keys(mixin.sitemap)))
inds(kwargs::@NamedTuple{at::S}, mixin::PluggableMixin) where {S<:Site} = mixin.sitemap[kwargs.at]
sites(kwargs::@NamedTuple{at::Symbol}, mixin::PluggableMixin) = findfirst(==(kwargs.at), mixin.sitemap)

# optional methods
nsites(::@NamedTuple{}, mixin::PluggableMixin) = length(keys(mixin.sitemap))
hassite(mixin::PluggableMixin, site::Site) = haskey(mixin.sitemap, site)

# mutating methods
function addsite!(mixin::PluggableMixin, site::Site, ind::Symbol)
    hassite(mixin, site) && error("Site $site already exists")
    mixin.sitemap[site] = ind
end

function rmsite!(mixin::PluggableMixin, site::Site)
    if !hassite(mixin, site)
        @debug "Site $site does not exist"
        return nothing
    end
    delete!(mixin.sitemap, site)
end

# effect handlers
function handle!(mixin::PluggableMixin, effect::ReplaceEffect{Pair{Symbol,Symbol}})
    old, new = effect.f
    site = sites(mixin; at=old)
    isnothing(site) && return nothing
    mixin.sitemap[site] = new
end
