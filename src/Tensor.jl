using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using EinExprs

struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::NTuple{N,Symbol}
    meta::Dict{Symbol,Any}

    function Tensor{T,N,A}(data::A, inds::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}}
        meta = Dict{Symbol,Any}(meta...)
        haskey(meta, :tags) || (meta[:tags] = Set{String}())
        all(i -> allequal(Iterators.map(dim -> size(data, dim), findall(==(i), inds))), nonunique(collect(inds))) ||
            throw(DimensionMismatch("nonuniform size of repeated indices"))

        new{T,N,A}(data, inds, meta)
    end
end

Tensor(data, inds::Vector{Symbol}; meta...) = Tensor(data, tuple(inds...); meta...)
Tensor(data::A, inds::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, inds; meta...)
Tensor{T,N,A}(data::A, inds::NTuple{N,Symbol}, meta) where {T,N,A<:AbstractArray{T,N}} =
    Tensor{T,N,A}(data, inds; meta...)

Tensor(data::AbstractArray{T,0}; meta...) where {T} = Tensor(data, (); meta...)
Tensor(data::Number; meta...) = Tensor(fill(data); meta...)

Base.copy(t::Tensor) = Tensor(parent(t), inds(t); deepcopy(t.meta)...)

function Base.copy(t::Tensor{T,N,<:SubArray{T,N}}) where {T,N}
    data = copy(t.data)
    inds = t.inds
    meta = deepcopy(t.meta)
    return Tensor(data, inds; (k => v for (k, v) in meta)...)
end

# TODO pass new inds and meta
function Base.similar(t::Tensor{_,N}, ::Type{T}; kwargs...) where {_,T,N}
    if N == 0
        return Tensor(similar(parent(t), T), (); kwargs...)
    else
        similar(t, T, size(t)...; kwargs...)
    end
end
# TODO fix this
function Base.similar(t::Tensor, ::Type{T}, dims::Int64...; inds = inds(t), meta...) where {T}
    data = similar(parent(t), T, dims)

    # copy metadata
    metadata = copy(t.meta)
    merge!(metadata, meta)

    Tensor(data, inds; meta...)
end

function __find_index_permutation(a, b)
    inds_b = collect(Union{Missing,Symbol}, b)

    Iterators.map(a) do label
        i = findfirst(isequal(label), inds_b)

        # mark element as used
        inds_b[i] = missing

        i
    end |> collect
end

Base.:(==)(a::AbstractArray, b::Tensor) = isequal(b, a)
Base.:(==)(a::Tensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
function Base.isequal(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || return false
    perm = __find_index_permutation(inds(a), inds(b))
    return all(eachindex(IndexCartesian(), a)) do i
        j = CartesianIndex(Tuple(permute!(collect(Tuple(i)), invperm(perm))))
        isequal(a[i], b[j])
    end
end

Base.isapprox(a::AbstractArray, b::Tensor) = false
Base.isapprox(a::Tensor, b::AbstractArray) = false
function Base.isapprox(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || return false
    perm = __find_index_permutation(inds(a), inds(b))
    return all(eachindex(IndexCartesian(), a)) do i
        j = CartesianIndex(Tuple(permute!(collect(Tuple(i)), invperm(perm))))
        isapprox(a[i], b[j])
    end
end

EinExprs.inds(t::Tensor) = t.inds

# NOTE: `replace` does not currenly support cyclic replacements
function Base.replace(t::Tensor, old_new::Pair{Symbol,Symbol}...)
    new_inds = replace(inds(t), old_new...)
    new_meta = deepcopy(t.meta)
    old_new_dict = Base.ImmutableDict(old_new...)

    haskey(new_meta, :alias) && map!(values(new_meta[:alias])) do i
        get(old_new_dict, i, i)
    end

    return Tensor(parent(t), new_inds; new_meta...)
end

Base.parent(t::Tensor) = t.data
parenttype(::Type{Tensor{T,N,A}}) where {T,N,A} = A

dim(t::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = findall(==(i), inds(t)) |> first

# Iteration interface
Base.IteratorSize(T::Type{Tensor}) = Iterators.IteratorSize(parenttype(T))
Base.IteratorEltype(T::Type{Tensor}) = Iterators.IteratorEltype(parenttype(T))

Base.isdone(t::Tensor) = (Base.isdone ∘ parent)(t)
Base.isdone(t::Tensor, state) = (Base.isdone ∘ parent)(t)

# Indexing interface
Base.IndexStyle(T::Type{<:Tensor}) = IndexStyle(parenttype(T))

@propagate_inbounds Base.getindex(t::Tensor, i...) = getindex(parent(t), i...)
@propagate_inbounds function Base.getindex(t::Tensor; i...)
    length(i) == 0 && return (getindex ∘ parent)(t)
    return getindex(t, [get(i, label, Colon()) for label in inds(t)]...)
end

@propagate_inbounds Base.setindex!(t::Tensor, v, i...) = setindex!(parent(t), v, i...)
@propagate_inbounds function Base.setindex!(t::Tensor, v; i...)
    length(i) == 0 && return setindex!(parent(t), v)
    return setindex!(t, v, [get(i, label, Colon()) for label in inds(t)]...)
end

Base.firstindex(t::Tensor) = firstindex(parent(t))
Base.lastindex(t::Tensor) = lastindex(parent(t))

# AbstractArray interface
"""
    Base.size(::Tensor[, i])

Return the size of the underlying array or the dimension `i` (specified by `Symbol` or `Integer`).
"""
Base.size(t::Tensor) = size(parent(t))
Base.size(t::Tensor, i) = size(parent(t), dim(t, i))

Base.length(t::Tensor) = length(parent(t))

Base.axes(t::Tensor) = axes(parent(t))
Base.axes(t::Tensor, d) = axes(parent(t), dim(t, d))

# StridedArrays interface
Base.strides(t::Tensor) = strides(parent(t))
Base.stride(t::Tensor, i::Symbol) = stride(parent(t), dim(t, i))

Base.unsafe_convert(::Type{Ptr{T}}, t::Tensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(t))

Base.elsize(T::Type{<:Tensor}) = elsize(parenttype(T))

# Broadcasting
Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = ArrayStyle{T}()

function Base.similar(bc::Broadcasted{ArrayStyle{Tensor{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType}
    # NOTE already checked if dimension mismatch
    # TODO throw on label mismatch?
    tensor = first(arg for arg in bc.args if arg isa Tensor{T,N,A})
    similar(tensor, ElType)
end

Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), inds(t); t.meta...)
function Base.selectdim(t::Tensor, d::Integer, i::Integer)
    data = selectdim(parent(t), d, i)
    indices = [label for (i, label) in enumerate(inds(t)) if i != d]
    Tensor(data, indices; t.meta...)
end

Base.selectdim(t::Tensor, d::Symbol, i) = selectdim(t, dim(t, d), i)

Base.permutedims(t::Tensor, perm) = Tensor(permutedims(parent(t), perm), getindex.((inds(t),), perm); t.meta...)
Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

function Base.permutedims(t::Tensor{T,N}, perm::NTuple{N,Symbol}) where {T,N}
    perm = map(i -> findfirst(==(i), inds(t)), perm)
    permutedims(t, perm)
end

Base.dropdims(t::Tensor; dims = tuple(findall(==(1), size(t))...)) =
    Tensor(dropdims(parent(t); dims), inds(t)[setdiff(1:ndims(t), dims)]; t.meta...)

Base.view(t::Tensor, i...) =
    Tensor(view(parent(t), i...), [label for (label, j) in zip(inds(t), i) if !(j isa Integer)]; t.meta...)

function Base.view(t::Tensor, inds::Pair{Symbol,<:Any}...)
    indices = map(Tenet.inds(t)) do ind
        i = findfirst(x -> x == ind, first.(inds))
        !isnothing(i) ? inds[i].second : Colon()
    end

    let data = view(parent(t), indices...),
        inds = [label for (index, label) in zip(indices, Tenet.inds(t)) if !(index isa Integer)]

        Tensor(data, inds; t.meta...)
    end
end

Base.adjoint(t::Tensor) = Tensor(conj(parent(t)), inds(t); t.meta...)

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::Tensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = permutedims(t, (1,))
Base.transpose(t::Tensor{T,2,A}) where {T,A<:AbstractArray{T,2}} =
    Tensor(transpose(parent(t)), reverse(inds(t)); t.meta...)

function expand(tensor::Tensor; label, axis = 1, size = 1, method = :zeros)
    array = parent(tensor)
    data =
        size == 1 ? reshape(array, Base.size(array)[1:axis-1]..., 1, Base.size(array)[axis:end]...) :
        method === :zeros ? __expand_zeros(array, axis, size) :
        method === :repeat ? __expand_repeat(array, axis, size) :
        # method === :identity ? __expand_identity(array, axis, size) :
        throw(ArgumentError("method \"$method\" is not valid"))

    inds = (Tenet.inds(tensor)[1:axis-1]..., label, Tenet.inds(tensor)[axis:end]...)

    return Tensor(data, inds; tensor.meta...)
end

function __expand_zeros(array, axis, size)
    new = zeros(eltype(array), Base.size(array)[1:axis-1]..., size, Base.size(array)[axis:end]...)

    view = selectdim(new, axis, 1)
    copy!(view, array)

    return new
end

__expand_repeat(array, axis, size) = repeat(
    reshape(array, Base.size(array)[1:axis-1]..., 1, Base.size(array)[axis:end]...),
    outer = (fill(1, axis - 1)..., size, fill(1, ndims(array) - axis + 1)...),
)

"""
    tags(tensor)

Return a set of `String`s associated to `tensor`.
"""
tags(t::Tensor) = t.meta[:tags]

"""
    tag!(tensor, tag)

Mark `tensor` with `tag`.
"""
tag!(t::Tensor, tag::String) = push!(tags(t), tag)

"""
    hastag(tensor, tag)

Return `true` if `tensor` contains tag `tag`.
"""
hastag(t::Tensor, tag::String) = tag ∈ tags(t)

"""
    untag!(tensor, tag)

Removes tag `tag` from `tensor` if present.
"""
untag!(t::Tensor, tag::String) = delete!(tags(t), tag)
