using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using OMEinsum

struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    labels::NTuple{N,Symbol}
    meta::Dict{Symbol,Any}

    function Tensor{T,N,A}(data::A, labels::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}}
        meta = Dict{Symbol,Any}(meta...)
        !haskey(meta, :tags) && (meta[:tags] = Set{String}())

        new{T,N,A}(data, labels, meta)
    end
end

Tensor(data, labels::Vector{Symbol}; meta...) = Tensor(data, tuple(labels...); meta...)
Tensor(data::A, labels::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}} =
    Tensor{T,N,A}(data, labels; meta...)
Tensor{T,N,A}(data::A, labels::NTuple{N,Symbol}, meta) where {T,N,A<:AbstractArray{T,N}} =
    Tensor{T,N,A}(data, labels; meta...)

Base.copy(t::Tensor) = Tensor(parent(t), labels(t); deepcopy(t.meta)...)

# TODO pass new labels and meta
function Base.similar(t::Tensor{_,N}, ::Type{T}; kwargs...) where {_,T,N}
    if N == 0
        return Tensor(similar(parent(t), T), (); kwargs...)
    else
        similar(t, T, size(t)...; kwargs...)
    end
end
# TODO fix this
function Base.similar(t::Tensor, ::Type{T}, dims::Int64...; labels = labels(t), meta...) where {T}
    data = similar(parent(t), T, dims)

    # copy metadata
    metadata = copy(t.meta)
    merge!(metadata, meta)

    Tensor(data, labels; meta...)
end

Base.:(==)(a::AbstractArray, b::Tensor) = isequal(b, a)
Base.:(==)(a::Tensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
Base.isequal(a::Tensor, b::Tensor) = allequal(labels.((a, b))) && allequal(parent.((a, b)))

labels(t::Tensor) = t.labels

Base.replace(t::Tensor, old_new::Pair{Symbol,Symbol}...) =
    Tensor(parent(t), replace(labels(t), old_new...); copy(t.meta)...)

Base.parent(t::Tensor) = t.data
parenttype(::Type{Tensor{T,N,A}}) where {T,N,A} = A

dim(t::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = findall(==(i), labels(t)) |> first

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
    return getindex(t, [get(i, label, Colon()) for label in labels(t)]...)
end

@propagate_inbounds Base.setindex!(t::Tensor, v, i...) = setindex!(parent(t), v, i...)
@propagate_inbounds function Base.setindex!(t::Tensor, v; i...)
    length(i) == 0 && return setindex!(parent(t), v)
    return setindex!(t, v, [get(i, label, Colon()) for label in labels(t)]...)
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

Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), labels(t); t.meta...)
function Base.selectdim(t::Tensor, d::Integer, i::Integer)
    data = selectdim(parent(t), d, i)
    indices = [label for (i, label) in enumerate(labels(t)) if i != d]
    Tensor(data, indices; t.meta...)
end

Base.selectdim(t::Tensor, d::Symbol, i) = selectdim(t, dim(t, d), i)

Base.permutedims(t::Tensor, perm) = Tensor(permutedims(parent(t), perm), getindex.((labels(t),), perm); t.meta...)
Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

Base.view(t::Tensor, i...) =
    Tensor(view(parent(t), i...), [label for (label, j) in zip(labels(t), i) if !(j isa Integer)]; t.meta...)

function Base.view(t::Tensor, inds::Pair{Symbol,<:Any}...)
    indices = map(labels(t)) do ind
        i = findfirst(x -> x == ind, first.(inds))
        !isnothing(i) ? inds[i].second : Colon()
    end

    let data = view(parent(t), indices...),
        labels = [label for (index, label) in zip(indices, labels(t)) if !(index isa Integer)]

        Tensor(data, labels; t.meta...)
    end
end

# Metadata
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

# Contraction
function contract(a::Tensor, b::Tensor, i = ∩(labels(a), labels(b)))
    ia = labels(a)
    ib = labels(b)
    i = ∩(i, ia, ib)

    ic = tuple(setdiff(ia ∪ ib, i isa Sequence ? i : [i])...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(a, b)

    # TODO merge metadata?
    return Tensor(data, ic)
end

contract(a::Number, b) = a * b
contract(a::Number, b, _) = contract(a, b)
contract(a, b::Number) = a * b
contract(a, b::Number, _) = contract(a, b)
contract(a::Number, b::Number) = a * b
contract(a::Number, b::Number, _) = contract(a, b)

contract(a::AbstractArray{T,0}, b) where {T} = contract(only(a), b)
contract(a, b::AbstractArray{T,0}) where {T} = contract(a, only(b))
