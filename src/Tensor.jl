using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using OMEinsum

struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    labels::NTuple{N,Symbol}
    meta::Dict{Symbol,Any}

    function Tensor(data::A, labels::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}}
        meta = Dict{Symbol,Any}(meta...)
        !haskey(meta, :tags) && (meta[:tags] = Set{String}())

        new{T,N,A}(data, labels, meta)
    end
end

Tensor(data, labels::Vector{Symbol}; meta...) = Tensor(data, tuple(labels...); meta...)

Base.convert(::Type{Tensor{T,N,A}}, t::Tensor) where {T,N,A<:AbstractArray{T,N}} =
    Tensor(convert(A, parent(t)), labels(t); t.meta...)

Base.copy(t::Tensor) = Tensor(parent(t), labels(t); deepcopy(t.meta)...)

Base.:(==)(a::A, b::T) where {A<:AbstractArray,T<:Tensor} = isequal(b, a)
Base.:(==)(a::T, b::A) where {A<:AbstractArray,T<:Tensor} = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
Base.isequal(a::Tensor, b::Tensor) = allequal(labels.((a, b))) && allequal(parent.((a, b)))

labels(t::Tensor) = t.labels

reindex(t::Tensor, mapping::Pair{Symbol,Symbol}...) = Tensor(parent(t), replace(labels(t), mapping...); copy(t.meta)...)

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
    tensor = bc.args[1]
    data = similar(parent(tensor), ElType)

    Tensor(data, labels(tensor))
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
tags(t::Tensor) = t.meta[:tags]
tag!(t::Tensor, tag::String) = push!(tags(t), tag)
hastag(t::Tensor, tag::String) = tag ∈ tags(t)
untag!(t::Tensor, tag::String) = delete!(tags(t), tag)

# Contraction
# TODO arg to keep i
function contract(a::Tensor, b::Tensor, i)
    ia = labels(a)
    ib = labels(b)

    ic = tuple(setdiff(ia ∪ ib, i isa Sequence ? i : [i])...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(a, b)

    # TODO merge metadata?
    return Tensor(data, ic)
end