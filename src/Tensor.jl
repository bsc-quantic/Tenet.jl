using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle

struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::Vector{Symbol}

    function Tensor{T,N,A}(data::A, inds::AbstractVector) where {T,N,A<:AbstractArray{T,N}}
        length(inds) == N ||
            throw(ArgumentError("ndims(data) [$(ndims(data))] must be equal to length(inds) [$(length(inds))]"))
        all(i -> allequal(Iterators.map(dim -> size(data, dim), findall(==(i), inds))), nonunique(collect(inds))) ||
            throw(DimensionMismatch("nonuniform size of repeated indices"))

        return new{T,N,A}(data, inds)
    end
end

Tensor(data::A, inds::AbstractVector{Symbol}) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, inds)
Tensor(data::A, inds::NTuple{N,Symbol}) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, collect(inds))

Tensor(data::AbstractArray{T,0}) where {T} = Tensor(data, Symbol[])
Tensor(data::Number) = Tensor(fill(data))

inds(t::Tensor) = t.inds

function Base.copy(t::Tensor{T,N,<:SubArray{T,N}}) where {T,N}
    data = copy(t.data)
    inds = t.inds
    return Tensor(data, inds)
end

Base.similar(t::Tensor; inds=inds(t)) = Tensor(similar(parent(t)), inds)
Base.similar(t::Tensor, S::Type; inds=inds(t)) = Tensor(similar(parent(t), S), inds)
function Base.similar(t::Tensor{T,N}, S::Type, dims::Base.Dims{N}; inds=inds(t)) where {T,N}
    return Tensor(similar(parent(t), S, dims), inds)
end
Base.similar(t::Tensor{T,N}, dims::Base.Dims{N}; inds=inds(t)) where {T,N} = Tensor(similar(parent(t), dims), inds)

Base.zero(t::Tensor) = Tensor(zero(parent(t)), inds(t))

function __find_index_permutation(a, b)
    inds_b = collect(Union{Missing,Symbol}, b)

    return collect(
        Iterators.map(a) do label
            i = findfirst(isequal(label), inds_b)

            # mark element as used
            inds_b[i] = missing

            i
        end,
    )
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

Base.isequal(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = isequal(only(a), only(b))

Base.isapprox(a::AbstractArray, b::Tensor) = false
Base.isapprox(a::Tensor, b::AbstractArray) = false
function Base.isapprox(a::Tensor, b::Tensor; kwargs...)
    issetequal(inds(a), inds(b)) || return false
    perm = __find_index_permutation(inds(a), inds(b))
    return all(eachindex(IndexCartesian(), a)) do i
        j = CartesianIndex(Tuple(permute!(collect(Tuple(i)), invperm(perm))))
        isapprox(a[i], b[j]; kwargs...)
    end
end

Base.isapprox(a::Tensor{T,0}, b::T; kwargs...) where {T} = isapprox(only(a), b; kwargs...)
Base.isapprox(a::T, b::Tensor{T,0}; kwargs...) where {T} = isapprox(b, a; kwargs...)
Base.isapprox(a::Tensor{A,0}, b::Tensor{B,0}; kwargs...) where {A,B} = isapprox(only(a), only(b); kwargs...)

# NOTE: `replace` does not currenly support cyclic replacements
Base.replace(t::Tensor, old_new::Pair{Symbol,Symbol}...) = Tensor(parent(t), replace(inds(t), old_new...))

Base.parent(t::Tensor) = t.data
parenttype(::Type{Tensor{T,N,A}}) where {T,N,A} = A
parenttype(::T) where {T<:Tensor} = parenttype(T)

dim(::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = first(findall(==(i), inds(t)))

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
    return similar(tensor, ElType)
end

Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), inds(t))
function Base.selectdim(t::Tensor, d::Integer, i::Integer)
    data = selectdim(parent(t), d, i)
    indices = [label for (i, label) in enumerate(inds(t)) if i != d]
    return Tensor(data, indices)
end

Base.selectdim(t::Tensor, d::Symbol, i) = selectdim(t, dim(t, d), i)

Base.permutedims(t::Tensor, perm) = Tensor(permutedims(parent(t), perm), getindex.((inds(t),), perm))
Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

function Base.permutedims(t::Tensor{T}, perm::Base.AbstractVecOrTuple{Symbol}) where {T}
    perm = map(i -> findfirst(==(i), inds(t)), perm)
    return permutedims(t, perm)
end

function Base.dropdims(t::Tensor; dims=tuple(findall(==(1), size(t))...))
    return Tensor(dropdims(parent(t); dims), inds(t)[setdiff(1:ndims(t), dims)])
end

function Base.view(t::Tensor, i...)
    return Tensor(view(parent(t), i...), [label for (label, j) in zip(inds(t), i) if !(j isa Integer)])
end

function Base.view(t::Tensor, inds::Pair{Symbol,<:Any}...)
    indices = map(Tenet.inds(t)) do ind
        i = findfirst(x -> x == ind, first.(inds))
        !isnothing(i) ? inds[i].second : Colon()
    end

    let data = view(parent(t), indices...),
        inds = [label for (index, label) in zip(indices, Tenet.inds(t)) if !(index isa Integer)]

        Tensor(data, inds)
    end
end

Base.adjoint(t::Tensor) = Tensor(conj(parent(t)), inds(t))

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::Tensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = permutedims(t, (1,))
Base.transpose(t::Tensor{T,2,A}) where {T,A<:AbstractArray{T,2}} = Tensor(transpose(parent(t)), reverse(inds(t)))

function expand(tensor::Tensor; label, axis=1, size=1, method=:zeros)
    array = parent(tensor)
    data = if size == 1
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...)
    elseif method === :zeros
        __expand_zeros(array, axis, size)
    elseif method === :repeat
        __expand_repeat(array, axis, size)
    else
        # method === :identity ? __expand_identity(array, axis, size) :
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    inds = (Tenet.inds(tensor)[1:(axis - 1)]..., label, Tenet.inds(tensor)[axis:end]...)

    return Tensor(data, inds)
end

function __expand_zeros(array, axis, size)
    new = zeros(eltype(array), Base.size(array)[1:(axis - 1)]..., size, Base.size(array)[axis:end]...)

    view = selectdim(new, axis, 1)
    copy!(view, array)

    return new
end

function __expand_repeat(array, axis, size)
    return repeat(
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...);
        outer=(fill(1, axis - 1)..., size, fill(1, ndims(array) - axis + 1)...),
    )
end
