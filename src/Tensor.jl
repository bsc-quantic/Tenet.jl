struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    labels::NTuple{N,Symbol}
end

labels(t::Tensor) = t.labels

Base.parent(t::Tensor) = t.data
parenttype(::Type{Tensor{T,N,A}}) where {T,N,A} = A

dim(t::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = findall(==(i), parent(t)) |> first

# Indexing interface
Base.IndexStyle(T::Type{<:Tensor}) = IndexStyle(parenttype(T))

Base.getindex(t::Tensor, i) = getindex(parent(t), i)

Base.setindex!(t::Tensor, v, i) = setindex!(parent(t), v, i)

Base.firstindex(t::Tensor) = firstindex(parent(t))
Base.lastindex(t::Tensor) = lastindex(parent(t))

# AbstractArray interface
Base.size(t::Tensor) = size(parent(t))

Base.length(t::Tensor) = length(parent(t))

Base.axes(t::Tensor) = axes(parent(t))
Base.axes(t::Tensor, d) = axes(parent(t), dim(t, d))

# StridedArrays interface
Base.strides(t::Tensor) = strides(parent(t))

Base.stride(t::Tensor, i) = stride(parent(t), i)
Base.stride(t::Tensor, i::Symbol) = stride(parent(t), dim(t, i))

Base.elsize(T::Type{<:Tensor}) = elsize(parenttype(T))