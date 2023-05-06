using UUIDs: uuid4
using Base.Iterators: flatten
using IterTools: partition
using Random

abstract type MatrixProduct{P,B} <: Quantum where {P<:Plug,B<:Boundary} end

boundary(::Type{<:MatrixProduct{P,B}}) where {P,B} = B
plug(::Type{<:MatrixProduct{P}}) where {P} = P

function MatrixProduct{P}(arrays; boundary::Type{<:Boundary} = Open, kwargs...) where {P<:Plug}
    MatrixProduct{P,boundary}(arrays; kwargs...)
end

function checkmeta(::Type{MatrixProduct{P,B}}, tn::TensorNetwork) where {P,B}
    # meta exists
    haskey(tn.metadata, :χ) || return false

    # meta has correct type
    tn[:χ] isa Integer && tn[:χ] > 0 || isnothing(tn[:χ]) || return false

    # no virtual index has dimensionality bigger than χ
    all(i -> size(tn, i) <= tn[:χ], labels(tn, :inner)) || return false

    return true
end

_sitealias(::Type{MatrixProduct{State,Open}}, order, n, i) = order[indexin(if i == 1
    (:r, :o)
elseif i == n
    (:l, :o)
else
    (:l, :r, :o)
end, collect(order))]

_sitealias(::Type{MatrixProduct{State,Periodic}}, order, n, i) = tuple(order...)

function MatrixProduct{State,B}(arrays; χ = nothing, order = (:l, :r, :o), metadata...) where {B<:Boundary}
    issetequal(order, (:l, :r, :o)) || throw(ArgumentError("`order` must be a permutation of the :l, :r and :o"))

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    pinds = Dict(i => Symbol(uuid4()) for i in 1:n)

    # mark plug connectors
    plug = Dict((site, :out) => label for (site, label) in pinds)

    tensors = map(enumerate(arrays)) do (i, array)
        dirs = _sitealias(MatrixProduct{State,B}, order, n, i)

        labels = map(dirs) do dir
            if dir === :l
                vinds[(mod1(i - 1, n), i)]
            elseif dir === :r
                vinds[(i, mod1(i + 1, n))]
            elseif dir === :o
                pinds[i]
            end
        end
        alias = Dict(dir => label for (dir, label) in zip(dirs, labels))

        Tensor(array, labels; alias = alias)
    end

    return TensorNetwork{MatrixProduct{State,B}}(tensors; χ, plug, metadata...)
end

# NOTE does not use optimal contraction path, but "parallel-optimal" which costs x2 more
# function contractpath(a::TensorNetwork{<:MatrixProductState}, b::TensorNetwork{<:MatrixProductState})
#     !issetequal(sites(a), sites(b)) && throw(ArgumentError("both tensor networks are expected to have same sites"))

#     b = replace(b, [nameof(outsiteind(b, s)) => nameof(outsiteind(a, s)) for s in sites(a)]...)
#     path = nameof.(flatten([physicalinds(a), flatten(zip(virtualinds(a), virtualinds(b)))]) |> collect)
#     inputs = flatten([tensors(a), tensors(b)]) .|> labels
#     output = Symbol[]
#     size_dict = merge(size(a), size(b))

#     ContractionPath(path, inputs, output, size_dict)
# end

# struct MPSSampler{B<:Bounds,T} <: Random.Sampler{TensorNetwork{MatrixProductState{B}}}
#     n::Int
#     p::Int
#     χ::Int
# end

# Base.eltype(::MPSSampler{B}) where {B<:Bounds} = TensorNetwork{MatrixProductState{B}}

# function Base.rand(
#     ::Type{MatrixProductState{B}},
#     n::Integer,
#     p::Integer,
#     χ::Integer;
#     eltype::Type = Float64,
# ) where {B<:Bounds}
#     rand(MPSSampler{B,eltype}(n, p, χ))
# end

# # NOTE `Vararg{Integer,3}` for dissambiguation
# Base.rand(::Type{MatrixProductState}, args::Vararg{Integer,3}; kwargs...) =
#     rand(MatrixProductState{Open}, args...; kwargs...)

# # TODO let choose the orthogonality center
# function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Open,T}) where {T}
#     n, χ, p = getfield.((sampler,), (:n, :χ, :p))

#     arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
#         χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
#             χl = min(χ, p^(i - 1))
#             χr = min(χ, p^i)

#             # swap bond dims after mid
#             after_mid ? (χr, χl) : (χl, χr)
#         end

#         # fix for first site
#         i == 1 && ((χl, χr) = (χr, 1))

#         # orthogonalize by Gram-Schmidt algorithm
#         A = gramschmidt!(rand(rng, T, χl, χr * p))

#         reshape(A, χl, χr, p)
#     end

#     # reshape boundary sites
#     arrays[1] = reshape(arrays[1], p, p)
#     arrays[n] = reshape(arrays[n], p, p)

#     # normalize state
#     arrays[1] ./= sqrt(p)

#     MatrixProductState{Open}(arrays; χ = χ)
# end

# # TODO stable renormalization
# function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Closed,T}) where {T}
#     n, χ, p = getfield.((sampler,), (:n, :χ, :p))
#     ψ = MatrixProductState{Closed}([rand(rng, T, χ, χ, p) for _ in 1:n]; χ = χ)
#     normalize!(ψ)

#     return ψ
# end
