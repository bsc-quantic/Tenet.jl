using UUIDs: uuid4
using Base.Iterators: flatten
using IterTools: partition
using Random

abstract type MatrixProductState{B} <: State{B} end

function MatrixProductState(arrays; bounds::Type{<:Bounds} = Open, kwargs...)
    MatrixProductState{bounds}(arrays; kwargs...)
end

function MatrixProductState{Open}(arrays; χ = nothing, order = (:l, :r, :p), meta...)
    !issetequal(order, (:l, :r, :p)) && throw(ArgumentError("`order` must be a permutation of the :l, :r and :p"))
    order = Dict(dim => i for (i, dim) in enumerate(order))

    # check format
    # NOTE matching index dimensions are checked on `push!(::TensorNetwork,...)`
    # TODO check χ?
    if !all(==(3) ∘ ndims, arrays[2:end-1]) || ndims(first(arrays)) != 2 || ndims(last(arrays)) != 2
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductState{Open}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in partition(1:n, 2, 1))
    pinds = Dict(i => Symbol(uuid4()) for i in 1:n)

    # add tensors
    for (i, data) in enumerate(arrays)
        # Handle boundary cases and inner tensors separately
        if i == 1
            labels = [vinds[(1, 2)], pinds[1]]
            original_order = [:r, :p]
        elseif i == n
            labels = [vinds[(n - 1, n)], pinds[n]]
            original_order = [:l, :p]
        else
            lind = vinds[(mod1(i - 1, n), i)]
            rind = vinds[(i, mod1(i + 1, n))]
            labels = [lind, rind, pinds[i]]
            original_order = [:l, :r, :p]
        end

        # Filter the order dictionary based on the original_order and sort it following order
        filtered_order = [p[1] for p in sort(filter(p -> p.first ∈ original_order, collect(order)), by = x -> x[2])]
        permutator = (ind -> Dict(p => i for (i, p) in enumerate(filtered_order))[ind]).(original_order)

        invpermute!(labels, permutator)
        alias = Dict([x => y for (x, y) in zip(invpermute!(original_order, permutator), labels)])

        push!(tn, Tensor(data, labels; alias = alias))
    end

    # mark physical indices
    for (site, label) in pinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
end

function MatrixProductState{Closed}(arrays; χ = nothing, order = (:l, :r, :p), meta...)
    !issetequal(order, (:l, :r, :p)) && throw(ArgumentError("`order` must be a permutation of the :l, :r and :p"))
    order = Dict(side => i for (i, side) in enumerate(order))

    # check format
    # TODO check χ?
    if !all(==(3) ∘ ndims, arrays)
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductState{Closed}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    pinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    permutator = [order[i] for i in (:l, :r, :p)]

    # add tensors
    for (i, data) in enumerate(arrays)
        lind = vinds[(mod1(i - 1, n), i)]
        rind = vinds[(i, mod1(i + 1, n))]

        labels = [lind, rind, pinds[i]]
        invpermute!(labels, permutator)
        alias = Dict([x => y for (x, y) in zip(invpermute!([:l, :r, :p], permutator), labels)])

        tensor = Tensor(data, labels; alias = alias)
        push!(tn, tensor)
    end

    # mark physical indices
    for (site, label) in pinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
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

# Base.push!(::TensorNetwork{MatrixProductState}, args...; kwargs...) =
#     throw(MethodError("push! is forbidden for MatrixProductState"))

# Base.pop!(::TensorNetwork{MatrixProductState}, args...; kwargs...) =
#     throw(MethodError("pop! is forbidden for MatrixProductState"))

struct MPSSampler{B<:Bounds,T} <: Random.Sampler{TensorNetwork{MatrixProductState{B}}}
    n::Int
    p::Int
    χ::Int
end

Base.eltype(::MPSSampler{B}) where {B<:Bounds} = TensorNetwork{MatrixProductState{B}}

function Base.rand(
    ::Type{MatrixProductState{B}},
    n::Integer,
    p::Integer,
    χ::Integer;
    eltype::Type = Float64,
) where {B<:Bounds}
    rand(MPSSampler{B,eltype}(n, p, χ))
end

# NOTE `Vararg{Integer,3}` for dissambiguation
Base.rand(::Type{MatrixProductState}, args::Vararg{Integer,3}; kwargs...) =
    rand(MatrixProductState{Open}, args...; kwargs...)

# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Open,T}) where {T}
    n, χ, p = getfield.((sampler,), (:n, :χ, :p))

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # fix for first site
        i == 1 && ((χl, χr) = (χr, 1))

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, χl, χr * p))

        reshape(A, χl, χr, p)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    # normalize state
    arrays[1] ./= sqrt(p)

    MatrixProductState{Open}(arrays; χ = χ)
end

# TODO stable renormalization
function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Closed,T}) where {T}
    n, χ, p = getfield.((sampler,), (:n, :χ, :p))
    ψ = MatrixProductState{Closed}([rand(rng, T, χ, χ, p) for _ in 1:n]; χ = χ)
    normalize!(ψ)

    return ψ
end

function canonize(ψ::TensorNetwork{MatrixProductState{Open}}, center::Union{Integer,UnitRange}; chi::Int = 0, return_singular_values::Bool=false)
    canonize!(deepcopy(ψ), center; chi = chi, return_singular_values = return_singular_values)
end

# TODO modify ψ in place
function canonize!(ψ::TensorNetwork{MatrixProductState{Open}}, center::Union{Integer,UnitRange}; chi::Int = 0, return_singular_values::Bool=false)

    # Define the local helper function canonize_two_sites within canonize
    function canonize_two_sites(A::Tensor, B::Tensor, i::Int, center::Union{Integer,UnitRange}; chi::Int = 0, return_singular_values::Bool=false)
        left_edge, right_edge = extrema(center)

        A_alias = A.meta[:alias]
        B_alias = B.meta[:alias]

        C = contract(A, B)

        left_inds = (kv -> kv[2]).(filter(p -> p.first ≠ :r, collect(A_alias)))
        U, S, V = svd(C; left_inds=left_inds)

        # Truncate if desired bond dimension is provided
        if chi > 0 && chi < size(S, 1)
            U = view(U, labels(U)[end] => 1:chi)
            S = Tensor(Diagonal(view(S, labels(S)[1] => 1:chi,  labels(S)[2] => 1:chi)), labels(S))
            V = view(V, labels(V)[begin] => 1:chi)
        end

        # Reshape U and V and update tensors
        if i < left_edge # Move orthogonality center to the right
            A_new = replace(U, labels(U)[end]=> A_alias[:r])
            B_new = replace(V * S, labels(U)[end]=> B_alias[:l])
        elseif i > right_edge # Move orthogonality center to the left
            A_new = replace(U * S, labels(V)[begin]=> A_alias[:r])
            B_new = replace(V, labels(V)[begin]=> B_alias[:l])
        else # No need to update tensors
            A_new = A
            B_new = B
        end

        A_new = Tensor(parent(A_new), labels(A_new); alias = A_alias)
        B_new = Tensor(parent(B_new), labels(B_new); alias = B_alias)

        return return_singular_values ? (A_new, B_new, S) : (A_new, B_new)
    end

    N = length(ψ)
    return_singular_values ? singular_values = Vector{Vector{Float64}}(undef, N-1) : nothing

    # First sweep from left to right
    for i in 1:N-1
        A = tensors(ψ, i)
        B = tensors(ψ, i+1)

        A_new, B_new, S = return_singular_values ?
            canonize_two_sites(A, B, i, center; chi = chi, return_singular_values = true) :
            (canonize_two_sites(A, B, i, center; chi = chi)..., nothing) # Ignore S

        replace!(ψ, A => A_new)
        replace!(ψ, B => B_new)
    end

    # Second sweep from right to left to ensure canonical form
    for i in N-1:-1:1
        A = tensors(ψ, i)
        B = tensors(ψ, i+1)

        A_new, B_new, S = return_singular_values ?
            canonize_two_sites(A, B, i, center; chi = chi, return_singular_values = true) :
            (canonize_two_sites(A, B, i, center; chi = chi)..., nothing) # Ignore S

        # Normalize singular values with ./ sqrt(sum(parent(S).diag .^ 2))
        return_singular_values && (singular_values[i] = parent(S).diag ./ sqrt(sum(parent(S).diag .^ 2)))

        replace!(ψ, A => A_new)
        replace!(ψ, B => B_new)
    end

    return_singular_values ? (ψ, singular_values) : ψ
end

# TODO function iscanonical