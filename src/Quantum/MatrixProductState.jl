using OptimizedEinsum: ContractionPath
using UUIDs: uuid4
using Base.Iterators: flatten
using IterTools: partition
using Random
using OMEinsum

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

            # swap bond dims after mid
            after_mid ? (χr, χl) : (χl, χr)
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

# TODO modify ψ in place
function canonize(ψ::TensorNetwork{MatrixProductState{Open}}, center::Union{Integer,UnitRange}; chi::Int = 0, return_singular_values::Bool=false)

    # Define the local helper function canonize_two_sites within canonize
    function canonize_two_sites(A::Tensor, B::Tensor, i::Int, center::Union{Integer,UnitRange}; chi::Int = 0, return_singular_values::Bool=false)
        left_edge, right_edge = extrema(center)

        C = @ein C[a, b, c, d] := A[a, e, c] * B[e, b, d] # C ->  left, right, physical, physical
        C_perm = permutedims(C, (1, 3, 2, 4))
        C_matrix = reshape(C_perm, size(C, 1) * size(C, 3), size(C, 2) * size(C, 4)) # C_matrix -> left * physical, right * physical
        U, S, V = svd(C_matrix)

        # Truncate if desired bond dimension is provided
        if chi > 0 && chi < length(S)
            U = U[:, 1:chi] # U -> left * physical, chi
            S = S[1:chi] # S -> chi
            V = V[:, 1:chi] # V -> right * physical, chi
        end

        # Reshape U and V and update tensors
        if i < left_edge # Move orthogonality center to the right
            A_new = reshape(U, size(A, 1), size(A, 3), size(U, 2))
            A_new = permutedims(A_new, (1, 3, 2))
            B_new = reshape(permutedims(V * diagm(S),(2,1)), size(U, 2), size(B, 2), size(B, 3))

        elseif i > right_edge # Move orthogonality center to the left
            A_new = reshape(U * diagm(S), size(A, 1), size(A, 3), size(U, 2))
            A_new = permutedims(A_new, (1, 3, 2))
            B_new = reshape(permutedims(V, (2, 1)), size(U, 2), size(B, 2), size(B, 3))

        else # No need to update tensors
            A_new = A
            B_new = B
        end

        return return_singular_values ? (A_new, B_new, S) : (A_new, B_new)
    end

    N = length(ψ)
    tensors_array = [tensors(ψ, i) for i in 1:N] # Copy tensors to arrays
    return_singular_values ? singular_values = Vector{Vector{Float64}}(undef, N-1) : nothing

    # Transform the first and last tensors to be three-dimensional
    tensors_array[1] = Tensor(
        reshape(tensors_array[1], (1, size(tensors_array[1])...)),
        (:empty, labels(tensors_array[1])[1], labels(tensors_array[1])[2]);
        meta = tensors_array[1].meta,
    )

    tensors_array[end] = Tensor(
        reshape(tensors_array[end], (size(tensors_array[end], 1), 1, size(tensors_array[end], 2))),
        (labels(tensors_array[end])[1], :empty, labels(tensors_array[end])[2]);
        meta = tensors_array[end].meta,
    )

    # First sweep from left to right
    for i in 1:N-1
        A = tensors_array[i]
        B = tensors_array[i+1]

        A_new, B_new, S = return_singular_values ?
            canonize_two_sites(A, B, i, center; chi = chi, return_singular_values = true) :
            (canonize_two_sites(A, B, i, center; chi = chi)..., nothing) # Ignore S

        # Update tensors in the array
            tensors_array[i] = Tensor(A_new, labels(A); meta = A.meta)
            tensors_array[i+1] = Tensor(B_new, labels(B); meta = B.meta)
        end

    # Second sweep from right to left to ensure canonical form
    for i in N-1:-1:1
        A = tensors_array[i]
        B = tensors_array[i+1]

        A_new, B_new, S = return_singular_values ?
            canonize_two_sites(A, B, i, center; chi = chi, return_singular_values = true) :
            (canonize_two_sites(A, B, i, center; chi = chi)..., nothing) # Ignore S

        # Normalize singular values with ./ sqrt(sum(S .^ 2))
        return_singular_values && (singular_values[i] = S ./ sum(S .^ 2))

        # Update tensors in the array, reshape boundary tensors
        if i == 1
            tensors_array[i] =
                Tensor(reshape(A_new, size(A, 2), size(A, 3)), (labels(A)[2], labels(A)[3]); meta = A.meta)
            tensors_array[i+1] = Tensor(B_new, labels(B); meta = B.meta)
        elseif i == N - 1
            tensors_array[i] = Tensor(A_new, labels(A); meta = A.meta)
            tensors_array[i+1] = Tensor(reshape(B_new, size(B, 1), size(B, 3)), (labels(B)[1], labels(B)[3]); meta = B.meta)
        else
            tensors_array[i] = Tensor(A_new, labels(A); meta = A.meta)
            tensors_array[i+1] = Tensor(B_new, labels(B); meta = B.meta)
        end
    end


    # Create a new MPS object with the updated tensors
    ψ = MatrixProductState(tensors_array)

    return_singular_values ? (ψ, singular_values) : ψ
end

# TODO function iscanonical