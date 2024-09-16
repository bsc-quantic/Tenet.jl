using LinearAlgebra
using Graphs
using MetaGraphsNext

struct Product <: AbstractAnsatz
    tn::Ansatz
end

Ansatz(tn::Product) = tn.tn

Base.copy(x::Product) = Product(copy(Ansatz(x)))

Base.similar(x::Product) = Product(similar(Ansatz(x)))
Base.zero(x::Product) = Product(zero(Ansatz(x)))

function Product(arrays::Vector{<:AbstractVector})
    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:n]
    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(array, [symbols[i]])
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    qtn = Quantum(TensorNetwork(_tensors), sitemap)
    lattice = MetaGraph(Graph(n), Pair{Site,Nothing}[Site(i) => nothing for i in 1:n], Pair{Tuple{Site,Site},Nothing}[])
    ansatz = Ansatz(qtn, lattice)
    return Product(ansatz)
end

function Product(arrays::Vector{<:AbstractMatrix})
    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2 * length(arrays))]
    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(array, [symbols[i + n], symbols[i]], [])
    end

    sitemap = merge!(Dict(Site(i; dual=true) => symbols[i] for i in 1:n), Dict(Site(i) => symbols[i + n] for i in 1:n))
    qtn = Quantum(TensorNetwork(_tensors), sitemap)
    lattice = MetaGraph(Graph(n), Pair{Site,Nothing}[Site(i) => nothing for i in 1:n], Pair{Tuple{Site,Site},Nothing}[])
    ansatz = Ansatz(qtn, lattice)
    return Product(ansatz)
end

function Base.zeros(::Type{Product}, n::Integer; p::Int=2, eltype=Bool)
    return Product(fill(append!([one(eltype)], collect(Iterators.repeated(zero(eltype), p - 1))), n))
end

function Base.ones(::Type{Product}, n::Integer; p::Int=2, eltype=Bool)
    return Product(fill(append!([zero(eltype), one(eltype)], collect(Iterators.repeated(zero(eltype), p - 2))), n))
end

LinearAlgebra.norm(tn::Product, p::Real=2) = LinearAlgebra.norm(socket(tn), tn, p)
function LinearAlgebra.norm(::Union{State,Operator}, tn::Product, p::Real)
    return mapreduce(*, tensors(tn)) do tensor
        norm(tensor, p)
    end^(1//p)
end

LinearAlgebra.opnorm(tn::Product, p::Real=2) = LinearAlgebra.opnorm(socket(tn), tn, p)
function LinearAlgebra.opnorm(::Operator, tn::Product, p::Real)
    return mapreduce(*, tensors(tn)) do tensor
        opnorm(parent(tensor), p)
    end^(1//p)
end

LinearAlgebra.normalize!(tn::Product, p::Real=2) = LinearAlgebra.normalize!(socket(tn), tn, p)
function LinearAlgebra.normalize!(::Union{State,Operator}, tn::Product, p::Real)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end

overlap(a::Product, b::Product) = overlap(socket(a), a, socket(b), b)

function overlap(::State, a::Product, ::State, b::Product)
    @assert issetequal(sites(a), sites(b)) "Ansatzes must have the same sites"

    mapreduce(*, zip(tensors(a), tensors(b))) do (ta, tb)
        dot(parent(ta), conj(parent(tb)))
    end
end
