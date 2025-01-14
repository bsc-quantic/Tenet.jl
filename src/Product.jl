using LinearAlgebra
using Graphs: Graphs
using BijectiveDicts

"""
    Product <: AbstractAnsatz

An [`Ansatz`](@ref) represented as a tensor product.

# Constructors

If you pass an `Abstract{<:AbstractVector}` to the constructor, it will create a [`State`](@ref).
If you pass an `Abstract{<:AbstractMatrix}` to the constructor, it will create an [`Operator`](@ref).
"""
struct Product <: AbstractAnsatz
    tn::Ansatz
end

Ansatz(tn::Product) = tn.tn

Base.copy(x::Product) = Product(copy(Ansatz(x)))
Base.similar(x::Product) = Product(similar(Ansatz(x)))
Base.zero(x::Product) = Product(zero(Ansatz(x)))

function Product(arrays::AbstractArray{<:AbstractVector})
    n = length(arrays)
    gen = IndexCounter()
    symbols = map(arrays) do _
        nextindex!(gen)
    end
    _tensors = map(eachindex(arrays)) do i
        Tensor(arrays[i], [symbols[i]])
    end

    sitemap = Dict(Site(i) => symbols[i] for i in eachindex(arrays))
    qtn = Quantum(TensorNetwork(_tensors), sitemap)
    graph = Graphs.Graph(n)
    mapping = BijectiveDict{Lane,Int}(Pair{Lane,Int}[lane => i for (i, lane) in enumerate(lanes(qtn))])
    lattice = Lattice(mapping, graph)
    ansatz = Ansatz(qtn, lattice)
    return Product(ansatz)
end

function Product(arrays::AbstractArray{<:AbstractMatrix})
    n = length(arrays)
    gen = IndexCounter()
    symbols = map(arrays) do _
        (nextindex!(gen), nextindex!(gen))
    end
    _tensors = map(eachindex(arrays)) do i
        Tensor(arrays[i], [symbols[i][1], symbols[i][2]])
    end

    sitemap = merge!(
        Dict(Site(i; dual=true) => symbols[i][1] for i in eachindex(arrays)),
        Dict(Site(i) => symbols[i][2] for i in eachindex(arrays)),
    )
    qtn = Quantum(TensorNetwork(_tensors), sitemap)
    graph = Graphs.Graph(n)
    mapping = BijectiveDict{Lane,Int}(Pair{Lane,Int}[lane => i for (i, lane) in enumerate(lanes(qtn))])
    lattice = Lattice(mapping, graph)
    ansatz = Ansatz(qtn, lattice)
    return Product(ansatz)
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
