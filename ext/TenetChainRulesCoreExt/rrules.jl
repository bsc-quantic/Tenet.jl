# `Tensor` constructor
Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractArray) = (NoTangent(), Δ, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds) = T(data, inds), Tensor_pullback

# `TensorNetwork` constructor
TensorNetwork_pullback(Δ::TensorNetworkTangent) = (NoTangent(), tensors(Δ))
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
ChainRulesCore.rrule(::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), TensorNetwork_pullback

# `Quantum` constructor
Quantum_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
Quantum_pullback(ȳ::AbstractArray) = (NoTangent(), ȳ, NoTangent())
Quantum_pullback(ȳ::AbstractThunk) = Quantum_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{Quantum}, x::TensorNetwork, sites) = Quantum(x, sites), Quantum_pullback

# `Ansatz` constructor
Ansatz_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
Ansatz_pullback(ȳ::AbstractThunk) = Ansatz_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{Ansatz}, x::Quantum, lattice) = Ansatz(x, lattice), Ansatz_pullback

# `AbstractAnsatz`-subtype constructors
Product_pullback(ȳ) = (NoTangent(), ȳ.tn)
Product_pullback(ȳ::AbstractThunk) = Product_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{Product}, x::Ansatz) = Product(x), Product_pullback

Dense_pullback(ȳ) = (NoTangent(), ȳ.tn)
Dense_pullback(ȳ::AbstractThunk) = Dense_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{Dense}, x::Ansatz) = Dense(x), Dense_pullback

MPS_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
MPS_pullback(ȳ::AbstractThunk) = MPS_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{MPS}, x::Ansatz, form) = MPS(x, form), MPS_pullback

MPO_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
MPO_pullback(ȳ::AbstractThunk) = MPO_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{MPO}, x::Ansatz, form) = MPO(x, form), MPO_pullback

PEPS_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
PEPS_pullback(ȳ::AbstractThunk) = PEPS_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{PEPS}, x::Ansatz, form) = PEPS(x, form), PEPS_pullback

# `Base.conj` methods
conj_pullback(Δ::Tensor) = (NoTangent(), conj(Δ))
conj_pullback(Δ::Tangent{Tensor}) = (NoTangent(), conj(Δ))
conj_pullback(Δ::TensorNetworkTangent) = (NoTangent(), conj(Δ))
conj_pullback(Δ::AbstractThunk) = conj_pullback(unthunk(Δ))

ChainRulesCore.rrule(::typeof(Base.conj), tn::Tensor) = conj(tn), conj_pullback
ChainRulesCore.rrule(::typeof(Base.conj), tn::TensorNetwork) = conj(tn), conj_pullback

# `Base.getindex` methods
function ChainRulesCore.rrule(::typeof(Base.getindex), x::TensorNetwork, is::Symbol...; mul::Int=1)
    y = getindex(x, is...; mul)
    nots = map(_ -> NoTangent(), is)

    getindex_pullback(z::AbstractZero) = (NoTangent(), z, nots...)
    function getindex_pullback(ȳ)
        ithunk = InplaceableThunk(
            dx -> ∇getindex!(dx, unthunk(ȳ), is...; mul), @thunk(∇getindex(x, unthunk(ȳ), is...; mul))
        )
        return (NoTangent(), ithunk, nots...)
    end

    return y, getindex_pullback
end

# TODO multiplicity
∇getindex(x::TensorNetwork, dy, is...; mul) = TensorNetworkTangent(Tensor[dy])

# TODO multiplicity
∇getindex!(x::TensorNetwork, dy, is...; mul) = push!(x, dy)

# `Base.merge` methods
function ChainRulesCore.rrule(::typeof(Base.merge), a::TensorNetwork, b::TensorNetwork)
    c = merge(a, b)

    function merge_pullback(c̄)
        ā = TensorNetworkTangent([c̄.tensors[inds(tensor)] for tensor in tensors(a)])
        b̄ = TensorNetworkTangent([c̄.tensors[inds(tensor)] for tensor in tensors(b)])
        return NoTangent(), ā, b̄
    end
    merge_pullback(c̄::AbstractVector) = merge_pullback(TensorNetworkTangent(c̄))
    merge_pullback(c̄::AbstractThunk) = merge_pullback(unthunk(c̄))

    return c, merge_pullback
end

# `contract` methods
function ChainRulesCore.rrule(::typeof(contract), x::Tensor; kwargs...)
    y = contract(x; kwargs...)
    proj = ProjectTo(x)

    function contract_pullback(ȳ)
        y_shape_with_singletons = map(inds(x)) do i
            i ∉ inds(ȳ) ? 1 : size(ȳ, i)
        end

        dims_to_repeat = map(zip(size(x), y_shape_with_singletons .== 1)) do (dₓ, issingleton)
            issingleton ? dₓ : 1
        end
        x̄ = proj(repeat(reshape(parent(ȳ), y_shape_with_singletons...), dims_to_repeat...))

        return (NoTangent(), x̄)
    end
    contract_pullback(ȳ::AbstractThunk) = contract_pullback(unthunk(ȳ))

    return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(contract), a::Tensor, b::Tensor; kwargs...)
    c = contract(a, b; kwargs...)
    proj_a = ProjectTo(a)
    proj_b = ProjectTo(b)

    function contract_pullback(c̄::Tensor)
        ā = @thunk proj_a(contract(c̄, conj(b); out=inds(a)))
        b̄ = @thunk proj_b(contract(conj(a), c̄; out=inds(b)))
        return (NoTangent(), ā, b̄)
    end
    contract_pullback(c̄::AbstractVector) = contract_pullback(Tensor(c̄, inds(c)))
    contract_pullback(c̄::AbstractThunk) = contract_pullback(unthunk(c̄))

    return c, contract_pullback
end

copy_pullback(ȳ) = (NoTangent(), ȳ)
copy_pullback(ȳ::AbstractThunk) = unthunk(ȳ)
function ChainRulesCore.rrule(::typeof(copy), x::Quantum)
    y = copy(x)
    return y, copy_pullback
end
