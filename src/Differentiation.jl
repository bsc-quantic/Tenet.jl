using ChainRulesCore

## Tensor
function ChainRulesCore.frule((_, ȧ, ḃ)::NTuple{3,Any}, ::typeof(contract), a, b)
    c = contract(a, b)
    ċ = @thunk(contract(ȧ, b)) + @thunk(contract(a, ḃ))
    return c, ċ
end

function ChainRulesCore.frule((_, ȧ, ḃ, _)::NTuple{4,Any}, ::typeof(contract), a, b, i)
    c = contract(a, b, i)
    ċ = @thunk(contract(ȧ, b, i)) + @thunk(contract(a, ḃ, i))
    return c, ċ
end

function ChainRulesCore.ProjectTo(t::T) where {T<:Tensor}
    ProjectTo{T}(; data = ProjectTo(parent(t)), labels = labels(t), meta = t.meta)
end

(project::ProjectTo{Tensor{T,N,A}})(dx::A) where {T,N,A<:AbstractArray{T,N}} =
    Tensor{T,N,A}(project.data(dx), project.labels; project.meta...)
(project::ProjectTo{<:Number})(dx::Tensor{_,0}) where {_} = project(only(dx))
(project::ProjectTo{<:Tensor})(dx::T) where {T<:Tensor} = Tensor(project.data(dx.data), project.labels; project.meta...)
(project::ProjectTo{<:Tensor{T,0}})(dx::T) where {T} = Tensor(project.data(fill(dx)), project.labels; project.meta...)

# TODO is this the correct way?
(project::ProjectTo{<:Tensor})(dx::Thunk) = project(unthunk(dx))

function ChainRulesCore.rrule(f::Type{<:Tensor}, data, labels; meta...)
    t = f(data, labels; meta...)
    Tensor_pullback(t̄) = (NoTangent(), unthunk(t̄).data, NoTangent())

    return t, Tensor_pullback
end

function ChainRulesCore.rrule(::typeof(only), t::Tensor{T,0}) where {T}
    data = only(t)

    # TODO use `ProjectTo(t)`
    only_pullback(d̄) = (NoTangent(), Tensor(fill(d̄), labels(t); t.meta...))
    return data, only_pullback
end

ChainRulesCore.rrule(::typeof(contract), a::AbstractArray{T,0}, b) where {T} = rrule(contract, only(a), b)
ChainRulesCore.rrule(::typeof(contract), a, b::AbstractArray{T,0}) where {T} = rrule(contract, a, only(b))
ChainRulesCore.rrule(::typeof(contract), a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) =
    rrule(contract, only(a), only(b))

function ChainRulesCore.rrule(::typeof(contract), a::A, b::B) where {A,B}
    c = contract(a, b)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)
    project_c = ProjectTo(c)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B} # TODO @thunk type inference
        ȧ = contract(project_c(c̄), b')
        ḃ = contract(a', project_c(c̄))
        return NoTangent(), project_a(ȧ), project_b(ḃ)
    end
    return c, contract_pullback
end

function ChainRulesCore.rrule(::typeof(contract), a::A, b::B, i) where {A<:Tensor,B<:Tensor}
    c = contract(a, b, i)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B,NoTangent} # TODO @thunk type inference
        ȧ = contract(c̄, b', i)
        ḃ = contract(a', c̄, i)
        return NoTangent(), project_a(ȧ), project_b(ḃ), NoTangent()
    end
    return c, contract_pullback
end

## TensorNetwork
function ChainRulesCore.ProjectTo(tn::T) where {T<:TensorNetwork}
    ProjectTo{T}(; tensors = map(ProjectTo, tensors(tn)), meta = tn.meta) #= inds = inds(tn), =#
end

# TODO pass index metadata
function (project::ProjectTo{<:TensorNetwork})(dx::T) where {T<:TensorNetwork}
    T(map((proj, tensor) -> proj(tensor), project.tensors, tensors(dx)); project.meta...)
end

# function ChainRulesCore.frule((_, ṫn), ::typeof(TensorNetwork{A}), tensors; meta...) where {A}
# end

function ChainRulesCore.frule((_, ṫn)::NTuple{2,Any}, ::typeof(contract), tn::TensorNetwork; kwargs...)
    c = contract(tn; kwargs...)
    ċ = contract(ṫn; kwargs...)

    return c, ċ
end

# function ChainRulesCore.rrule(::typeof(T); meta...) where {T<:TensorNetwork}
#     TensorNetwork_pullback = identity
#     @info "[rrule :: $T] meta=$meta"
#     return T(; meta...), TensorNetwork_pullback
# end

function ChainRulesCore.rrule(::typeof(Base.copy), tn::TensorNetwork)
    copy_pullback(Δ) = (NoTangent(), Δ)
    return copy(tn), copy_pullback
end

function ChainRulesCore.rrule(::typeof(Base.replace), tn::TensorNetwork, old_new...)
    res = replace(tn, old_new...)

    # TODO `replace` or `replace!`?
    replace_pullback(Δ, m...) =
        (NoTangent(), replace(Δ, [new => old for (old, new) in old_new]...), fill(NoTangent(), length(m))...)

    return res, replace_pullback
end

# function ChainRulesCore.rrule(::typeof(Base.replace!), tn::TensorNetwork, old_new...)
#     @info "[rrule :: replace!] tn=$tn"
#     # TODO `replace` or `replace!`?
#     res = replace(tn, old_new...)

#     # TODO `replace` or `replace!`?
#     replace!_pullback(Δ, m...) =
#         (NoTangent(), replace(Δ, [new => old for (old, new) in old_new]...), fill(NoTangent(), length(m))...)

#     return res, replace!_pullback
# end

function ChainRulesCore.rrule(::typeof(Base.adjoint), tn::TensorNetwork)
    adjoint_pullback(Δ) = (NoTangent(), adjoint(Δ))

    return adjoint(tn), adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(Base.hcat), a::TensorNetwork, b::TensorNetwork)
    c = hcat(a, b)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)

    function hcat_pullback(c̄)
        # TODO
        ā = project_a(a ∩ c̄)
        b̄ = project_b(b ∩ c̄)

        return NoTangent(), ā, b̄
    end

    return c, hcat_pullback
end

function ChainRulesCore.rrule(::typeof(contract), tn::TensorNetwork{A}; kwargs...) where {A}
    c = contract(tn; kwargs...)
    project_tn = ProjectTo(tn)

    # NOTE one tangent per tensor
    function contract_pullback(c̄)
        f̄ = NoTangent()

        # TODO keep ansatz?
        # TODO project_tn
        t̄n = TensorNetwork{A}(map(tensors(tn)) do tensor
            ∂tn = delete!(copy(tn), tensor)
            contract(contract(∂tn), c̄)
        end)

        return (f̄, t̄n)
    end

    return c, contract_pullback
end

## used in tests
function ChainRulesCore.frule((_, Δargs...), ::typeof(only ∘ contract), args::Vararg{<:Any,N}) where {N}
    c, ċ = frule((nothing, Δargs...), contract, args...)
    return only(c), reduce((acc, _) -> only(acc), 1:N; init = ċ)
end

function ChainRulesCore.rrule(::typeof(only ∘ contract), args...)
    c, pb = rrule(contract, args...)
    return only(c), pb
end