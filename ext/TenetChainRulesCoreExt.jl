module TenetChainRulesCoreExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using ChainRulesCore

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

end
