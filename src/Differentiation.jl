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

ChainRulesCore.ProjectTo(t::T) where {T<:Tensor} =
    ProjectTo{T}(; data = ProjectTo(parent(t)), labels = labels(t), meta = t.meta)

(project::ProjectTo{Tensor{T,N,A}})(dx::A) where {T,N,A<:AbstractArray{T,N}} =
    Tensor{T,N,A}(project.data(dx), project.labels; project.meta...)
(project::ProjectTo{<:Number})(dx::Tensor{_,0}) where {_} = project(only(dx))
(project::ProjectTo{<:Tensor})(dx::Tensor) = Tensor(project.data(dx.data), project.labels; project.meta...)

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
    only_pullback(d̄) = Tensor(fill(d̄), labels(t); t.meta...)
    return data, only_pullback
end

function ChainRulesCore.rrule(::typeof(contract), a::A, b::B) where {A<:Tensor,B<:Tensor}
    c = contract(a, b)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)
    project_c = ProjectTo(c)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B} # TODO @thunk type inference
        f̄ = NoTangent()
        ā = project_a(contract(project_c(c̄), b)) # TODO @thunk
        b̄ = project_b(contract(a, project_c(c̄))) # TODO @thunk

        return f̄, ā, b̄
    end

    return c, contract_pullback
end

function ChainRulesCore.rrule(::typeof(contract), a::A, b::B, i) where {A<:Tensor,B<:Tensor}
    c = contract(a, b, i)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)
    project_c = ProjectTo(c)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B,NoTangent} # TODO @thunk type inference
        f̄ = NoTangent()
        ā = project_a(contract(project_c(c̄), b, i)) # TODO @thunk
        b̄ = project_b(contract(a, project_c(c̄), i)) # TODO @thunk
        ī = NoTangent()

        return f̄, ā, b̄, ī
    end

    return c, contract_pullback
end
