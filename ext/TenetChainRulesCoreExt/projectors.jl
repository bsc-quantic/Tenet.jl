# `Tensor` projector
function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds)
end

function (projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:Tensor}
    T(projector.data(dx.data), projector.inds)
end

# `TensorNetwork` projector
ChainRulesCore.ProjectTo(tn::TensorNetwork) = ProjectTo{TensorNetwork}(; tensors = ProjectTo(tensors(tn)))

function (projector::ProjectTo{TensorNetwork})(dx::TensorNetwork)
    Tangent{TensorNetwork}(tensormap = projector.tensors(tensors(dx)), indexmap = NoTangent())
end

function (projector::ProjectTo{TensorNetwork})(dx::Tangent{TensorNetwork})
    dx.tensormap isa NoTangent && return NoTangent()
    Tangent{TensorNetwork}(tensormap = projector.tensors(dx.tensors), indexmap = NoTangent())
end
