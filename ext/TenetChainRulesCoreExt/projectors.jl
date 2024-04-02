# `Tensor` projector
function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds)
end

function (projector::ProjectTo{T})(dx::T) where {T<:Tensor}
    @assert projector.inds == inds(dx)
    T(projector.data(parent(dx)), inds(dx))
end

function (projector::ProjectTo{T})(dx::Tangent{T}) where {T<:Tensor}
    T(projector.data(dx.data), projector.inds)
end

function (projector::ProjectTo{Tensor{T,0}})(dx::T) where {T}
    T(projector.data(fill(dx)), projector.inds)
end

function (projector::ProjectTo{Tensor{T,N,A}})(dx::A) where {T,N,A<:AbstractArray{T,N}}
    Tensor{T,N,A}(projector.data(dx), projector.inds)
end

# `TensorNetwork` projector
ChainRulesCore.ProjectTo(tn::TensorNetwork) = ProjectTo{TensorNetwork}(; tensors = ProjectTo(tensors(tn)))

function (projector::ProjectTo{TensorNetwork})(dx)
    projmap = Dict(proj.inds => proj for proj in projector.tensors.elements)
    TensorNetworkTangent(map(tensors(dx)) do tensor
        projmap[inds(tensor)](tensor)
    end)
end
