# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

@non_differentiable Base.merge!(::TensorNetwork, ::TensorNetwork...)
@non_differentiable Base.merge(::TensorNetwork, ::TensorNetwork...)

@non_differentiable Base.replace!(::TensorNetwork, ::Any...)
@non_differentiable Base.replace(::TensorNetwork, ::Any...)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
