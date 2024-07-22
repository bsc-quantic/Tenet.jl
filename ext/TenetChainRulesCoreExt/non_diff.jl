# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

# TODO maybe we need to convert this into a frule/rrule? such that the tangents change their indices too
@non_differentiable Base.replace!(::TensorNetwork, ::Pair{Symbol,Symbol}...)

@non_differentiable Tenet.currindex(::Tenet.IndexCounter)
@non_differentiable Tenet.nextindex!(::Tenet.IndexCounter)
@non_differentiable Tenet.resetindex!(::Tenet.IndexCounter)

# WARN type-piracy
@non_differentiable Base.setdiff(::Vector{Symbol}, ::Base.ValueIterator)

@non_differentiable Tenet.inputs(::Quantum)
@non_differentiable Tenet.ninputs(::Quantum)
@non_differentiable Tenet.outputs(::Quantum)
@non_differentiable Tenet.noutputs(::Quantum)
