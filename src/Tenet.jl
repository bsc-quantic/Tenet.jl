module Tenet

import EinExprs: inds
using Compat

include("Helpers.jl")
@compat public letter, nonunique, IndexCounter, currindex, nextindex!

include("Tensor.jl")
export Tensor, contract, dim, expand

include("Numerics.jl")

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, neighbors, slice!, contract, contract!, groupinds!
@compat public AbstractTensorNetwork, ninds, ntensors, @unsafe_region, tryprune!, resetindex!

include("Transformations.jl")
export transform, transform!
#! format: off
@compat public Transformation,
    HyperFlatten,
    HyperGroup,
    ContractSimplification,
    Truncate,
    DiagonalReduction,
    AntiDiagonalGauging,
    SplitSimplification
#! format: on

include("Site.jl")
export Lane, @lane_str
export Site, @site_str, isdual
@compat public id Moment

include("Quantum.jl")
export Quantum, ninputs, noutputs, inputs, outputs, sites, lanes, socket
@compat public AbstractQuantum, Socket, Scalar, State, Operator, reindex!, @reindex!, nsites, nlanes, hassite

include("Circuit.jl")
export Gate, Circuit

include("Lattice.jl")
@compat public Lattice

include("Ansatz.jl")
#! format: off
export Ansatz,
    boundary,
    Open,
    Periodic,
    form,
    NonCanonical,
    MixedCanonical,
    Canonical,
    canonize,
    canonize!,
    mixed_canonize,
    mixed_canonize!,
    truncate!,
    isisometry,
    expect,
    evolve!,
    simple_update!,
    overlap
#! format: on
@compat public AbstractAnsatz, Boundary, Form

include("Product.jl")
export Product

include("MPS.jl")
export MPS, MPO
@compat public AbstractMPS, AbstractMPO, defaultorder, check_form

include("PEPS.jl")

# reexports from EinExprs
export einexpr, inds

end
