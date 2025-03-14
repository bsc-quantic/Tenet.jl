module Tenet

import EinExprs: inds
using Compat

include("Helpers.jl")
@compat public letter, nonunique, IndexCounter, currindex, nextindex!

include("Tensor.jl")
export Tensor, contract, dim, expand, fuse

include("Numerics.jl")

include("Site.jl")
export Lane, @lane_str
export Site, @site_str, isdual
@compat public id, Moment

include("Lattice.jl")
export Lattice

include("Interfaces/Effects.jl")

include("Interfaces/TensorNetwork.jl")
export AbstractTensorNetwork, tensors, ntensors, ninds, hastensor, hasind, arrays

include("Interfaces/Pluggable.jl")
export sites, nsites, hassite, addsite!, rmsite!, align!

include("Interfaces/Form.jl")
export form, canonize, canonize!, NonCanonical, MixedCanonical, Canonical

include("Interfaces/Ansatz.jl")
export lanes, nlanes, haslane, addlane!, rmlane!
export bonds, nbonds, hasbond, addbond!, rmbond!

include("TensorNetwork.jl")
export TensorNetwork, slice!, contract!, fuse!
@compat public @unsafe_region, tryprune!, resetinds!

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

include("Mixins/Pluggable.jl")
@compat public PluggableMixin

include("Mixins/Ansatz.jl")
@compat public AnsatzMixin

include("Gate.jl")
export Gate

include("Stack.jl")
export Stack, component

include("Ansatzes/Product.jl")
export Product, ProductState, ProductOperator

include("Ansatzes/MPS.jl")
export MatrixProductState, MPS, MatrixProductOperator, MPO

# include("Pluggable.jl")
# export Quantum, ninputs, noutputs, inputs, outputs, sites, lanes, socket
# @compat public AbstractQuantum, Socket, Scalar, State, Operator, align!, nsites, nlanes, hassite

# include("Circuit.jl")
# export Gate, Circuit, moments

# include("Ansatz.jl")
# #! format: off
# export Ansatz,
#     boundary,
#     Open,
#     Periodic,
#     form,
#     NonCanonical,
#     MixedCanonical,
#     Canonical,
#     canonize,
#     canonize!,
#     mixed_canonize,
#     mixed_canonize!,
#     truncate!,
#     isisometry,
#     expect,
#     evolve!,
#     simple_update!,
#     overlap
# #! format: on
# @compat public AbstractAnsatz, Boundary, Form

# include("MPS.jl")
# export MPS, MPO, absorb, absorb!
# @compat public AbstractMPS, AbstractMPO, defaultorder, check_form

# include("Algorithms/SimpleUpdate.jl")
# export simple_update!

# reexports from EinExprs
export einexpr, inds

# reexports from Graphs
export neighbors

end
