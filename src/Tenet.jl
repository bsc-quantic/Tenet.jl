module Tenet

import EinExprs: inds
using Compat

include("Helpers.jl")
@compat public letter, nonunique, IndexCounter, currindex, nextindex!

include("Tensor.jl")
export Tensor, contract, dim, expand, fuse

include("Operations/Operations.jl")

include("Numerics.jl")

include("Site.jl")
export Lane, @lane_str
export Site, @site_str, isdual
@compat public id, Moment

include("Lattice.jl")
export Lattice, Bond

include("Interfaces/Effects.jl")

include("Interfaces/UnsafeScope.jl")

include("Interfaces/TensorNetwork.jl")
export AbstractTensorNetwork, tensors, ntensors, ninds, hastensor, hasind, arrays

include("Interfaces/Pluggable.jl")
export sites, nsites, hassite, addsite!, rmsite!, align!, @align!

include("Interfaces/Form.jl")
export form, canonize, canonize!, NonCanonical, MixedCanonical, Canonical

include("Interfaces/Ansatz.jl")
export lanes, nlanes, haslane, addlane!, rmlane!
export bonds, nbonds, hasbond, addbond!, rmbond!
export isisometry, absorb!, absorb

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
export Stack, layer, nlayers

include("Ansatzes/Product.jl")
export Product, ProductState, ProductOperator

include("Ansatzes/MPS.jl")
export MatrixProductState, MPS, MatrixProductOperator, MPO

include("Algorithms/Truncate.jl")
export truncate!

# include("Circuit.jl")
# export Gate, Circuit, moments

include("Algorithms/SimpleUpdate.jl")
export simple_update!

# reexports from EinExprs
export einexpr, inds

# reexports from Graphs
export neighbors

end
