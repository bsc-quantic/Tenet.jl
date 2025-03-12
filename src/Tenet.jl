module Tenet

import EinExprs: inds
import Graphs: neighbors
using Compat

include("Helpers.jl")
@compat public letter, nonunique, IndexCounter, currindex, nextindex!

include("Tensor.jl")
export Tensor, contract, dim, expand, fuse

include("Numerics.jl")

include("Site.jl")
export Lane, @lane_str
export Site, @site_str, isdual
export Bond
@compat public id, Moment

include("Lattice.jl")
export Lattice

include("Interfaces/Effects.jl")

include("Interfaces/TensorNetwork.jl")
export AbstractTensorNetwork, tensors, ntensors, ninds, hastensor, hasind, arrays, contract!

include("TensorNetwork.jl")
export TensorNetwork, slice!, fuse!
@compat public @unsafe_region, tryprune!, resetinds!

include("Interfaces/Pluggable.jl")
export sites, nsites, hassite, align!, @align!, socket
@compat public addsite!, rmsite!, Socket, Scalar, State, Operator

include("Mixins/Pluggable.jl")
@compat public PluggableMixin

include("Quantum.jl")
export Quantum
@compat public AbstractQuantum

include("Gate.jl")
export Gate

include("Interfaces/Ansatz.jl")
export lanes, bonds, haslane, hasbond, nlanes, nbonds
@compat public addbond!, rmbond!

include("Mixins/Ansatz.jl")
@compat public AnsatzMixin

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

include("Circuit.jl")
export Circuit, moments

include("Product.jl")
export Product

include("MPS.jl")
export MPS, MPO, absorb, absorb!
@compat public AbstractMPS, AbstractMPO, defaultorder, check_form

# reexports from EinExprs
export einexpr, inds

# reexports from Graphs
export neighbors

end
