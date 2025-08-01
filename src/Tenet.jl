module Tenet

using Reexport

using DelegatorTraits
import DelegatorTraits: ImplementorTrait, DelegatorTrait, Implements, NotImplements
using DelegatorTraits: Interface, fallback

@reexport using QuantumTags
@reexport using Muscle
using Networks
@reexport using Tangles

abstract type AbstractTangle <: Tangles.AbstractTensorNetwork end
struct Tangle <: Interface end

# utils
## `Base.Checked.checked_pow` introduced in Julia 1.11
if hasproperty(Base.Checked, :checked_pow)
    const checked_pow = Base.Checked.checked_pow
else
    const checked_pow = ^
end

# traits
include("CanonicalForm.jl")
export CanonicalForm, form, NonCanonical, MixedCanonical, BondCanonical, VidalGauge

# components
include("Components/ProductState.jl")
export ProductState, ProductOperator

include("Components/MPO.jl")
export MatrixProductOperator, MPO

include("Components/MPS.jl")
export MatrixProductState, MPS

# legacy aliases
# TODO remove in future version
const MixedCanonicalMatrixProductState = MatrixProductState
const MixedCanonicalMPS = MixedCanonicalMatrixProductState
export MixedCanonicalMatrixProductState, MixedCanonicalMPS

include("Components/VidalMPS.jl")
export VidalMatrixProductState, VidalMPS

include("Components/PEPS.jl")
export ProjectedEntangledPairState, PEPS

# operations
include("Operations/canonize.jl")
export canonize, canonize!

include("Operations/absorb.jl")
export absorb, absorb!

include("Operations/evolve.jl")
export evolve, evolve!

include("Operations/simple_update.jl")
export simple_update, simple_update!

include("Operations/overlap.jl")
export overlap

include("Operations/norm.jl")
export norm

include("Operations/normalize.jl")
export normalize!

include("Operations/compress.jl")
export compress!, compress

include("Operations/entropy.jl")

include("Operations/sample.jl")

include("Algorithms/DMRG.jl")
import .DMRG: dmrg!

include("Models/Models.jl")

end
