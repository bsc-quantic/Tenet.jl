using Test
using Tenet
using OMEinsum

@testset "Unit tests" verbose = true begin
    include("Helpers_test.jl")
    include("Tensor_test.jl")
    include("Numerics_test.jl")
    include("TensorNetwork_test.jl")
    include("Transformations_test.jl")
    include("Site_test.jl")
    include("Quantum_test.jl")
    include("Product_test.jl")
    include("Chain_test.jl")
end

# CI hangs on these tests for some unknown reason on Julia 1.9
if VERSION >= v"1.10"
    @testset "Integration tests" verbose = true begin
        include("integration/ChainRules_test.jl")
        # include("integration/BlockArray_test.jl")
        include("integration/Dagger_test.jl")
        include("integration/Makie_test.jl")
        include("integration/KrylovKit_test.jl")
        include("integration/Quac_test.jl")
    end
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        @testset "Method ambiguity (manual)" Aqua.test_ambiguities(Tenet, recursive=false, exclude=[==])
        Aqua.test_all(Tenet; ambiguities=false, stale_deps=false)
    end
end
