using Test
using Tenet
using OMEinsum

@testset "Unit tests" verbose = true begin
    include("Helpers_test.jl")
    include("Tensor_test.jl")
    include("Numerics_test.jl")
    include("TensorNetwork_test.jl")
    include("Transformations_test.jl")
end

@testset "Integration tests" verbose = true begin
    include("integration/ChainRules_test.jl")
    include("integration/BlockArray_test.jl")
    include("integration/Makie_test.jl")
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        @testset "Method ambiguity (manual)" Aqua.test_ambiguities(Tenet, recursive = false, exclude = [==])
        Aqua.test_all(Tenet, ambiguities = false, stale_deps = false)
    end
end
