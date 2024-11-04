using Test
using Tenet
using OMEinsum

include("Utils.jl")

@testset "Unit tests" verbose = true begin
    include("Helpers_test.jl")
    include("Tensor_test.jl")
    include("Numerics_test.jl")
    include("TensorNetwork_test.jl")
    include("Transformations_test.jl")
    include("Site_test.jl")
    include("Quantum_test.jl")
    include("Lattice_test.jl")
    include("Ansatz_test.jl")
    include("Product_test.jl")
    include("Chain_test.jl")
end

@testset "Integration tests" verbose = true begin
    include("integration/ChainRules_test.jl")
    # include("integration/BlockArray_test.jl")
    include("integration/Dagger_test.jl")
    include("integration/Makie_test.jl")
    include("integration/KrylovKit_test.jl")
    include("integration/Quac_test.jl")
    include("integration/ITensors_test.jl")
    include("integration/ITensorNetworks_test.jl")

    @testset "Python" begin
        include("integration/python/test_quimb.jl")
        include("integration/python/test_qiskit.jl")
        include("integration/python/test_qibo.jl")
    end
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        @testset "Method ambiguity (manual)" Aqua.test_ambiguities(Tenet, recursive=false, exclude=[==])
        Aqua.test_all(Tenet; ambiguities=false, stale_deps=false)
    end
end
