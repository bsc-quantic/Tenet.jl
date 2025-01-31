using Test
using SafeTestsets
using Tenet
using OMEinsum

include("Utils.jl")

const TENET_TEST_GROUP = lowercase(get(ENV, "TENET_TEST_GROUP", "all"))

if TENET_TEST_GROUP == "all" || TENET_TEST_GROUP == "unit"
    @testset "Unit tests" verbose = true begin
        include("unit/Helpers_test.jl")
        include("unit/Tensor_test.jl")
        include("unit/Numerics_test.jl")
        include("unit/TensorNetwork_test.jl")
        include("unit/Transformations_test.jl")
        include("unit/Lane_test.jl")
        include("unit/Site_test.jl")
        include("unit/Moment_test.jl")
        include("unit/Quantum_test.jl")
        include("unit/Gate_test.jl")
        include("unit/Circuit_test.jl")
        include("unit/Lattice_test.jl")
        include("unit/Ansatz_test.jl")
        include("unit/Product_test.jl")
        include("unit/MPS_test.jl")
        include("unit/MPO_test.jl")
    end
end

if TENET_TEST_GROUP == "all" || TENET_TEST_GROUP == "integration"
    @testset "Integration tests" verbose = true begin
        @safetestset "Python" begin
            run(`cp CondaPkg.toml ../CondaPkg.toml`)
            using Test
            using Tenet
            using CondaPkg
            CondaPkg.update()
            using PythonCall

            include("integration/python/test_cirq.jl")
            include("integration/python/test_quimb.jl")
            include("integration/python/test_qiskit.jl")
            include("integration/python/test_qibo.jl")
            run(`rm ../CondaPkg.toml`)
        end

        include("integration/Reactant_test.jl")
        include("integration/ChainRules_test.jl")
        # include("integration/BlockArray_test.jl")
        include("integration/Dagger_test.jl")
        include("integration/Makie_test.jl")
        include("integration/KrylovKit_test.jl")
        include("integration/Quac_test.jl")
        include("integration/ITensors_test.jl")
        include("integration/ITensorNetworks_test.jl")
        include("integration/YaoBlocks_test.jl")
    end
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        @testset "Method ambiguity (manual)" Aqua.test_ambiguities(Tenet, recursive=false, exclude=[==])
        Aqua.test_all(Tenet; ambiguities=false, stale_deps=false)
    end
end
