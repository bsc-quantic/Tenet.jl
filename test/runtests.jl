using Test
using SafeTestsets
using Tenet
using OMEinsum

include("Utils.jl")
include("Interfaces.jl")

const TENET_TEST_GROUP = lowercase(get(ENV, "TENET_TEST_GROUP", "all"))

if TENET_TEST_GROUP == "all" || TENET_TEST_GROUP == "unit"
    @testset "Unit tests" verbose = true begin
        @testset "Helpers" include("unit/Helpers_test.jl")
        @testset "Tensor" include("unit/Tensor_test.jl")
        @testset "Numerics" include("unit/Numerics_test.jl")
        @testset "TensorNetwork" include("unit/TensorNetwork_test.jl")
        @testset "Transformations" include("unit/Transformations_test.jl")
        @testset "Lane" include("unit/Lane_test.jl")
        @testset "Site" include("unit/Site_test.jl")
        @testset "Moment" include("unit/Moment_test.jl")
        # @testset "Gate" include("unit/Gate_test.jl")
        # @testset "Quantum" include("unit/Quantum_test.jl")
        # @testset "Circuit" include("unit/Circuit_test.jl")
        # @testset "Lattice" include("unit/Lattice_test.jl")
        # @testset "Ansatz" include("unit/Ansatz_test.jl")
        # @testset "Product" include("unit/Product_test.jl")
        # @testset "MPS" include("unit/MPS_test.jl")
        # @testset "MPO" include("unit/MPO_test.jl")
    end
end

# if TENET_TEST_GROUP == "all" || TENET_TEST_GROUP == "integration"
#     @testset "Integration tests" verbose = true begin
#         @safetestset "Python" begin
#             run(`cp CondaPkg.toml ../CondaPkg.toml`)
#             using Test
#             using Tenet
#             using CondaPkg
#             CondaPkg.update()
#             using PythonCall

#             include("integration/python/test_cirq.jl")
#             include("integration/python/test_quimb.jl")
#             include("integration/python/test_qiskit.jl")
#             include("integration/python/test_qibo.jl")
#             run(`rm ../CondaPkg.toml`)
#         end

#         @safetestset "Reactant" begin
#             include("integration/Reactant_test.jl")
#         end

#         @safetestset "ChainRules" begin
#             include("integration/ChainRules_test.jl")
#         end

#         @safetestset "Dagger" begin
#             include("integration/Dagger_test.jl")
#         end

#         @safetestset "Makie" begin
#             include("integration/Makie_test.jl")
#         end

#         @safetestset "KrylovKit" begin
#             include("integration/KrylovKit_test.jl")
#         end

#         @safetestset "Quac" begin
#             include("integration/Quac_test.jl")
#         end

#         @safetestset "ITensors" begin
#             include("integration/ITensors_test.jl")
#         end

#         @safetestset "ITensorMPS" begin
#             include("integration/ITensorMPS_test.jl")
#         end

#         @safetestset "ITensorNetworks" begin
#             include("integration/ITensorNetworks_test.jl")
#         end

#         @safetestset "YaoBlocks" begin
#             include("integration/YaoBlocks_test.jl")
#         end
#     end
# end

# if haskey(ENV, "ENABLE_AQUA_TESTS")
#     @testset "Aqua" verbose = true begin
#         using Aqua
#         @testset "Method ambiguity (manual)" Aqua.test_ambiguities(Tenet, recursive=false, exclude=[==])
#         Aqua.test_all(Tenet; ambiguities=false, stale_deps=false)
#     end
# end
