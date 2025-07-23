using Test
using SafeTestsets
using Tenet

TEST_GROUP = get(ENV, "TENET_TEST_GROUP", "all")

if TEST_GROUP == "all" || TEST_GROUP == "unit"
    @testset "Unit" verbose = true begin
        @safetestset "ProductState" include("unit/product_state.jl")
        @safetestset "ProductOperator" include("unit/product_operator.jl")
        @safetestset "MatrixProductState" include("unit/mps.jl")
        @safetestset "MatrixProductOperator" include("unit/mpo.jl")
    end
end

if TEST_GROUP == "all" || TEST_GROUP == "integration"
    @testset "Integration" verbose = true begin
        @safetestset "ITensorMPS" include("integration/itensormps.jl")
    end
end
