using Test
using SafeTestsets
using Tenet

@testset "Unit" verbose = true begin
    @safetestset "ProductState" include("unit/product_state.jl")
    @safetestset "ProductOperator" include("unit/product_operator.jl")
    @safetestset "MatrixProductState" include("unit/mps.jl")
    @safetestset "MatrixProductOperator" include("unit/mpo.jl")
end

@testset "Integration" verbose = true begin
    @safetestset "ITensorMPS" include("integration/itensormps.jl")
end
