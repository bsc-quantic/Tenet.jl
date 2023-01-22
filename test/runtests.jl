using Test
import Tenet

@testset "Unit tests" verbose = true begin
    include("Helpers_test.jl")
    include("Einsum_test.jl")
    include("Tensor_test.jl")
    include("Index_test.jl")
end