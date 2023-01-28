using Test
import Tenet

@testset "Unit tests" verbose = true begin
    include("Helpers_test.jl")
    include("Einsum_test.jl")
    include("Tensor_test.jl")
    include("Index_test.jl")
    include("TensorNetwork_test.jl")
    include("Quantum_test.jl")
    include("Transformations_test.jl")
end

@testset "Integration tests" verbose = true begin
    include("Integration/Quac_test.jl")
end