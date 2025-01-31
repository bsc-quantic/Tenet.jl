using Test
using Tenet
using Distributed
using Dagger

addprocs(1)
@everywhere using Dagger, Tenet

try
    @testset "Tensor" begin
        data = rand(4, 4)
        block_array = DArray(data, Dagger.Blocks(2, 2))
        indices = (:i, :j)

        tensor = Tensor(data, indices)
        block_tensor = Tensor(block_array, indices)

        @test inds(block_tensor) == inds(tensor)
        @test Array(parent(block_tensor)) ≈ parent(tensor)
    end

    @testset "contract" begin
        @testset "block-block" begin
            data1, data2 = rand(4, 4), rand(4, 4)
            block_array1 = distribute(data1, Dagger.Blocks(2, 2))
            block_array2 = distribute(data2, Dagger.Blocks(2, 2))

            tensor1 = Tensor(data1, [:i, :j])
            tensor2 = Tensor(data2, [:j, :k])
            block_tensor1 = Tensor(block_array1, [:i, :j])
            block_tensor2 = Tensor(block_array2, [:j, :k])

            contracted_tensor = contract(tensor1, tensor2)
            contracted_block_tensor = contract(block_tensor1, block_tensor2)

            @test parent(contracted_block_tensor) isa DArray
            @test inds(contracted_block_tensor) == (:i, :k)
            @test all(==((2, 2)) ∘ size, Dagger.domainchunks(parent(contracted_block_tensor)))
            @test collect(parent(contracted_block_tensor)) ≈ parent(contracted_tensor)
        end
    end
finally
    rmprocs(workers())
end
