@testset "Product ansatz" begin
    using LinearAlgebra

    # TODO test `Product` with `Scalar` socket

    qtn = Product([rand(2) for _ in 1:3])
    @test socket(qtn) == State()
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 3
    @test norm(qtn) isa Number
    @test begin
        normalize!(qtn)
        norm(qtn) ≈ 1
    end

    # conversion to `Quantum`
    @test Quantum(qtn) isa Quantum

    qtn = Product([rand(2, 2) for _ in 1:3])
    @test socket(qtn) == Operator()
    @test ninputs(qtn) == 3
    @test noutputs(qtn) == 3
    @test norm(qtn) isa Number
    @test opnorm(qtn) isa Number
    @test begin
        normalize!(qtn)
        norm(qtn) ≈ 1
    end
end
