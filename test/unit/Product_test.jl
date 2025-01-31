using LinearAlgebra
using Tenet: nsites, State, Operator

@testset "Product" begin
    # TODO test `Product` with `Scalar` socket

    qtn = Product([rand(2) for _ in 1:3])
    @test socket(qtn) == State()
    @test nsites(qtn; set=:inputs) == 0
    @test nsites(qtn; set=:outputs) == 3
    @test norm(qtn) isa Number
    @test begin
        normalize!(qtn)
        norm(qtn) ≈ 1
    end
    @test adjoint(qtn) isa Product
    @test socket(adjoint(qtn)) == State(; dual=true)
    @test Ansatz(qtn) isa Ansatz

    qtn = Product([rand(2, 2) for _ in 1:3])
    @test socket(qtn) == Operator()
    @test nsites(qtn; set=:inputs) == 3
    @test nsites(qtn; set=:outputs) == 3
    @test norm(qtn) isa Number
    @test opnorm(qtn) isa Number
    @test begin
        normalize!(qtn)
        norm(qtn) ≈ 1
    end
    @test adjoint(qtn) isa Product
    @test socket(adjoint(qtn)) == Operator()
    @test Ansatz(qtn) isa Ansatz
end
