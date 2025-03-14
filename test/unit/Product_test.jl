using Test
using Tenet
using Tenet: socket, State, Operator
using LinearAlgebra

# TODO test `Product` with `Scalar` socket

qtn = ProductState([rand(2) for _ in 1:3])
@test socket(qtn) == State()
@test nsites(qtn; set=:inputs) == 0
@test nsites(qtn; set=:outputs) == 3
@test norm(qtn) isa Number
@test adjoint(qtn) isa ProductState
@test socket(adjoint(qtn)) == State(; dual=true)
@test norm(qtn) ≈ norm(contract(tensors(qtn)))
@test let qtn = normalize(qtn)
    norm(qtn) ≈ 1 && norm(contract(tensors(qtn))) ≈ 1
end

qtn = ProductOperator([rand(2, 2) for _ in 1:3])
@test socket(qtn) == Operator()
@test nsites(qtn; set=:inputs) == 3
@test nsites(qtn; set=:outputs) == 3
@test norm(qtn) isa Number
@test opnorm(qtn) isa Number
@test adjoint(qtn) isa ProductOperator
@test socket(adjoint(qtn)) == Operator()
@test norm(qtn) ≈ norm(contract(tensors(qtn)))
@test let qtn = normalize(qtn)
    norm(qtn) ≈ 1 && norm(contract(tensors(qtn))) ≈ 1
end
