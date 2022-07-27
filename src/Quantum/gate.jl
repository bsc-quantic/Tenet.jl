using LinearAlgebra
using WhereTraits
import Base: isapprox, adjoint

# TODO Call Diagonal on Matrix method on gates with Digonal method implemented. use @traits with hasmethod?

abstract type AbstractGate end

@traits lane(g::T) where {T<:AbstractGate,hasfield(T, :lane)} = g.lane
Matrix{T}(_::AbstractGate) where {T} = error("Implementation not found")

Base.adjoint(g::T) where {T<:AbstractGate} = Base.adjoint(T)(lane(g))

# pauli gates
struct I <: AbstractGate
    lane::Int
end

Matrix{T}(_::I) where {T} = Matrix{T}(LinearAlgebra.I, 2, 2)
Diagonal{T}(_::I) where {T} = Diagonal{T}(LinearAlgebra.I, 2)
Base.adjoint(::Type{I}) = I

struct X <: AbstractGate
    lane::Int
end

Matrix{T}(_::X) where {T} = Matrix{T}([0 1; 1 0])
Base.adjoint(::Type{X}) = X

struct Y <: AbstractGate
    lane::Int
end

Matrix{T}(_::Y) where {T<:Complex} = Matrix{T}([0 -1im; 1im 0])
Base.adjoint(::Type{Y}) = Y

struct Z <: AbstractGate
    lane::Int
end

Matrix{T}(_::Z) where {T} = Matrix{T}([1 0; 0 -1])
Diagonal{T}(_::Z) where {T} = Diagonal{T}([1, -1], 2)
Base.adjoint(::Type{Z}) = Z

struct H <: AbstractGate
    lane::Int
end

Matrix{T}(_::H) where {T} = 1 / sqrt(2) * Matrix{T}([1 1; 1 -1])
Base.adjoint(::Type{H}) = H

struct S <: AbstractGate
    lane::Int
end

Matrix{T}(_::S) where {T<:Complex} = Matrix{T}([1 0; 0 1im])
Diagonal{T}(_::S) where {T} = Diagonal{T}([1, 1im])
Base.adjoint(::Type{S}) = Sd

struct Sd <: AbstractGate
    lane::Int
end

Matrix{T}(_::Sd) where {T<:Complex} = Matrix{T}([1 0; 0 -1im])
Diagonal{T}(_::S) where {T} = Diagonal{T}([1, -1im])
Base.adjoint(::Type{Sd}) = S

struct T <: AbstractGate
    lane::Int
end

Matrix{T}(_::T) where {T<:Complex} = Matrix{T}([1 0; 0 cispi(1 // 4)])
Diagonal{T}(_::S) where {T} = Diagonal{T}([1, cispi(1 // 4)])
Base.adjoint(::Type{T}) = Td

struct Td <: AbstractGate
    lane::Int
end

Matrix{T}(_::Td) where {T<:Complex} = Matrix{T}([1 0; 0 cispi(-1 // 4)])
Diagonal{T}(_::S) where {T} = Diagonal{T}([1, cispi(-1 // 4)])
Base.adjoint(::Type{Td}) = T

# parametric gates
abstract type AbstractParametricGate <: AbstractGate end

isparametric(::Type{T}) where {T<:AbstractGate} = false
isparametric(::Type{T}) where {T<:AbstractParametricGate} = true

parameters(_::AbstractParametricGate) = error("Implementation not found")

struct Rx <: AbstractParametricGate
    lane::Int
    θ::Float32
end

parameters(g::Rx) = g.θ

Matrix{T}(g::Rx) where {T} = Matrix{T}([cos(g.θ / 2) -1im*sin(g.θ / 2); -1im*sin(g.θ / 2) cos(g.θ / 2)])
Base.adjoint(::Type{Rx}) = Rx
Base.adjoint(g::Rx) = Rx(g.lane, -g.θ)

struct Ry <: AbstractParametricGate
    lane::Int
    θ::Float32
end

parameters(g::Ry) = g.θ

Matrix{T}(g::Ry) where {T<:Complex} = Matrix{T}([cos(g.θ / 2) -sin(g.θ / 2); sin(g.θ / 2) cos(g.θ / 2)])
Base.adjoint(::Type{Ry}) = Ry
Base.adjoint(g::Ry) = Ry(g.lane, -g.θ)

struct Rz <: AbstractParametricGate
    lane::Int
    θ::Float32
end

parameters(g::Rz) = g.θ

Matrix{T}(g::Rz) where {T<:Complex} = Matrix{T}([1 0; 0 cis(g.θ)])
Base.adjoint(::Type{Rz}) = Rz
Base.adjoint(g::Rz) = Rz(g.lane, -g.θ)

U1 = Rz

struct U2 <: AbstractParametricGate
    lane::Int
    ϕ::Float32
    λ::Float32
end

parameters(g::U2) = (g.ϕ, g.λ)

Matrix{T}(g::U2) where {T<:Complex} = 1 / sqrt(2) * Matrix{T}([1 -cis(g.λ); cis(g.ϕ) cis(g.ϕ + g.λ)])
Base.adjoint(::Type{U2}) = U2
Base.adjoint(g::U2) = error("not implemented yet")

struct U3 <: AbstractParametricGate
    lane::Int
    θ::Float32
    ϕ::Float32
    λ::Float32
end

parameters(g::U3) = (g.θ, g.ϕ, g.λ)

Matrix{T}(g::U3) where {T<:Complex} = Matrix{T}([cos(g.θ / 2) -cis(g.λ)*sin(g.θ / 2); cis(g.ϕ)*sin(g.θ / 2) cis(g.ϕ + g.λ)*cos(g.θ / 2)])
Base.adjoint(::Type{U3}) = U3
Base.adjoint(g::U3) = error("not implemented yet")

# control gates
struct Control{T} <: AbstractGate where {T<:AbstractGate}
    lane::Int
    op::T
end

control(_::AbstractGate) = error("Implementation not found")
target(_::AbstractGate) = error("Implementation not found")

control(g::Control{T}) where {T} = g.lane
control(g::Control{T}) where {T<:Control} = (g.lane, control(g.op)...)
target(g::Control{T}) where {T} = lane(g.op)
target(g::Control{T}) where {T<:Control} = target(g.op)
lane(g::Control{T}) where {T} = (control(g)..., target(g)...)

Base.adjoint(::Type{Control{T}}) where {T<:AbstractGate} = Control{adjoint(T)}
Base.adjoint(g::Control{T}) where {T<:AbstractGate} = Control(g.lane, adjoint(g.op))

# special case for Control{T} where {T<:AbstractParametricGate}, as it is parametric
isparametric(::Type{Control{T}}) where {T<:AbstractParametricGate} = true
isparametric(::Type{Control{T}}) where {T<:Control} = isparametric(T)
parameters(g::Control{T}) where {T<:AbstractParametricGate} = parameters(g.op)
parameters(g::Control{T}) where {T<:Control} = parameters(g.op)

CX(control, target) = Control(control, X(target))
CY(control, target) = Control(control, Y(target))
CZ(control, target) = Control(control, Z(target))
CRx(control, target, θ) = Control(control, Rx(target, θ))
CRy(control, target, θ) = Control(control, Ry(target, θ))
CRz(control, target, θ) = Control(control, Rz(target, θ))

# multiqubit gates
struct Swap <: AbstractGate
    lane::NTuple{2,Int}
end

Matrix{T}(g::Swap) = Matrix{T}([1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1])
Base.adjoint(::Type{Swap}) = Swap
