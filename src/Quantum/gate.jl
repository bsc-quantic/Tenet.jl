using LinearAlgebra
using WhereTraits

abstract type AbstractGate end

lane(_::AbstractGate) = error("Implementation not found")
@traits lane(g::T) where {T<:AbstractGate,hasfield(T, :lane)} = g.lane

control(_::AbstractGate) = error("Implementation not found")
target(_::AbstractGate) = error("Implementation not found")
parameter(_::AbstractGate) = error("Implementation not found")
Matrix{T}(_::AbstractGate) = error("Implementation not found")

struct I <: AbstractGate
    lane::UInt
end

Matrix{T}(g::I) where {T} = LinearAlgebra.I # TODO

struct X <: AbstractGate
    lane::UInt
end

Matrix{T}(g::X) where {T} = Matrix{T}([0 1; 1 0])

struct Y <: AbstractGate
    lane::UInt
end

Matrix{T}(g::Y) where {T<:Complex} = Matrix{T}([0 -1im; 1im 0])

struct Z <: AbstractGate
    lane::UInt
end

Matrix{T}(g::Z) where {T} = Matrix{T}([1 0; 0 -1])

struct Rx <: AbstractGate
    lane::UInt
    θ::Float32
end

Matrix{T}(g::Rx) where {T} = ...

struct Ry <: AbstractGate
    lane::UInt
    θ::Float32
end

Matrix{T}(g::Ry) where {T} = ...

struct Rz <: AbstractGate
    lane::UInt
    θ::Float32
end

Matrix{T}(g::Rz) where {T} = Matrix{T}([1 0; 0 cis(g.θ)])

struct H <: AbstractGate
    lane::UInt
end

Matrix{T}(g::H) where {T} = 1 / sqrt(2) * Matrix{T}([1 1; 1 -1])

struct S <: AbstractGate
    lane::UInt
end

Matrix{T}(g::S) where {T} = ...

struct Sd <: AbstractGate
    lane::UInt
end

Matrix{T}(g::_) where {T} = ...

struct T <: AbstractGate
    lane::UInt
end

Matrix{T}(g::_) where {T} = ...

struct Td <: AbstractGate
    lane::UInt
end

Matrix{T}(g::_) where {T} = ...

struct U3 <: AbstractGate
    lane::UInt
    θ::Float32
    ϕ::Float32
    γ::Float32
end

Matrix{T}(g::_) where {T} = ...

struct Control{T} <: AbstractGate where {T<:AbstractGate}
    lane::UInt
    op::AbstractGate
end

control(g::Control{T}) where {T} = g.lane
target(g::Control{T}) where {T} = lane(g.op)
lane(g::Control{T}) where {T} = (control(g), target(g)...)

CX(control, target) = Control(control, X(target))
CY(control, target) = Control(control, Y(target))
CZ(control, target) = Control(control, Z(target))
CRx(control, target, θ) = Control(control, Rx(target, θ))
CRy(control, target, θ) = Control(control, Ry(target, θ))
CRz(control, target, θ) = Control(control, Rz(target, θ))
