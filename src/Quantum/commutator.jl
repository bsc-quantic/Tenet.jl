using WhereTraits

commutator(A::Matrix, B::Matrix) = A * B - B * A
anticommutator(A::Matrix, B::Matrix) = A * B + B * A

commutes(A::Matrix{T}, B::Matrix{T}) where {T<:Number} = A * B == B * A
commutes(A::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat} = Base.isapprox(A * B, B * A)
commutes(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}) where {T<:AbstractFloat} = Base.isapprox(A * B, B * A)

anticommutes(A::Matrix{T}, B::Matrix{T}) where {T<:Number} = A * B == -B * A
anticommutes(A::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat} = Base.isapprox(A * B, -B * A)
anticommutes(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}) where {T<:AbstractFloat} = Base.isapprox(A * B, -B * A)

# if sites where they operate do not overlap, then they must commute
@traits commutes(A::AbstractGate, B::AbstractGate) where {isempty(intersect(Set(lane(A)), Set(lane(B))))} = true
@traits anticommutes(A::AbstractGate, B::AbstractGate) where {isempty(intersect(Set(lane(A)), Set(lane(B))))} = false

# any gate commutes with itself
_commutes(A::T, B::T) where {T<:AbstractGate} = true
_anticommutes(A::T, B::T) where {T<:AbstractGate} = false

# identity commutes with everything
_commutes(A::I, B::T) where {T<:AbstractGate} = true
_anticommutes(A::I, B::T) where {T<:AbstractGate} = false

# rotations
_commutes(A::X, B::Rx; atol::Real = 0.0) = isapprox(B.θ, π; atol=atol)
_commutes(A::Y, B::Ry; atol::Real = 0.0) = isapprox(B.θ, π; atol=atol)
_commutes(A::Z, B::Rz; atol::Real = 0.0) = isapprox(B.θ, π; atol=atol)