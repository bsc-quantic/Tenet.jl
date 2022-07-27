commutator(A::Matrix, B::Matrix) = A * B - B * A
anticommutator(A::Matrix, B::Matrix) = A * B + B * A

commutator(A::AbstractGate, B::AbstractGate) = begin
    if isempty(intersect(Set(lane(A)), Set(lane(B))))
        return zeros(...)
    end
    mA = Matrix(A)
    mB = Matrix(B)

    mA * mB - mB * mA
end
commutator(A, B::AbstractGate) = commutator(B, A)

commutator(A, B::I) = zeros(2, 2)
commutator(A::T, B::T) where {T<:AbstractGate} = zeros(2, 2)

commutes(A::Matrix{T}, B::Matrix{T}) where {T<:Number} = A * B == B * A
commutes(A::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat} = Base.approx(A * B, B * A)