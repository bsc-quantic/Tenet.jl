"""
    gramschmidt!(A[, p; atol])

Creates an orthogonal basis from a random matrix `A`.
"""
function gramschmidt!(A::AbstractMatrix{T}, p::Real = 2; atol::Real = 1e-9) where {T}
    m = size(A, 1)
    for i in 1:m-1
        # `v` is used as reference
        v = @view A[i:i, :] # NOTE `i:i` to keep it being a row vector
        normalize!(v)

        sA = @view A[i+1:end, :]

        # `u` is the vector of projections
        u = sA * v'

        while norm(u, p) > atol
            # remove `v` component with a rank-1 update
            BLAS.ger!(-one(T), vec(u), vec(conj(v)), sA)
            u .= sA * v'
        end
    end

    # normalize last row
    eachrow(A) |> last |> normalize!

    return A
end
