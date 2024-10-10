"""
    einsum

Compute the unary or pairwise contraction of [`Tensor`](@ref)s.
"""
function einsum end

# unary einsum
function einsum!(c, ic, a, ia) end

# binary einsum
function einsum!(c, ic, a, ia, b, ib) end
