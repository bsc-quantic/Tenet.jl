
"""
    simple_update!(tn, gate; threshold = nothing, maxdim = nothing, kwargs...)

Update a Tensor Network with a [`Gate`](@ref) operator using the "Simple Update" algorithm.
`kwargs` are passed to the `truncate!` method in the case of a multi-site gate.

!!! warning

    Currently only 1-site and 2-site gates are supported.

# Arguments

  - `ψ`: Tensor Network representing the state.
  - `gate`: The gate operator to update the state with.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `normalize`: Whether to normalize the state after truncation.

# Notes

  - If both `threshold` and `maxdim` are provided, `maxdim` is used.
"""
function simple_update!(ψ, gate::Gate; kwargs...)
    @assert issetequal(adjoint.(sites(gate; set=:inputs)), sites(gate; set=:outputs)) "Inputs of the gate must match outputs"
    @assert isconnectable(ψ, gate) "Gate must be connectable to the Tensor Network"

    align!(ψ => gate)

    if nlanes(gate) == 1
        return simple_update_1site!(ψ, gate)
    elseif nlanes(gate) == 2
        bond = Bond(lanes(gate)...)
        hasbond(ψ, bond) || throw(ArgumentError("Gate must act on neighboring sites of the lattice"))
        return simple_update_2site!(form(ψ), ψ, gate; kwargs...)
    else
        throw(ArgumentError("Only 1-site and 2-site gates are currently supported"))
    end
end

function simple_update_1site!(tn, gate::Gate)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    # reindex output of gate to match TN sitemap
    align!(tn => gate)

    # contract gate with target tensor
    target_lane = only(lanes(gate))
    pind = inds(tn; at=Site(target_lane))
    pind_tmp = inds(gate; at=Site(target_lane))
    target_tensor = tensors(tn; at=target_lane)

    new_tensor = contract(target_tensor, parent(gate))

    # replace target tensor with new tensor
    new_tensor = replace(new_tensor, pind_tmp => pind)
    replace!(tn, target_tensor => new_tensor)

    return tn
end

################################################################################

function simple_update_2site!(::MixedCanonical, ψ, gate; kwargs...)
    canonize!(ψ, MixedCanonical(collect(lanes(gate))))
    return simple_update_2site!(NonCanonical(), ψ, gate; kwargs...)
end

function simple_update_2site!(::NonCanonical, ψ, gate; kwargs...)
    lanel, laner = minmax(lanes(gate)...)
    acting_bond = Bond(lanel, laner)

    @assert haslane(ψ, lanel)
    @assert haslane(ψ, laner)
    @assert hasbond(ψ, acting_bond)

    @assert inds(ψ; at=Site(lanel)) === inds(gate; at=Site(lanel; dual=true))
    @assert inds(ψ; at=Site(laner)) === inds(gate; at=Site(laner; dual=true))

    A = tensors(ψ; at=lanel)
    B = tensors(ψ; at=laner)

    U, V = Operations.simple_update(
        A,
        inds(ψ; at=Site(lanel)),
        B,
        inds(ψ; at=Site(laner)),
        inds(ψ; bond=Bond(lanel, laner)),
        Tensor(gate),
        inds(gate; at=Site(lanel)),
        inds(gate; at=Site(laner));
        absorb=Operations.AbsorbEqually(),
        normalize=false,
        kwargs...,
    )

    @unsafe_region ψ begin
        replace!(ψ, A => U)
        replace!(ψ, B => V)
    end

    return ψ
end

# TODO remove `normalize` argument?
function simple_update_2site!(::Canonical, ψ, gate; kwargs...)
    lanel, laner = minmax(lanes(gate)...)
    acting_bond = Bond(lanel, laner)

    @assert haslane(ψ, lanel)
    @assert haslane(ψ, laner)
    @assert hasbond(ψ, acting_bond)

    Γl = tensors(ψ; at=lanel)
    Γr = tensors(ψ; at=laner)

    # contract inner Λ
    Λ = tensors(ψ; bond=Bond(lanel, laner))
    Al = contract(Λ, Γl; dims=())

    # contract outer Λ
    for bond in filter(!=(acting_bond), bonds(ψ, lanel))
        Λb = tensors(ψ; bond)
        Al = contract(Al, Λb; dims=())
    end

    for bond in filter(!=(acting_bond), bonds(ψ, laner))
        Λb = tensors(ψ; bond)
        Γr = contract(Γr, Λb; dims=())
    end

    # perform "simple update" procedure
    Bl, BΛ, Br = Operations.simple_update(
        Al,
        inds(ψ; at=Site(lanel)),
        Γr,
        inds(ψ; at=Site(laner)),
        inds(ψ; bond=Bond(lanel, laner)),
        Tensor(gate),
        inds(gate; at=Site(lanel)),
        inds(gate; at=Site(laner));
        absorb=Operations.DontAbsorb(),
        normalize=false,
        kwargs...,
    )

    # Λᵢ₋₁ = id(lanel) == 1 ? nothing : tensors(ψ; bond=(Lane(id(lanel) - 1), lanel))
    # Λᵢ₊₁ = id(lanel) == nsites(ψ) - 1 ? nothing : tensors(ψ; bond=(laner, Lane(id(laner) + 1)))

    # !isnothing(Λᵢ₋₁) && absorb!(ψ; bond=(Lane(id(lanel) - 1), lanel), dir=:right, delete_Λ=false)
    # !isnothing(Λᵢ₊₁) && absorb!(ψ; bond=(laner, Lane(id(laner) + 1)), dir=:left, delete_Λ=false)

    # simple_update_2site!(NonCanonical(), ψ, gate; threshold, maxdim, normalize=false, canonize=false)

    # # contract the updated tensors with the inverse of Λᵢ and Λᵢ₊₂, to get the new Γ tensors
    # U, Vt = tensors(ψ; at=lanel), tensors(ψ; at=laner)
    # Γᵢ₋₁ = if isnothing(Λᵢ₋₁)
    #     U
    # else
    #     contract(U, Tensor(diag(pinv(Diagonal(parent(Λᵢ₋₁)); atol=wrap_eps(eltype(U)))), inds(Λᵢ₋₁)); dims=())
    # end
    # Γᵢ = if isnothing(Λᵢ₊₁)
    #     Vt
    # else
    #     contract(Tensor(diag(pinv(Diagonal(parent(Λᵢ₊₁)); atol=wrap_eps(eltype(Vt)))), inds(Λᵢ₊₁)), Vt; dims=())
    # end

    # # Update the tensors in the tensor network
    # replace!(ψ, tensors(ψ; at=lanel) => Γᵢ₋₁)
    # replace!(ψ, tensors(ψ; at=laner) => Γᵢ)

    # if canonize
    #     canonize!(ψ; normalize)
    # else
    #     normalize && normalize!(ψ, collect((lanel, laner)))
    # end

    return ψ
end
