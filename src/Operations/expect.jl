""" Expectation value for a one-site operator `op` on site `op_site` of mps `psi`, brings `psi` to canonical form. """
function expect_1site_can!(psi::MPS, op::AbstractArray, op_site::Int)

        canonize!(psi, MixedCanonical(site"$(op_site)"))
        plug_ind = ind_at(psi, plug"$(op_site)")
        temp_ind = Index(:_temp_bra)
        ev = binary_einsum(psi[op_site], Tensor(op, [temp_ind, plug_ind]))
        ev = binary_einsum(ev, conj(replace(psi[op_site], plug_ind => temp_ind)))

        return ev
end

""" Expectation value for a one-site operator `op` on site `op_site` of mps `psi`.
This makes a copy of the MPS """
function expect_1site(psi::MPS, op::AbstractArray, op_site::Int)
    if CanonicalForm(psi) isa MixedCanonical
        expect_1site_can!(psi, op, op_site)
    elseif CanonicalForm(psi) isa NonCanonical
        ev = Tensor(1.)
        for jj = 1:op_site-1
            ev = binary_einsum(ev, psi[jj])
            ev = binary_einsum(ev, conj(psi[jj]))
        end
        ev = binary_einsum(ev, psi[op_site])

        temp_ind = Index(:_temp)
        plug_ind = ind_at(psi, plug"$(op_site)")
        ev = replace(binary_einsum(ev, Tensor(op, [temp_ind, plug_ind])), temp_ind => plug_ind)
        ev = binary_einsum(ev, conj(psi[op_site]))
        for jj = op_site+1:nsites(psi)
            ev = binary_einsum(ev, psi[jj])
            ev = binary_einsum(ev, conj(psi[jj]))
        end
        ev
    else
        error("Not implemented yet")
    end
end

function expect_1site_alt(psi::MPS, op::AbstractArray, op_site::Int; kwargs...)
    fallback(overlap)
    phi = resetinds!(conj(psi))
    align!(psi, :outputs, phi, :outputs)
    temp_ind = Index(:_temp)
    plug_ind = ind_at(psi, plug"$(op_site)")
    phi[op_site] = replace(phi[op_site], plug_ind => temp_ind)
    tn = GenericTensorNetwork()
    push!(tn, Tensor(op, [plug_ind, temp_ind]))
    append!(tn, all_tensors(psi))
    append!(tn, all_tensors(phi))
    return contract(tn; kwargs...)
end

""" zipper contraction for MPS-MPO-MPS. Does *not* conjugate anything as of now """
function zipcontract(psi::MPS, o::MPO, phi::MPS)
    resetinds!(o)
    resetinds!(phi)
    align!(psi, :outputs, o, :inputs)
    align!(o, :outputs, phi, :outputs)

    result = Tensor(1.)
    for i = 1:nsites(psi)
        result = binary_einsum(result, psi[i])
        result = binary_einsum(result, o[i])
        result = binary_einsum(result, phi[i])
    end

    only(result)
  
end
function zipcontract(psi::MPS, phi::MPS)
    resetinds!(phi)
    align!(psi, :outputs, phi, :outputs)

    result = Tensor(1.)
    for i = 1:nsites(psi)
        result = binary_einsum(result, psi[i])
        result = binary_einsum(result, phi[i])
    end

    only(result)
  
end

# TODO fix interface:  expect(psi, o) or expect(o, psi) ? 
function expect(psi::MPS, o::MPO)
    zipcontract(psi, o, conj(psi))
end