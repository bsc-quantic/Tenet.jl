function expect(ψ, O; kwargs...)
    tn = stack(ψ, O, ψ')
    return contract(tn; kwargs...)
end
