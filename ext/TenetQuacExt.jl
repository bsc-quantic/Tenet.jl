module TenetQuacExt

using Tenet
using Quac: Quac, arraytype, Swap

function Base.convert(::Type{Tenet.Gate}, gate::Quac.Gate)
    return Tenet.Gate(arraytype(gate)(gate), Site[Site.(Quac.lanes(gate))..., Site.(Quac.lanes(gate); dual=true)...])
end

Tenet.evolve!(qtn::Tenet.AbstractAnsatz, gate::Quac.Gate; kwargs...) = evolve!(qtn, convert(Gate, gate); kwargs...)

function Base.convert(::Type{Tenet.Circuit}, circuit::Quac.Circuit)
    tenetcirc = Tenet.Circuit()

    for gate in circuit
        push!(tenetcirc, convert(Tenet.Gate, gate))
    end

    return tenetcirc
end

end
