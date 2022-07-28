import Base: adjoint

struct Circuit
    nodes::Dict{Base.UUID,AbstractGate}
end

Base.adjoint(circuit::Circuit) = error("not implemented")

