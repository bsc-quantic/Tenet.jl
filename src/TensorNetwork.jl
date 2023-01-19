import OptimizedEinsum
import OptimizedEinsum: contractpath, Solver, Greedy, ContractionPath
using OMEinsum

abstract type TensorNetwork end

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) = print(io, "$(typeof(tn))(#tensors=$(length(tn)), #inds=$(length(inds(tn))))")

Base.length(x::TensorNetwork) = length(tensors(x))

function tensors end

arrays(tn::TensorNetwork) = parent.(tensors(tn))

function inds end

openinds(tn::TensorNetwork) = Iterators.filter(isopenind, inds(tn)) |> collect
hyperinds(tn::TensorNetwork) = Iterators.filter(ishyperind, inds(tn)) |> collect

labels(tn::TensorNetwork) = nameof.(inds(tn))

Base.size(tn::TensorNetwork) = Dict(nameof(i) => size(i) for i in inds(tn))

function contractpath(tn::TensorNetwork; solver = Greedy, output = openinds(tn), kwargs...)
    inputs = collect.(labels.(tensors(tn)))
    output = collect(output)
    size_dict = size(tn)

    contractpath(solver, inputs, output, size_dict)
end

function contract(tn::TensorNetwork; output = openinds(tn), kwargs...)
    path = OptimizedEinsum.contractpath(tn; output = output, kwargs...)

    # SSA-to-tensor mapping
    mapping = Dict{Int,Tensor}(i => t for (i, t) in enumerate(tensors(tn)))

    for (c, (a, b)) in zip(Iterators.countfrom(length(path.inputs) + 1), path)
        A = pop!(mapping, a)
        B = pop!(mapping, b)

        indsA = labels(A)
        indsB = labels(B)
        indsC = symdiff(indsA, indsB) ∪ ∩(output, indsA, indsB)

        C = EinCode((map(String, indsA), map(String, indsB)), tuple(map(String, indsC)...))(A, B)

        mapping[c] = Tensor(C, tuple(indsC...))
    end

    only(values(mapping))
end