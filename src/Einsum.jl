import Permutations

abstract type EinsumOperation end

"""
	parse([::Type{<:EinsumOperation}], output, inputs[...])
"""
function parse end

abstract type OuterProduct <: EinsumOperation end
abstract type InnerProduct <: EinsumOperation end
abstract type Trace <: EinsumOperation end
abstract type Permutation <: EinsumOperation end

parse(::Type{OuterProduct}, output, inputs...) = symdiff(inputs...) ∩ output
parse(::Type{InnerProduct}, output, inputs...) = setdiff(∩(inputs...), output)
parse(::Type{Trace}, output, input) = (xs = sort(input); [a for (a, b) ∈ zip(xs, xs[2:end]) if a == b])
parse(::Type{Permutation}, output, input) = Permutations.Permutation([findfirst(output .== ind) for ind in input])

function parse(output, inputs...)
    d = Dict{EinsumOperation,Set{Symbol}}()

    if length(inputs) == 1

        for einop in [Trace, Permutation]
            d[einop] = mapreduce(∪, inputs) do input
                parse(einop, output, input)
            end
        end

    elseif length(inputs) == 2

        for einop in [OuterProduct, InnerProduct]
            d[einop] = parse(einop, output, inputs...)
        end

    elseif length(inputs) > 2
        throw(ArgumentError("`parse` is not prepared for more than inputs"))
    end

    return d
end