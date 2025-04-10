struct Stack <: AbstractTensorNetwork
    tn::TensorNetwork
    pluggable::PluggableMixin
    layers::Vector{AbstractTensorNetwork}
end

function Stack(tn)
    # always retain copies of the TNs:
    # - Tensors are not copies, so no memory overhead
    # - but mutating changes can destroy higher-level mappings as in here
    tn = copy(tn)

    # reset indices to avoid conflicts between indices of any two TNs
    resetinds!(tn)

    pluggable = PluggableMixin()
    for site in sites(tn)
        addsite!(pluggable, site, inds(tn; at=site))
    end
    return Stack(TensorNetwork(tensors(tn)), pluggable, [tn])
end

matching_plug_lanes(a, b) = Lane.(sites(a; set=:outputs)) ∩ Lane.(sites(b; set=:inputs))

function Base.push!(stn::Stack, tn)
    @assert isconnectable(stn, tn) "Cannot merge the two Tensor Networks: they are not connectable"

    tn = copy(tn)
    resetinds!(tn)

    # match physical indices
    @align! outputs(stn) => inputs(tn)

    # add the new TN to the stack
    append!(stn.tn, tensors(tn))
    push!(stn.layers, tn)

    # update site-index mapping
    for lane in matching_plug_lanes(stn, tn)
        rmsite!(stn, Site(lane))
    end

    for site in sites(tn; set=:outputs)
        addsite!(stn, site, inds(tn; at=site))
    end

    return stn
end

function Base.stack(tns::AbstractTensorNetwork...)
    stn = Stack(tns[1])
    for i in 2:length(tns)
        push!(stn, tns[i])
    end
    return stn
end

nlayers(tn::Stack) = length(tn.layers)
layer(tn::Stack, i) = tn.layers[i]
layers(tn::Stack) = tn.layers

trait(::TensorNetworkInterface, ::Stack) = WrapsTensorNetwork()
unwrap(::TensorNetworkInterface, tn::Stack) = tn.tn

trait(::PluggableInterface, ::Stack) = WrapsPluggable()
unwrap(::PluggableInterface, tn::Stack) = tn.pluggable

Base.copy(tn::Stack) = Stack(copy(tn.tn), copy(tn.pluggable), copy(tn.layers))

function handle!(tn::Stack, effect::ReplaceEffect{Pair{Tensor,Tensor}})
    # reflect the effect on the underlying tensor network
    handle!(unwrap(TensorNetworkInterface(), tn), effect)

    # propagate the effect to the layers
    for layer in layers(tn)
        handle!(layer, effect)
    end
end

function adjoint_sites!(tn::Stack)
    # adjoint sites of Stack itself (pass it to mixin)
    adjoint_sites!(unwrap(PluggableInterface(), tn))

    # adjoint sites of layers
    for layer in layers(tn)
        adjoint_sites!(layer)
    end

    # reverse the order of the layers
    reverse!(tn.layers)

    return tn
end
