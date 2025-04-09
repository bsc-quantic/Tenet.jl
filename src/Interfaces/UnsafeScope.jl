# UnsafeScope
struct UnsafeScope
    refs::Vector{WeakRef}

    UnsafeScope() = new(Vector{WeakRef}())
end

Base.values(uc::UnsafeScope) = map(x -> x.value, uc.refs)
Base.push!(uc::UnsafeScope, ref::WeakRef) = push!(uc.refs, ref)
Base.push!(uc::UnsafeScope, tn) = push!(uc.refs, WeakRef(tn))

inscope(tn, uc::UnsafeScope) = tn ∈ uc.refs
inscope(tn, ::Nothing) = false

isscoped(tn) = inscope(tn, get_unsafe_scope(tn))

# TODO document that a `TensorNetworkInterface` implementor must implement these methods
function get_unsafe_scope end
function set_unsafe_scope! end

macro unsafe_region(tn, block)
    return esc(
        quote
            local old = copy($tn)

            # Create a new UnsafeScope and set it to the current tn
            local _uc = Tenet.UnsafeScope()
            Tenet.set_unsafe_scope!($tn, _uc)

            # Register the tensor network in the UnsafeScope
            push!(Tenet.get_unsafe_scope($tn).refs, WeakRef($tn))

            e = nothing
            try
                $block # Execute the user-provided block
            catch e
                $tn = old # Restore the original tensor network in case of an exception
                rethrow(e)
            finally
                if isnothing(e)
                    # Perform checks of registered tensor networks
                    for ref in values(Tenet.get_unsafe_scope($tn))
                        if !isnothing(ref) && ref ∈ Tenet.get_unsafe_scope($tn).refs
                            if !Tenet.checksizes(ref)
                                $tn = old

                                # Set `unsafe` field to `nothing`
                                Tenet.set_unsafe_scope!($tn, nothing)

                                throw(DimensionMismatch("Inconsistent size of indices"))
                            end
                        end
                    end
                end
            end
        end,
    )
end
