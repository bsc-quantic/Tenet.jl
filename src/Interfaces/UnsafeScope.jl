# UnsafeScope
struct UnsafeScope
    refs::Vector{WeakRef}

    UnsafeScope() = new(Vector{WeakRef}())
end

Base.values(uc::UnsafeScope) = map(x -> x.value, uc.refs)

is_unsafe_scoped(tn, uc::UnsafeScope) = tn ∈ values(uc)

# TODO document that a `TensorNetworkInterface` implementor must implement these methods
function get_unsafe_scope end
function set_unsafe_scope! end

macro unsafe_region(tn_sym, block)
    return esc(
        quote
            local old = copy($tn_sym)

            # Create a new UnsafeScope and set it to the current tn
            local _uc = Tenet.UnsafeScope()
            Tenet.set_unsafe_scope!($tn_sym, _uc)

            # Register the tensor network in the UnsafeScope
            push!(Tenet.get_unsafe_scope($tn_sym).refs, WeakRef($tn_sym))

            e = nothing
            try
                $(block) # Execute the user-provided block
            catch e
                $(tn_sym) = old # Restore the original tensor network in case of an exception
                rethrow(e)
            finally
                if isnothing(e)
                    # Perform checks of registered tensor networks
                    for ref in Tenet.get_unsafe_scope($tn_sym).refs
                        tn = ref.value
                        if !isnothing(tn) && tn ∈ values(Tenet.get_unsafe_scope($tn_sym))
                            if !Tenet.checksizes(tn)
                                $(tn_sym) = old

                                # Set `unsafe` field to `nothing`
                                Tenet.set_unsafe_scope!($tn_sym, nothing)

                                throw(DimensionMismatch("Inconsistent size of indices"))
                            end
                        end
                    end
                end
            end
        end,
    )
end
