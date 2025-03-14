# This file defines utilities to work with "effects"; i.e. mutation signals sent from low-level types to higher-level wrapping types

"""
    Effect

Abstract type for effects.
"""
abstract type Effect end

"""
    canhandle(x, effect::Effect)

Returns `true` if `x` can handle the effect.
"""
canhandle(::T, ::E) where {T,E} = canhandle(T, E)
canhandle(::Type{T}, ::Type{E}) where {T,E} = hasmethod(handle!, Tuple{T,E})

function checkhandle(x::T, effect::E) where {T,E}
    if !canhandle(x, effect)
        throw(ArgumentError("$T cannot handle effect $E"))
    end
end

"""
    handle!(x, effect::Effect)

Handle the `effect` on `x`. By default, does nothing.
"""
function handle! end

"""
    PushEffect{F} <: Effect

Represents the effect of pushing an object.
"""
struct PushEffect{F} <: Effect
    f::F
end

PushEffect(@nospecialize(f::Tensor)) = PushEffect{Tensor}(f)

"""
    DeleteEffect{F} <: Effect

Represents the effect of deleting an object.
"""
struct DeleteEffect{F} <: Effect
    f::F
end

DeleteEffect(@nospecialize(f::Tensor)) = DeleteEffect{Tensor}(f)

"""
    ReplaceEffect{F} <: Effect

Represents the effect of replacing an object with a new one.
"""
struct ReplaceEffect{F} <: Effect
    f::F
end

ReplaceEffect(@nospecialize(f::Pair{<:Tensor,<:Tensor})) = ReplaceEffect{Pair{Tensor,Tensor}}(f)
ReplaceEffect(@nospecialize(f::Pair{<:Tensor,TN})) where {TN} = ReplaceEffect{Pair{Tensor,TN}}(f)
ReplaceEffect(f::Pair{Symbol,Symbol}) = ReplaceEffect{Pair{Symbol,Symbol}}(f)
