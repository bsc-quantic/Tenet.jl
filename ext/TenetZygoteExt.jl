module TenetZygoteExt

using Tenet
using Zygote

Zygote.@adjoint (T::Type{<:Tensor})(data, inds; meta...) = T(data, inds; meta...), y -> (nothing, y.data, nothing)

# WARN type-piracy
Zygote.@adjoint Base.setdiff(s, itrs...) =
    setdiff(s, itrs...), _ -> (nothing, nothing, [nothing for _ in 1:length(itrs)]...)
Zygote.@adjoint Base.union(s, itrs...) =
    union(s, itrs...), _ -> (nothing, nothing, [nothing for _ in 1:length(itrs)]...)
Zygote.@adjoint Base.intersect(s, itrs...) =
    intersect(s, itrs...), _ -> (nothing, nothing, [nothing for _ in 1:length(itrs)]...)

end