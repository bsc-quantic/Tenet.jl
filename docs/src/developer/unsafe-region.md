# Unsafe regions

There are cases in which you may want to temporarily avoid index size checks on `push!` to a [`TensorNetwork`](@ref).

```julia
@unsafe_region tn begin
    ...
end
```
