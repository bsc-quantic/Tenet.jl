# Unsafe regions

In order to avoid inconsistency issues, [`TensorNetwork`](@ref) checks the index sizes are correct whenever a [`Tensor`](@ref) is [`push!`](@ref)ed and it already contains some of the its indices.
There are cases in which you may want to temporarily avoid index size checks (for performance or for ergonomy) on `push!` to a [`TensorNetwork`](@ref).
But mutating a [`TensorNetwork`](@ref) without checks is dangerous, as it can leave it in a inconsistent state which would lead to hard to trace errors.

Instead, we developed the `@unsafe_region` macro. The first argument is the [`AbstractTensorNetwork`](@ref) you want to disable the checks for, and the second argument is the code where you modify the [`AbstractTensorNetwork`](@ref) without checks.

```julia
@unsafe_region tn begin
    ...
end
```

When the scope of the `@unsafe_region` ends, it will automatically run a full check on `tn` to assert that the final state of the [`AbstractTensorNetwork`](@ref) is consistent.

Note that this only affects disables the checks for one [`AbstractTensorNetwork`](@ref), but multiple `@unsafe_region`s can be nested.
