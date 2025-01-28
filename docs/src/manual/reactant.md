# Acceleration with Reactant.jl

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) is a new MLIR & XLA frontend for the Julia language. It's similar to JAX, in the sense that it traces some code, compiles array operations using the XLA compiler and can run the final compiled function in CPU, GPU or TPU.

Tenet.jl has top-class integration with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). Let's dive in on how to combine both Tenet.jl and Reactant.jl with a simple example: compute an expectation value between a MPS and a MPO.

Let's first initialize the state $\psi$ and the operator $H$ randomly:

```julia
julia> using Tenet

julia> H = rand(MPO; n=10, maxdim=32)
MPO (inputs=10, outputs=10)

julia> ψ = rand(MPS; n=10, maxdim=8)
MPS (inputs=0, outputs=10)

julia> expect(ψ, H)
0-dimensional Tensor{Float64, 0, Array{Float64, 0}}:
-0.0024015322172776864
```

In order to be able to transparently move between CPU and GPU executions, Reactant.jl needs to manage the memory itself through its own `Array` implementation: `ConcreteRArray`.
Moreover, these `ConcreteRArray`s are used to know which array operations need to be traced.

Thanks to its integration with Adapt.jl, it's super easy to convert deep, nested data-structures containing arrays (like [`TensorNetwork`](@ref)):

```julia
julia> using Reactant, Adapt

julia> ψ_re = adapt(ConcreteRArray, ψ)
MPS (inputs=0, outputs=10)

julia> H_re = adapt(ConcreteRArray, H)
MPO (inputs=10, outputs=10)
```

In principle, all you need to do for Reactant.jl to do it's magic is just call `Reactant.compile` or `@compile`:

```julia
julia> expect_re = @compile expect(ψ_re, H_re)
Reactant.Compiler.Thunk{typeof(expect), Symbol("##expect_reactant#1880"), Tuple{MPS, MPO}, false}(expect)

julia> expect_re(ψ_re, H_re)
0-dimensional Tensor{Float64, 0, ConcreteRArray{Float64, 0}}:
-0.0024015322172776838
```

Even if the first compilation can take a while (which we are addressing and will go faster in future versions), the speedup obtained by the compiled function compesates it if it's called many times:

```julia
julia> @b expect(ψ, H)
17.796 ms (208204 allocs: 11.031 MiB)

julia> @b expect_re(ψ_re, H_re)
353.959 μs (16 allocs: 688 bytes)
```

!!! note
    The number of allocations and memory usage of a Reactant.jl-compile function reported by `@time`, `BenchmarkTools.@btime` and `Chairmarks.@time` are not representative of the real execution. These functions only report heap allocations performed by Julia code, which Reactant.jl generated MLIR code is not.
    In any case, Reactant.jl can optimize the code to reduce allocations so the number of allocations **should** be less than the non-compiled code.

!!! warning
    Reactant.jl does not support matrix factorizations like SVD, QR or Eigendecomposition yet. There is an issue tracking this in [Reactant.jl#336](https://github.com/EnzymeAD/Reactant.jl/issues/336).
