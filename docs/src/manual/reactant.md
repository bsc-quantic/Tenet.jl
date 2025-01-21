# Compilation with Reactant.jl

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) is a new MLIR & XLA frontend for the Julia language. It's similar to JAX, in the sense that it traces some code, compiles array operations using the XLA compiler and can run the final compiled function in CPU, GPU or TPU.

Tenet.jl has top-class integration with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). Let's dive in on how to combine both Tenet.jl and Reactant.jl with a simple example: compute an expectation value between a MPS and a MPO.

Let's first initialize the state $\psi$ and the operator $H$ randomly:

```julia
julia> using Tenet

julia> H = rand(MPO; n=20, maxdim=32, eltype=ComplexF64)
MPO (inputs=20, outputs=10)

julia> ψ = rand(MPS; n=20, maxdim=8, eltype=ComplexF64)
MPS (inputs=0, outputs=20)

julia> expect(ψ, H)
0-dimensional Tensor{ComplexF64, 0, ConcreteRArray{ComplexF64, 0}}:
-2.4941116992789764e-6 - 2.1026840059684137e-7im
```

In order to be able to transparently move between CPU and GPU executions, Reactant.jl needs to manage the memory itself through its own `Array` implementation: `ConcreteRArray`.
Moreover, these `ConcreteRArray`s are used to know which array operations need to be traced.

Thanks to its integration with Adapt.jl, it's super easy to convert deep, nested data-structures containing arrays (like [`TensorNetwork`](@ref)):

```julia
julia> using Reactant, Adapt

julia> ψ_re = adapt(ConcreteRArray, ψ)
MPS (inputs=0, outputs=20)

julia> H_re = adapt(ConcreteRArray, H)
MPO (inputs=20, outputs=20)
```

In principle, all you need to do for Reactant.jl to do it's magic is just call `Reactant.compile` or `@compile`:

```julia
julia> expect_re = @compile expect(ψ_re, H_re)
2025-01-21 12:03:45.965896: I external/xla/xla/service/llvm_ir/llvm_command_line_options.cc:50] XLA (re)initializing LLVM with options fingerprint: 7776707050747573779
Reactant.Compiler.Thunk{typeof(expect), Symbol("##expect_reactant#631"), Tuple{MPS, MPO}, false}(expect)

julia> expect_re(ψ_re, H_re)
0-dimensional Tensor{ComplexF64, 0, ConcreteRArray{ComplexF64, 0}}:
-2.4941116992789764e-6 - 2.1026840059684137e-7im
```

Even if the first compilation can take a while (which we are addressing and will go faster in future versions), the speedup obtained by the compiled function compesates it if it's called many times:

```julia
julia> @b expect(ψ, H)
74.172 ms (906489 allocs: 49.189 MiB)

julia> @b expect_re(ψ_re, H_re)
11.154 ms (16 allocs: 688 bytes)
```

!!! note
    The number of allocations and memory usage of a Reactant.jl-compile function reported by `@time`, `BenchmarkTools.@btime` and `Chairmarks.@time` are not representative of the real execution. These functions only report heap allocations performed by Julia code, which Reactant.jl generated MLIR code is not.
    In any case, Reactant.jl can optimize the code to reduce allocations so the number of allocations **should** be less than the non-compiled code.

Furthermore, Reactant.jl supports state-of-art Automatic Differentiation through the Enzyme.jl AD engine.

```julia
julia> using Enzyme

julia> expect_grad = Reactant.compile((ψ_re, H_re)) do ψ_traced, H_traced
        return Enzyme.gradient(ReverseWithPrimal, expect, ψ_traced, Const(H_traced))
    end
...

julia> expect_grad(ψ_re, H_re)
...
```

!!! warning
    Reactant.jl does not support matrix factorizations like SVD, QR or Eigendecomposition yet. There is an issue tracking this in [Reactant.jl#336](https://github.com/EnzymeAD/Reactant.jl/issues/336).
