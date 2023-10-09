# Automatic Differentiation on Tensor Network contraction

```@setup autodiff
using CairoMakie
```

Tensor Networks have recently gained popularity for Machine Learning tasks.
In this example, we show how to perform Automatic Differentiation on Tensor Network contraction to overlap the overlap between two [Matrix Product States (MPS)](@ref) with a smaller dimension.

```@example autodiff
using Tenet
using Zygote
using Random: seed! # hide

rng = seed!(4) # hide

ψ = rand(MPS{Open}, n = 4, p = 2, χ = 2)
ϕ = rand(MPS{Open}, n = 4, p = 2, χ = 4)
ψ = rand(rng, MPS{Open}, n = 4, p = 2, χ = 2) # hide
ϕ = rand(rng, MPS{Open}, n = 4, p = 2, χ = 4) # hide

tn = merge(ψ, ϕ')

plot(tn) # hide
```

This problem is known as _MPS compression_.
While there are better methods for this matter, this example excels for its simplicity and it can easily be modified for ML tasks.
The loss function minimizes when the overlap between the two states ``\psi`` and ``\phi`` maximizes, constrained to normalized states.

```math
\begin{aligned}
\min_\psi \quad & \left(\braket{\phi | \psi} - 1\right)^2 \\
\textrm{s.t.} \quad & \lVert \psi \rVert^2 = \braket{\psi \mid \psi} = 1 \\
 & \lVert \phi \rVert^2 = \braket{\phi \mid \phi} = 1
\end{aligned}
```

!!! warning "Implicit parameters"
    Currently, calling `Zygote.gradient`/`Zygote.jacobian` on functions with explicit parameters doesn't interact well with `Tenet` data-structures (i.e. `Tensor` and `TensorNetwork`) on the interface.

    While the problem persists, use implicit parameters with `Zygote.Params` on the arrays (i.e. call `Params([parent(tensor)])` or `Params([arrays(tensor_network)])`).

```@example autodiff
η = 0.01
@time losses = map(1:200) do it
    # compute gradient
    loss, ∇ = withgradient(Params(arrays(ψ))) do
        ((contract(tn) |> first) - 1)^2
    end

    # direct gradient descent
    for array in arrays(ψ)
        array .-= η * ∇[array]
    end

    # normalize state
    normalize!(ψ)

    return loss
end

f = Figure() # hide
ax = Axis(f[1, 1], yscale = log10, xscale = identity, xlabel="Iterations") # hide
lines!(losses, label="Loss") # hide
lines!(map(x -> 1 - sqrt(x), losses), label="Overlap") # hide
f[1,2] = Legend(f, ax, framevisible=false) # hide
f # hide
```
