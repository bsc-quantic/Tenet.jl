# Matrix Product State classifier

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Take this code as an example of what we want to achieve.

```julia
using Tenet
using Zygote

ψ = rand(MatrixProductState, 40, 2, 128)
```

```math
L(\psi) = \frac{1}{N} \sum^N_{i=1} \left( \braket{\phi(\mathbf{x}^{(i)}) \mid \psi} - 1 \right)^2
```

```julia
loss(ψ) = 1/N * sum((fidelity(ψ, ɸ(sample)) - 1)^2 for sample in dataset)

λ = 0.001
for it in 1:100
    ∇ = gradient(loss, ψ)
    ψ -= λ * ∇
end
```
