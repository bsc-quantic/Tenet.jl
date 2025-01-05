# Quantum Tensor Networks

```@setup examples
using Tenet
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))
using GraphMakie
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

Quantum mechanics is a generalization of probability theory to complex probabilities[^1] with a very nice property: all of its objects are linear entities[^2]. A quantum state can be viewed as a vector-like object that represents a complex probability distribution, and a quantum operator can be viewed as a matrix-like object that represents a transformation of the probability distribution (so it preserves that the sum of probabilities is $1$).

[^1]: Believe me, I'm not lying. If you want to read more about this perspective, I recommend you the book "Quantum Computing since Democritus" by Scott Aaronson.

[^2]: Excepting measurements, but those are still an open problem.

⚠️ WIP

...
Tensor Network states and operators can efficiently represent some vectors living in an exponentially large vector space!
...

## The `Site` type

A [`Site`](@ref) is a helper type for representing sites.

```@repl examples
Site(1)
site"1"
```

[`Site`](@ref)s can refer to multi-dimensional cartesian indices too:

```@repl examples
Site(4,3)
site"4,3"
```

By default, a [`Site`](@ref) refers to a normal physical, output or _covariant_ index. If you want to create a dual physical, input or _contravariant_ index, you can pass the `dual=true` kwarg to the constructor or just call [`adjoint`](@ref Base.adjoint(::Site)) on it.

```@repl examples
Site(5; dual=true)
Site(5)'
adjoint(Site(5))
site"5'"
```

!!! warning
    We recommend the use of the [`@site_str`](@ref) macro (i.e. the `site"..."` expression) instead of the [`Site`](@ref) constructor, but the only caveat is it cannot interpolate variables yet. For now, use it only in cases where you can hardcode the site like in the REPL or scripts. Check out the tracking issue fo[#274](https://github.com/bsc-quantic/Tenet.jl/issues/274) for more information.

## The `Quantum` type

A [`Quantum`](@ref) is just a [`TensorNetwork`](@ref) with a mapping between indices and [`Site`](@ref)s.
It implements the [`AbstractQuantum`](@ref Tenet.AbstractQuantum) interface so any subtype of [`AbstractQuantum`](@ref Tenet.AbstractQuantum) can access to its features.

```@repl examples
tn = TensorNetwork([Tensor(zeros(2,2,2), [:p1,:i,:k]), Tensor(zeros(2,2,2), [:p2,:i,:j]), Tensor(zeros(2,2,2), [:p3,:j,:k])])
qtn = Quantum(tn, Dict([site"1" => :p1, site"2" => :p2, site"3" => :p3]))
```

```@example examples
graphplot(qtn, labels=true) # hide
```

Note that [`Quantum`](@ref) implements the [`AbstractTensorNetwork`](@ref Tenet.AbstractTensorNetwork) contract, so it inherits all the functionality of [`TensorNetwork`](@ref)!

```@repl examples
:i ∈ qtn
tensors(qtn; contains=:p1) .|> inds
```

[`Quantum`](@ref) provides a new function [`sites`](@ref), which returns the 

```@repl examples
sites(qtn)
```

Just like [`tensors`](@ref) and [`inds`](@ref), it has some kwarg-dispatched methods.

```@repl examples
sites(qtn; set=:inputs)
sites(qtn; set=:outputs)
sites(qtn; at=:p2)
```

Speaking of which, [`Quantum`](@ref) also extends [`tensors`](@ref) and [`inds`](@ref) with new kwarg-methods:

```@repl examples
tensors(qtn; at=site"1")
inds(qtn; at=site"1")
inds(qtn; set=:physical)
inds(qtn; set=:virtual)
inds(qtn; set=:inputs)
inds(qtn; set=:outputs)
```

!!! note
    In Tenet, an open index ≠ a physical index. Physical indices are the ones used to connect and interact with other states or operators, and are only the ones marked with [`Site`](@ref)s in [`Quantum`](@ref).
    Open indices not marked as physical are still virtual indices and can be later used to coomplement with other [`Quantum`](@ref); e.g. like a quantum state purification.

## Connectivity

The whole point of [`Quantum`](@ref) is to be able to connect [`TensorNetwork`](@ref)s without requiring the user to manually match the indices.
[`Quantum`](@ref) automatically takes care of matching input-output [`Site`](@ref)s, renaming the indices accordingly and finally connecting all tensors into one [`Quantum`](ref) Tensor Network.

For example, the following expectation value...

```math
\braket{\psi \mid O \mid \psi}
```

...can be expressed in Tenet as:

```@example examples
ψ = rand(MPS; n=3) # hide
O = Product(fill(zeros(2,2),3)) # hide
exp_val = merge(ψ, O, ψ')
graphplot(exp_val, layout=Stress()) # hide
```

!!! warning
    Unlike linear algebra notation, in which time evolution goes from right-to-left, [`merge`](@ref) and [`merge!`](@ref) go from left-to-right to keep consistency with Julia semantics.
    This is not such a big deal as you can imagine: quantum circuits are drawn from left-to-right and there are Tensor Networks pictures in the literature where time goes down-to-up and up-to-down!

Note that [`adjoint`](@ref Base.adjoint(::AbstractQuantum)) and [`adjoint!`](@ref) just conjugate the tensors and switch the [`Site`](@ref)s from normal to dual (output to input) and viceversa.

### The `Socket` trait

Depending on the number of inputs and outputs, a [`Quantum`] can be a state, a dual state or an operator. You can use the [`socket`](@ref) function to find out of which type it is your [`Quantum`] object:

```@repl examples
socket(ψ)
socket(ψ')
socket(O)
socket(exp_val)
```

[`socket`](@ref) is a dynamic trait: it returns an object whose type represents a property of the [`Quantum`](@ref) object, which in this case represents whether the [`Quantum`](@ref) object is a [`State`](@ref) (has information about being a normal or dual state), an [`Operator`](@ref) or a [`Scalar`](@ref) (a Tensor Network with no physical indices).

The motivation to use traits is that depending on the trait value, methods can dispatch to more specialized methods; i.e. you have a method optimized for [`State`](@ref) and another method optimized for [`Operator`](@ref) for instance. You can read more about it in [Interfaces and Traits](@ref).
