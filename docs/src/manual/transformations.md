# Transformations

```@setup plot
using Makie
using CairoMakie
using GraphMakie
using NetworkLayout
using Tenet

CairoMakie.activate!(type = "svg")
Makie.inline!(true)
set_theme!(resolution=(800,200))
```

In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network's structure locally by permuting, contracting, factoring or truncating tensors.

A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.

Our approach is based in [gray2021hyper](@cite), which can also be found in [quimb](https://quimb.readthedocs.io/).

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform`/`transform!` functions.

## Available transformations

### Hyperindex converter

### Contraction simplification

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2, 2), (:m, :n, :o)) #hide
E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j)) #hide

tn = TensorNetwork([A, B, C, E]) #hide
reduced = transform(tn, Tenet.ContractSimplification) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Diagonal reduction

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = zeros(Float64, 2, 2, 2, 2) #hide
for i in 1:2 #hide
    for j in 1:2 #hide
        for k in 1:2 #hide
            data[i, i, j, k] = k #hide
        end #hide
    end #hide
end #hide

A = Tensor(data, (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2), (:j, :n)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.DiagonalReduction) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Anti-diagonal reduction

### Dimension truncation

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = rand(3, 3, 3) #hide
data[:, 1:2, :] .= 0 #hide

A = Tensor(data, (:i, :j, :k)) #hide
B = Tensor(rand(3, 3), (:j, :l)) #hide
C = Tensor(rand(3, 3), (:l, :m)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.Truncate) #hide

graphplot!(fig[1, 1], tn; layout=Spring(C=10), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Spring(C=10), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Split simplification

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

v1 = Tensor([1, 2, 3], (:i,)) #hide
v2 = Tensor([4, 5, 6], (:j,)) #hide
m1 = Tensor(rand(3, 3), (:k, :l)) #hide

t1 = contract(v1, v2) #hide
tensor = contract(t1, m1)  #hide

tn = TensorNetwork([ #hide
    tensor, #hide
    Tensor(rand(3, 3, 3), (:k, :m, :n)), #hide
    Tensor(rand(3, 3, 3), (:l, :n, :o)) #hide
]) #hide
reduced = transform(tn, Tenet.SplitSimplification) #hide

graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide
graphplot!(fig[1, 2], reduced; layout=Spring(C=11), labels=true) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```
