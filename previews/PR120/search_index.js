var documenterSearchIndex = {"docs":
[{"location":"references.html#References","page":"References","title":"References","text":"","category":"section"},{"location":"references.html","page":"References","title":"References","text":"Fishman, M.; White, S. R. and Stoudenmire, E. M. (2022). The ITensor Software Library for Tensor Network Calculations. SciPost Phys. Codebases, 4.\n\n\n\nGray, J. (2018), quimb: A python package for quantum information and many-body calculations. Journal of Open Source Software 3, 819.\n\n\n\nGray, J. and Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum 5, 410.\n\n\n\nHauschild, J.; Pollmann, F. and Zaletel, M. (2021). The Tensor Network Python (TeNPy) Library. In: APS March Meeting Abstracts, Vol. 2021; p. R21–006.\n\n\n\nRamón Pareja Monturiol, J.; Pérez-García, D. and Pozas-Kerstjens, A. (2023). TensorKrowch: Smooth integration of tensor networks in machine learning, arXiv e-prints, arXiv–2306.\n\n\n\n","category":"page"},{"location":"tensors.html#Tensors","page":"Tensors","title":"Tensors","text":"","category":"section"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"using Tenet","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"There are many jokes[1] about how to define a tensor. The definition we are giving here might not be the most correct one, but it is good enough for our use case (don't kill me please, mathematicians). A tensor T of order[2] n is a multilinear[3] application between n vector spaces over a field mathcalF.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"[1]: For example, recursive definitions like a tensor is whatever that transforms as a tensor.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"[2]: The order of a tensor may also be known as rank or dimensionality in other fields. However, these can be missleading, since it has nothing to do with the rank of linear algebra nor with the dimensionality of a vector space. We prefer to use word order.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"[3]: Meaning that the relationships between the output and the inputs, and the inputs between them, are linear.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"T  mathcalF^dim(1) times dots times mathcalF^dim(n) mapsto mathcalF","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"In layman's terms, it is a linear function whose inputs are vectors and the output is a scalar number.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"T(mathbfv^(1) dots mathbfv^(n)) = c in mathcalF qquadqquad forall i mathbfv^(i) in mathcalF^dim(i)","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"Tensor algebra is a higher-order generalization of linear algebra, where scalar numbers can be viewed as order-0 tensors, vectors as order-1 tensors, matrices as order-2 tensors, ...","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"<img src=\"assets/tensor.excalidraw.svg\" class=\"invert-on-dark\"/>","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"Letters are used to identify each of the vector spaces the tensor relates to. In computer science, you would intuitively think of tensors as \"n-dimensional arrays with named dimensions\".","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"T_ijk iff mathttTijk","category":"page"},{"location":"tensors.html#The-Tensor-type","page":"Tensors","title":"The Tensor type","text":"","category":"section"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"In Tenet, a tensor is represented by the Tensor type, which wraps an array and a list of symbols. As it subtypes AbstractArray, many array operations can be dispatched to it.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"You can create a Tensor by passing an array and a list of Symbols that name indices.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"Tᵢⱼₖ = Tensor(rand(3,5,2), (:i,:j,:k))","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"The dimensionality or size of each index can be consulted using the size function.","category":"page"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"Base.size(::Tensor)","category":"page"},{"location":"tensors.html#Base.size-Tuple{Tensor}","page":"Tensors","title":"Base.size","text":"Base.size(::Tensor[, i])\n\nReturn the size of the underlying array or the dimension i (specified by Symbol or Integer).\n\n\n\n\n\n","category":"method"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"size(Tᵢⱼₖ)\nsize(Tᵢⱼₖ, :j)\nlength(Tᵢⱼₖ)","category":"page"},{"location":"tensors.html#Operations","page":"Tensors","title":"Operations","text":"","category":"section"},{"location":"tensors.html#Contraction","page":"Tensors","title":"Contraction","text":"","category":"section"},{"location":"tensors.html","page":"Tensors","title":"Tensors","text":"Tenet.contract(::Tensor, ::Tensor)","category":"page"},{"location":"tensors.html#Tenet.contract-Tuple{Tensor, Tensor}","page":"Tensors","title":"Tenet.contract","text":"contract(a::Tensor[, b::Tensor]; dims=nonunique([inds(a)..., inds(b)...]))\n\nPerform tensor contraction operation.\n\n\n\n\n\n","category":"method"},{"location":"contraction.html#Contraction","page":"Contraction","title":"Contraction","text":"","category":"section"},{"location":"contraction.html","page":"Contraction","title":"Contraction","text":"Contraction path optimization and execution is delegated to the EinExprs library. A EinExpr is a lower-level form of a Tensor Network, in which the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. Expr) but in which every node represents an Einstein summation expression (aka einsum).","category":"page"},{"location":"contraction.html","page":"Contraction","title":"Contraction","text":"einexpr(::Tenet.AbstractTensorNetwork)\ncontract(::Tenet.AbstractTensorNetwork)\ncontract!","category":"page"},{"location":"contraction.html#EinExprs.einexpr-Tuple{Tenet.AbstractTensorNetwork}","page":"Contraction","title":"EinExprs.einexpr","text":"einexpr(tn::AbstractTensorNetwork; optimizer = EinExprs.Greedy, output = inds(tn, :open), kwargs...)\n\nSearch a contraction path for the given TensorNetwork and return it as a EinExpr.\n\nKeyword Arguments\n\noptimizer Contraction path optimizer. Check EinExprs documentation for more info.\noutputs Indices that won't be contracted. Defaults to open indices.\nkwargs Options to be passed to the optimizer.\n\nSee also: contract.\n\n\n\n\n\n","category":"method"},{"location":"contraction.html#Tenet.contract-Tuple{Tenet.AbstractTensorNetwork}","page":"Contraction","title":"Tenet.contract","text":"contract(tn::AbstractTensorNetwork; kwargs...)\n\nContract a TensorNetwork. The contraction order will be first computed by einexpr.\n\nThe kwargs will be passed down to the einexpr function.\n\nSee also: einexpr, contract!.\n\n\n\n\n\n","category":"method"},{"location":"contraction.html#Tenet.contract!","page":"Contraction","title":"Tenet.contract!","text":"contract!(tn::AbstractTensorNetwork, index)\n\nIn-place contraction of tensors connected to index.\n\nSee also: contract.\n\n\n\n\n\n","category":"function"},{"location":"alternatives.html#Alternatives","page":"Alternatives","title":"Alternatives","text":"","category":"section"},{"location":"alternatives.html","page":"Alternatives","title":"Alternatives","text":"Tenet is strongly opinionated. We acknowledge that it may not suit all cases (although we try 🙂). If your case doesn't fit Tenet's design, you can try the following libraries:","category":"page"},{"location":"alternatives.html","page":"Alternatives","title":"Alternatives","text":"quimb (Gray, 2018) Flexible Tensor Network written in Python. Main source of inspiration for Tenet.\ntenpy (Hauschild et al., 2021) Tensor Network library written in Python with a strong focus on physics.\nITensors.jl (Fishman et al., 2022) Mature Tensor Network framework written in Julia.\ntensorkrowch (Ramón Pareja Monturiol et al., 2023) A new Tensor Network library built on top of PyTorch.","category":"page"},{"location":"visualization.html#Visualization","page":"Visualization","title":"Visualization","text":"","category":"section"},{"location":"visualization.html","page":"Visualization","title":"Visualization","text":"using Makie\nMakie.inline!(true)\nset_theme!(resolution=(800,400))\n\nusing CairoMakie\nCairoMakie.activate!(type = \"svg\")\n\nusing Tenet","category":"page"},{"location":"visualization.html","page":"Visualization","title":"Visualization","text":"Tenet provides a Package Extension for Makie support. You can just import a Makie backend and call Makie.plot on a TensorNetwork.","category":"page"},{"location":"visualization.html","page":"Visualization","title":"Visualization","text":"Makie.plot(::Tenet.TensorNetwork)","category":"page"},{"location":"visualization.html#MakieCore.plot-Tuple{TensorNetwork}","page":"Visualization","title":"MakieCore.plot","text":"plot(tn::TensorNetwork; kwargs...)\nplot!(f::Union{Figure,GridPosition}, tn::TensorNetwork; kwargs...)\nplot!(ax::Union{Axis,Axis3}, tn::TensorNetwork; kwargs...)\n\nPlot a TensorNetwork as a graph.\n\nKeyword Arguments\n\nlabels If true, show the labels of the tensor indices. Defaults to false.\nThe rest of kwargs are passed to GraphMakie.graphplot.\n\n\n\n\n\n","category":"method"},{"location":"visualization.html","page":"Visualization","title":"Visualization","text":"tn = rand(TensorNetwork, 14, 4, seed=0) # hide\nplot(tn, labels=true)","category":"page"},{"location":"index.html#Tenet.jl","page":"Home","title":"Tenet.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"info: BSC-Quantic's Registry\nTenet and some of its dependencies are located in our own Julia registry. In order to download Tenet, add our registry to your Julia installation by using the Pkg mode in a REPL session,using Pkg\npkg\"registry add https://github.com/bsc-quantic/Registry\"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"A Julia library for Tensor Networks. Tenet can be executed both at local environments and on large supercomputers. Its goals are,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Expressiveness Simple to use 👶\nFlexibility Extend it to your needs 🔧\nPerformance Goes brr... fast 🏎️","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"A video of its presentation at JuliaCon 2023 can be seen here:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"<div class=\"youtube-video\">\n<iframe class=\"youtube-video\" width=\"560\" src=\"https://www.youtube-nocookie.com/embed/8BHGtm6FRMk?si=bPXB6bPtK695HFIR\" title=\"YouTube video player\" frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share\" allowfullscreen></iframe>\n</div>","category":"page"},{"location":"index.html#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Optimized Tensor Network contraction, powered by EinExprs\nTensor Network slicing/cuttings\nAutomatic Differentiation of TN contraction, powered by EinExprs and ChainRules\n3D visualization of large networks, powered by Makie","category":"page"},{"location":"tensor-network.html#Tensor-Networks","page":"Tensor Networks","title":"Tensor Networks","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"Tensor Networks (TN) are a graphical notation for representing complex multi-linear functions. For example, the following equation","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"sum_ijklmnop A_im B_ijp C_njk D_pkl E_mno F_ol","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"can be represented visually as","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"<figure>\n<img width=500 src=\"assets/tn-sketch.svg\" alt=\"Sketch of a Tensor Network\">\n<figcaption>Sketch of a Tensor Network</figcaption>\n</figure>","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"The graph's nodes represent tensors and edges represent tensor indices.","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"In Tenet, these objects are represented by the TensorNetwork type.","category":"page"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"TensorNetwork","category":"page"},{"location":"tensor-network.html#Tenet.TensorNetwork","page":"Tensor Networks","title":"Tenet.TensorNetwork","text":"TensorNetwork\n\nGraph of interconnected tensors, representing a multilinear equation. Graph vertices represent tensors and graph edges, tensor indices.\n\n\n\n\n\n","category":"type"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"Information about a TensorNetwork can be queried with the following functions.","category":"page"},{"location":"tensor-network.html#Query-information","page":"Tensor Networks","title":"Query information","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"inds(::Tenet.AbstractTensorNetwork)\nsize(::Tenet.AbstractTensorNetwork)\ntensors(::Tenet.AbstractTensorNetwork)","category":"page"},{"location":"tensor-network.html#EinExprs.inds-Tuple{Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"EinExprs.inds","text":"inds(tn::AbstractTensorNetwork, set = :all)\n\nReturn the names of the indices in the TensorNetwork.\n\nKeyword Arguments\n\nset\n:all (default) All indices.\n:open Indices only mentioned in one tensor.\n:inner Indices mentioned at least twice.\n:hyper Indices mentioned at least in three tensors.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.size-Tuple{Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"Base.size","text":"size(tn::AbstractTensorNetwork)\nsize(tn::AbstractTensorNetwork, index)\n\nReturn a mapping from indices to their dimensionalities.\n\nIf index is set, return the dimensionality of index. This is equivalent to size(tn)[index].\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Tenet.tensors-Tuple{Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"Tenet.tensors","text":"tensors(tn::AbstractTensorNetwork)\n\nReturn a list of the Tensors in the TensorNetwork.\n\nImplementation details\n\nAs the tensors of a TensorNetwork are stored as keys of the .tensormap dictionary and it uses objectid as hash, order is not stable so it sorts for repeated evaluations.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Modification","page":"Tensor Networks","title":"Modification","text":"","category":"section"},{"location":"tensor-network.html#Add/Remove-tensors","page":"Tensor Networks","title":"Add/Remove tensors","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"push!(::Tenet.AbstractTensorNetwork, ::Tensor)\nappend!(::Tenet.AbstractTensorNetwork, ::Base.AbstractVecOrTuple{<:Tensor})\nmerge!(::Tenet.AbstractTensorNetwork, ::Tenet.AbstractTensorNetwork)\npop!(::Tenet.AbstractTensorNetwork, ::Tensor)\ndelete!(::Tenet.AbstractTensorNetwork, ::Any)","category":"page"},{"location":"tensor-network.html#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}","page":"Tensor Networks","title":"Base.push!","text":"push!(tn::AbstractTensorNetwork, tensor::Tensor)\n\nAdd a new tensor to the Tensor Network.\n\nSee also: append!, pop!.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.append!-Tuple{Tenet.AbstractTensorNetwork, Union{Tuple{Vararg{var\"#s6\"}}, AbstractVector{<:var\"#s6\"}} where var\"#s6\"<:Tensor}","page":"Tensor Networks","title":"Base.append!","text":"append!(tn::AbstractTensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})\n\nAdd a list of tensors to a TensorNetwork.\n\nSee also: push!, merge!.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.merge!-Tuple{Tenet.AbstractTensorNetwork, Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"Base.merge!","text":"merge!(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)\nmerge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)\n\nFuse various TensorNetworks into one.\n\nSee also: append!.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.pop!-Tuple{Tenet.AbstractTensorNetwork, Tensor}","page":"Tensor Networks","title":"Base.pop!","text":"pop!(tn::AbstractTensorNetwork, tensor::Tensor)\npop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})\n\nRemove a tensor from the Tensor Network and returns it. If a Tensor is passed, then the first tensor satisfies egality (i.e. ≡ or ===) will be removed. If a Symbol or a list of Symbols is passed, then remove and return the tensors that contain all the indices.\n\nSee also: push!, delete!.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.delete!-Tuple{Tenet.AbstractTensorNetwork, Any}","page":"Tensor Networks","title":"Base.delete!","text":"delete!(tn::AbstractTensorNetwork, x)\n\nLike pop! but return the TensorNetwork instead.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Replace-existing-elements","page":"Tensor Networks","title":"Replace existing elements","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"replace!","category":"page"},{"location":"tensor-network.html#Base.replace!","page":"Tensor Networks","title":"Base.replace!","text":"replace!(tn::AbstractTensorNetwork, old => new...)\nreplace(tn::AbstractTensorNetwork, old => new...)\n\nReplace the element in old with the one in new. Depending on the types of old and new, the following behaviour is expected:\n\nIf Symbols, it will correspond to a index renaming.\nIf Tensors, first element that satisfies egality (≡ or ===) will be replaced.\n\n\n\n\n\n","category":"function"},{"location":"tensor-network.html#Selection","page":"Tensor Networks","title":"Selection","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"select\nselectdim\nslice!\nview(::Tenet.AbstractTensorNetwork)","category":"page"},{"location":"tensor-network.html#Tenet.select","page":"Tensor Networks","title":"Tenet.select","text":"select(tn::AbstractTensorNetwork, i)\n\nReturn tensors whose indices match with the list of indices i.\n\n\n\n\n\n","category":"function"},{"location":"tensor-network.html#Base.selectdim","page":"Tensor Networks","title":"Base.selectdim","text":"selectdim(tn::AbstractTensorNetwork, index::Symbol, i)\n\nReturn a copy of the TensorNetwork where index has been projected to dimension i.\n\nSee also: view, slice!.\n\n\n\n\n\n","category":"function"},{"location":"tensor-network.html#Tenet.slice!","page":"Tensor Networks","title":"Tenet.slice!","text":"slice!(tn::AbstractTensorNetwork, index::Symbol, i)\n\nIn-place projection of index on dimension i.\n\nSee also: selectdim, view.\n\n\n\n\n\n","category":"function"},{"location":"tensor-network.html#Base.view-Tuple{Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"Base.view","text":"view(tn::AbstractTensorNetwork, index => i...)\n\nReturn a copy of the TensorNetwork where each index has been projected to dimension i. It is equivalent to a recursive call of selectdim.\n\nSee also: selectdim, slice!.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Miscelaneous","page":"Tensor Networks","title":"Miscelaneous","text":"","category":"section"},{"location":"tensor-network.html","page":"Tensor Networks","title":"Tensor Networks","text":"Base.copy(::Tenet.AbstractTensorNetwork)\nBase.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)","category":"page"},{"location":"tensor-network.html#Base.copy-Tuple{Tenet.AbstractTensorNetwork}","page":"Tensor Networks","title":"Base.copy","text":"copy(tn::TensorNetwork)\n\nReturn a shallow copy of a TensorNetwork.\n\n\n\n\n\n","category":"method"},{"location":"tensor-network.html#Base.rand-Tuple{Type{TensorNetwork}, Integer, Integer}","page":"Tensor Networks","title":"Base.rand","text":"rand(TensorNetwork, n::Integer, regularity::Integer; out = 0, dim = 2:9, seed = nothing, globalind = false)\n\nGenerate a random tensor network.\n\nArguments\n\nn Number of tensors.\nregularity Average number of indices per tensor.\nout Number of open indices.\ndim Range of dimension sizes.\nseed If not nothing, seed random generator with this value.\nglobalind Add a global 'broadcast' dimension to every tensor.\n\n\n\n\n\n","category":"method"},{"location":"transformations.html#Transformations","page":"Transformations","title":"Transformations","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"using Makie\nMakie.inline!(true)\n\nusing CairoMakie\nusing Tenet\nusing NetworkLayout\n\nfunction smooth_annotation!(f; color=Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 60 // 256), xlims=[-2, 2], ylims=[-2, 2], offset_x=0.0, offset_y=0.0, radius_x=1.0, radius_y=1.0, num_waves=5, fluctuation_amplitude=0.1, phase_shift=0.0)\n    ax = Axis(f)\n    hidedecorations!(ax)\n    hidespines!(ax)\n\n    # Define limits of the plot\n    xlims!(ax, xlims...)\n    ylims!(ax, ylims...)\n\n    # Create a perturbed filled shape\n    theta = LinRange(0, 2π, 100)\n\n    fluctuations = fluctuation_amplitude .* sin.(num_waves .* theta .+ phase_shift)\n\n    # Apply the fluctuations and radius scaling\n    perturbed_radius_x = radius_x .+ fluctuations\n    perturbed_radius_y = radius_y .+ fluctuations\n\n    circle_points = [Point2f((perturbed_radius_x[i]) * cos(theta[i]) + offset_x,\n                              (perturbed_radius_y[i]) * sin(theta[i]) + offset_y) for i in 1:length(theta)]\n\n    poly!(ax, circle_points, color=color, closed=true)\nend\n\nbg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)\norange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)\nred = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)","category":"page"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network's structure locally by permuting, contracting, factoring or truncating tensors.","category":"page"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.","category":"page"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Our approach is based in (Gray and Kourtis, 2021), which can also be found in quimb.","category":"page"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"In Tenet, we provide a set of predefined transformations which you can apply to your TensorNetwork using both the transform/transform! functions.","category":"page"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"transform\ntransform!","category":"page"},{"location":"transformations.html#Tenet.transform","page":"Transformations","title":"Tenet.transform","text":"transform(tn::AbstractTensorNetwork, config::Transformation)\ntransform(tn::AbstractTensorNetwork, configs)\n\nReturn a new TensorNetwork where some Transformation has been performed into it.\n\nSee also: transform!.\n\n\n\n\n\n","category":"function"},{"location":"transformations.html#Tenet.transform!","page":"Transformations","title":"Tenet.transform!","text":"transform!(tn::AbstractTensorNetwork, config::Transformation)\ntransform!(tn::AbstractTensorNetwork, configs)\n\nIn-place version of transform.\n\n\n\n\n\n","category":"function"},{"location":"transformations.html#Transformations-2","page":"Transformations","title":"Transformations","text":"","category":"section"},{"location":"transformations.html#Hyperindex-converter","page":"Transformations","title":"Hyperindex converter","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.HyperindConverter","category":"page"},{"location":"transformations.html#Tenet.HyperindConverter","page":"Transformations","title":"Tenet.HyperindConverter","text":"HyperindConverter <: Transformation\n\nConvert hyperindices to COPY-tensors, represented by DeltaArrays. This transformation is always used by default when visualizing a TensorNetwork with plot.\n\n\n\n\n\n","category":"type"},{"location":"transformations.html#Diagonal-reduction","page":"Transformations","title":"Diagonal reduction","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.DiagonalReduction","category":"page"},{"location":"transformations.html#Tenet.DiagonalReduction","page":"Transformations","title":"Tenet.DiagonalReduction","text":"DiagonalReduction <: Transformation\n\nReduce the dimension of a Tensor in a TensorNetwork when it has a pair of indices that fulfil a diagonal structure.\n\nKeyword Arguments\n\natol Absolute tolerance. Defaults to 1e-12.\n\n\n\n\n\n","category":"type"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"set_theme!(resolution=(800,200)) # hide\nfig = Figure() #hide\n\ndata = zeros(Float64, 2, 2, 2, 2) #hide\nfor i in 1:2 #hide\n    for j in 1:2 #hide\n        for k in 1:2 #hide\n            data[i, i, j, k] = k #hide\n        end #hide\n    end #hide\nend #hide\n\nA = Tensor(data, (:i, :j, :k, :l)) #hide\nB = Tensor(rand(2, 2), (:i, :m)) #hide\nC = Tensor(rand(2, 2), (:j, :n)) #hide\n\ntn = TensorNetwork([A, B, C]) #hide\nreduced = transform(tn, Tenet.DiagonalReduction) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 1]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = -0.21, #hide\n    offset_y = -0.42, #hide\n    radius_x = 0.38, #hide\n    radius_y = 0.8, #hide\n    num_waves = 6, #hide\n    fluctuation_amplitude = 0.02, #hide\n    phase_shift = 0.0) #hide\nplot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=100); node_color=[red, orange, orange, :black, :black,:black, :black]) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 2]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = 0.1, #hide\n    offset_y = -0.35, #hide\n    radius_x = 0.38, #hide\n    radius_y = 1.1, #hide\n    num_waves = 5, #hide\n    fluctuation_amplitude = 0.02, #hide\n    phase_shift = 1.9) #hide\nplot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=100),  node_color=[orange, orange, red, :black, :black, :black, :black, :black]) #hide\n\nLabel(fig[1, 1, Bottom()], \"Original\") #hide\nLabel(fig[1, 2, Bottom()], \"Transformed\") #hide\n\nfig #hide","category":"page"},{"location":"transformations.html#Anti-diagonal-reduction","page":"Transformations","title":"Anti-diagonal reduction","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.AntiDiagonalGauging","category":"page"},{"location":"transformations.html#Tenet.AntiDiagonalGauging","page":"Transformations","title":"Tenet.AntiDiagonalGauging","text":"AntiDiagonalGauging <: Transformation\n\nReverse the order of tensor indices that fulfill the anti-diagonal condition. While this transformation doesn't directly enhance computational efficiency, it sets up the TensorNetwork for other operations that do.\n\nKeyword Arguments\n\natol Absolute tolerance. Defaults to 1e-12.\nskip List of indices to skip. Defaults to [].\n\n\n\n\n\n","category":"type"},{"location":"transformations.html#Rank-simplification","page":"Transformations","title":"Rank simplification","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.RankSimplification","category":"page"},{"location":"transformations.html#Tenet.RankSimplification","page":"Transformations","title":"Tenet.RankSimplification","text":"RankSimplification <: Transformation\n\nPreemptively contract tensors whose result doesn't increase in size.\n\n\n\n\n\n","category":"type"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"set_theme!(resolution=(800,200)) # hide\nfig = Figure() #hide\n\nA = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l)) #hide\nB = Tensor(rand(2, 2), (:i, :m)) #hide\nC = Tensor(rand(2, 2, 2), (:m, :n, :o)) #hide\nE = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j)) #hide\n\ntn = TensorNetwork([A, B, C, E]) #hide\nreduced = transform(tn, Tenet.RankSimplification) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 1]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = -0.32, #hide\n    offset_y = -0.5, #hide\n    radius_x = 0.25, #hide\n    radius_y = 0.94, #hide\n    num_waves = 6, #hide\n    fluctuation_amplitude = 0.01, #hide\n    phase_shift = 0.0) #hide\nplot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=20); node_color=[orange, red, orange, orange, :black, :black, :black, :black, :black]) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 2]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = 0.12, #hide\n    offset_y = -0.62, #hide\n    radius_x = 0.18, #hide\n    radius_y = 0.46, #hide\n    num_waves = 5, #hide\n    fluctuation_amplitude = 0.01, #hide\n    phase_shift = 0) #hide\nplot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=1); node_color=[red, orange, orange, :black, :black, :black, :black, :black]) #hide\n\nLabel(fig[1, 1, Bottom()], \"Original\") #hide\nLabel(fig[1, 2, Bottom()], \"Transformed\") #hide\n\nfig #hide","category":"page"},{"location":"transformations.html#Column-reduction","page":"Transformations","title":"Column reduction","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.ColumnReduction","category":"page"},{"location":"transformations.html#Tenet.ColumnReduction","page":"Transformations","title":"Tenet.ColumnReduction","text":"ColumnReduction <: Transformation\n\nTruncate the dimension of a Tensor in a TensorNetwork when it contains columns with all elements smaller than atol.\n\nKeyword Arguments\n\natol Absolute tolerance. Defaults to 1e-12.\nskip List of indices to skip. Defaults to [].\n\n\n\n\n\n","category":"type"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"set_theme!(resolution=(800,200)) # hide\nfig = Figure() #hide\n\ndata = rand(3, 3, 3) #hide\ndata[:, 1:2, :] .= 0 #hide\n\nA = Tensor(data, (:i, :j, :k)) #hide\nB = Tensor(rand(3, 3), (:j, :l)) #hide\nC = Tensor(rand(3, 3), (:l, :m)) #hide\n\ntn = TensorNetwork([A, B, C]) #hide\nreduced = transform(tn, Tenet.ColumnReduction) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 1]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = -1.12, #hide\n    offset_y = -0.22, #hide\n    radius_x = 0.35, #hide\n    radius_y = 0.84, #hide\n    num_waves = 4, #hide\n    fluctuation_amplitude = 0.02, #hide\n    phase_shift = 0.0) #hide\nplot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=6); node_color=[red, orange, orange, :black, :black, :black]) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 2]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = -0.64, #hide\n    offset_y = 1.2, #hide\n    radius_x = 0.32, #hide\n    radius_y = 0.78, #hide\n    num_waves = 5, #hide\n    fluctuation_amplitude = 0.02, #hide\n    phase_shift = 0) #hide\n\nLabel(fig[1, 1, Bottom()], \"Original\") #hide\nLabel(fig[1, 2, Bottom()], \"Transformed\") #hide\nplot!(fig[1, 2], reduced, layout=Spring(iterations=2000, C=40, seed=8); node_color=[red, orange, orange, :black, :black, :black]) #hide\n\nfig #hide","category":"page"},{"location":"transformations.html#Split-simplification","page":"Transformations","title":"Split simplification","text":"","category":"section"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"Tenet.SplitSimplification","category":"page"},{"location":"transformations.html#Tenet.SplitSimplification","page":"Transformations","title":"Tenet.SplitSimplification","text":"SplitSimplification <: Transformation\n\nReduce the rank of tensors in the TensorNetwork by decomposing them using the Singular Value Decomposition (SVD). Tensors whose factorization do not increase the maximum rank of the network are left decomposed.\n\nKeyword Arguments\n\natol Absolute tolerance. Defaults to 1e-10.\n\n\n\n\n\n","category":"type"},{"location":"transformations.html","page":"Transformations","title":"Transformations","text":"set_theme!(resolution=(800,200)) # hide\nfig = Figure() #hide\n\nv1 = Tensor([1, 2, 3], (:i,)) #hide\nv2 = Tensor([4, 5, 6], (:j,)) #hide\nm1 = Tensor(rand(3, 3), (:k, :l)) #hide\n\nt1 = contract(v1, v2) #hide\ntensor = contract(t1, m1)  #hide\n\ntn = TensorNetwork([tensor, Tensor(rand(3, 3, 3), (:k, :m, :n)), Tensor(rand(3, 3, 3), (:l, :n, :o))]) #hide\nreduced = transform(tn, Tenet.SplitSimplification) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 1]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = 0.24, #hide\n    offset_y = 0.6, #hide\n    radius_x = 0.32, #hide\n    radius_y = 0.78, #hide\n    num_waves = 5, #hide\n    fluctuation_amplitude = 0.015, #hide\n    phase_shift = 0.0) #hide\nplot!(fig[1, 1], tn, layout=Spring(iterations=10000, C=0.5, seed=12); node_color=[red, orange,  orange, :black, :black, :black, :black]) #hide\n\nsmooth_annotation!( #hide\n    fig[1, 2]; #hide\n    color = bg_blue, #hide\n    xlims = [-2, 2], #hide\n    ylims = [-2, 2], #hide\n    offset_x = -0.2, #hide\n    offset_y = -0.4, #hide\n    radius_x = 1.1, #hide\n    radius_y = 0.75, #hide\n    num_waves = 3, #hide\n    fluctuation_amplitude = 0.18, #hide\n    phase_shift = 0.8) #hide\n\nLabel(fig[1, 1, Bottom()], \"Original\") #hide\nLabel(fig[1, 2, Bottom()], \"Transformed\") #hide\nplot!(fig[1, 2], reduced, layout=Spring(iterations=10000, C=13, seed=151); node_color=[orange, orange, red, red, red, :black, :black, :black, :black]) #hide\n\nfig #hide","category":"page"}]
}
