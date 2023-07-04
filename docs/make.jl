using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using Tenet
using CairoMakie
using LinearAlgebra

DocMeta.setdocmeta!(Tenet, :DocTestSetup, :(using Tenet); recursive = true)

makedocs(
    modules = [
        Tenet,
        isdefined(Base, :get_extension) ? Base.get_extension(Tenet, :TenetMakieExt) : Tenet.TenetMakieExt,
    ],
    sitename = "Tenet.jl",
    authors = "Sergio Sánchez Ramírez and contributors",
    pages = Any[
        "Home"=>"index.md",
        "Tensor Networks"=>[
            "Introduction" => "tensor-network.md",
            "Contraction" => "contraction.md",
            "Visualization" => "visualization.md",
            "Transformations" => "transformations.md",
        ],
        "Quantum Tensor Networks"=>[
            "Introduction" => "quantum/index.md",
            "Matrix Product States (MPS)" => "quantum/mps.md",
        ],
        "Examples"=>[
            "Google's Quantum Advantage experiment" => "examples/google-rqc.md",
            "Matrix Product State classifier" => "examples/mps-ml.md",
        ],
        "Alternatives"=>"alternatives.md",
    ],
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
)

deploydocs(repo = "github.com/bsc-quantic/Tenet.jl.git", devbranch = "master", push_preview = true)
