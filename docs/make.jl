using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using DocumenterCitations
using Tenet
using CairoMakie
using LinearAlgebra

DocMeta.setdocmeta!(Tenet, :DocTestSetup, :(using Tenet); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"); style = :authoryear)

makedocs(
    modules = [Tenet, Base.get_extension(Tenet, :TenetMakieExt)],
    sitename = "Tenet.jl",
    authors = "Sergio Sánchez Ramírez and contributors",
    pages = Any[
        "Home"=>"index.md",
        "Tensors"=>"tensors.md",
        "Tensor Networks"=>[
            "Introduction" => "tensor-network.md",
            "Contraction" => "contraction.md",
            "Transformations" => "transformations.md",
            "Visualization" => "visualization.md",
        ],
        "Quantum Tensor Networks"=>[
            "Introduction" => "quantum/index.md",
            "Matrix Product States (MPS)" => "quantum/mps.md",
            "Projected Entangled Pair States (PEPS)" => "quantum/peps.md",
        ],
        "Examples"=>[
            "Google's Quantum Advantage experiment" => "examples/google-rqc.md",
            "Automatic Differentiation on Tensor Network contraction" => "examples/ad-tn.md",
        ],
        "Alternatives"=>"alternatives.md",
        "References"=>"references.md",
    ],
    format = Documenter.HTML(
        prettyurls = false,
        assets = ["assets/favicon.ico", "assets/citations.css", "assets/youtube.css"],
    ),
    plugins = [bib],
)

deploydocs(repo = "github.com/bsc-quantic/Tenet.jl.git", devbranch = "master", push_preview = true)
