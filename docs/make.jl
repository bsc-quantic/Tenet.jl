using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using DocumenterCitations
using Tenet
using CairoMakie
using GraphMakie
using LinearAlgebra

DocMeta.setdocmeta!(Tenet, :DocTestSetup, :(using Tenet); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"); style=:authoryear)

makedocs(;
    modules=[Tenet, Base.get_extension(Tenet, :TenetGraphMakieExt)],
    sitename="Tenet.jl",
    authors="Sergio Sánchez Ramírez and contributors",
    pages=Any[
        "Home" => "index.md",
        "Tensors" => "tensors.md",
        "Tensor Networks" => "tensor-network.md",
        "Contraction" => "contraction.md",
        "Transformations" => "transformations.md",
        "Quantum" => [
            "Introduction" => "quantum.md",
            "Ansatzes" => ["`Product` ansatz" => "ansatz/product.md", "`Chain` ansatz" => "ansatz/chain.md"],
        ],
        "Visualization" => "visualization.md",
        "Alternatives" => "alternatives.md",
        "References" => "references.md",
        "⚒️ Developer Reference" => ["Inheritance and Traits" => "developer/inheritance.md"],
    ],
    pagesonly=true,
    format=Documenter.HTML(;
        prettyurls=false, assets=["assets/favicon.ico", "assets/citations.css", "assets/youtube.css"]
    ),
    plugins=[bib],
    checkdocs=:exports,
    warnonly=true,
)

deploydocs(; repo="github.com/bsc-quantic/Tenet.jl.git", devbranch="master", push_preview=true)
