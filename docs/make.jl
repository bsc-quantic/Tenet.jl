using Documenter
using DocumenterVitepress
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
    pages=[
        "Home" => "index.md",
        "📖 Manual" => [
            "Tensors" => "manual/tensors.md",
            "Tensor Networks" => "manual/tensor-network.md",
            "Contraction" => "manual/contraction.md",
            "Transformations" => "manual/transformations.md",
            "Quantum" => "manual/quantum.md",
            "Ansatz" => [
                "Introduction" => "manual/ansatz/index.md",
                "Product ansatz" => "manual/ansatz/product.md",
                "MPS/MPO ansatz" => "manual/ansatz/mps.md",
            ],
        ],
        "Visualization" => "visualization.md",
        "Alternatives" => "alternatives.md",
        # "References" => "references.md",
        "API Reference" => "api.md",
        "⚒️ Developer Reference" => ["`TensorNetwork` type hierarchy" => "developer/type-hierarchy.md"],
    ],
    pagesonly=true,
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="https://github.com/bsc-quantic/Tenet.jl",
        assets=["assets/favicon.ico", "assets/citations.css", "assets/youtube.css"],
    ),
    plugins=[bib],
    checkdocs=:exports,
    warnonly=true,
)

deploydocs(; repo="github.com/bsc-quantic/Tenet.jl.git", target="build", devbranch="master", push_preview=true)
