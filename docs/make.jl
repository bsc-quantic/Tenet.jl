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
            "Tensor Networks" => [
                "Introduction" => "manual/tensor-network.md",
                "Quantum" => "manual/quantum.md",
                "Ansatz" => [
                    "Introduction" => "manual/ansatz/index.md",
                    "Product ansatz" => "manual/ansatz/product.md",
                    "MPS/MPO ansatz" => "manual/ansatz/mps.md",
                ],
            ],
            "🤝 Interoperation" => "manual/interop.md",
        ],
        "🧭 API" => [
            "Tensor" => "api/tensor.md",
            "TensorNetwork" => "api/tensornetwork.md",
            "Transformations" => "api/transformations.md",
            "Quantum" => "api/quantum.md",
            "Ansatz" => "api/ansatz.md",
            "Product" => "api/product.md",
            "MPS" => "api/mps.md",
        ],
        "⚒️ Developer Reference" => [
            "Hypergraph representation" => "developer/hypergraph.md",
            "Type Hierarchy" => "developer/type-hierarchy.md",
            "Unsafe region" => "developer/unsafe-region.md",
            "Cached field" => "developer/cached-field.md",
            "Keyword Dispatch" => "developer/keyword-dispatch.md",
        ],
    ],
    pagesonly=true,
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="https://github.com/bsc-quantic/Tenet.jl",
        assets=["assets/favicon.ico", "assets/citations.css", "assets/youtube.css"],
        # build_vitepress=false,
    ),
    plugins=[bib],
    checkdocs=:exports,
    warnonly=true,
)

deploydocs(; repo="github.com/bsc-quantic/Tenet.jl.git", target="build", devbranch="master", push_preview=true)
