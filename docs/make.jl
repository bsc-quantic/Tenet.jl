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
        "Tensor Networks"=>"tensor-network.md",
        "Contraction"=>"contraction.md",
        "Transformations"=>"transformations.md",
        "Visualization"=>"visualization.md",
        "Alternatives"=>"alternatives.md",
        "References"=>"references.md",
        "Developer Notes"=>Any["`AbstractTensorNetwork` interface"=>"interface.md"],
    ],
    format = Documenter.HTML(
        prettyurls = false,
        assets = ["assets/favicon.ico", "assets/citations.css", "assets/youtube.css"],
    ),
    plugins = [bib],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(repo = "github.com/bsc-quantic/Tenet.jl.git", devbranch = "develop", push_preview = true)
