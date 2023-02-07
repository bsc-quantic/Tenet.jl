using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using Tenet

DocMeta.setdocmeta!(Tenet, :DocTestSetup, :(using Tenet); recursive=true)

makedocs(
    modules=[Tenet],
    sitename="Tenet.jl",
    authors="Sergio Sánchez Ramírez and contributors",
    pages=Any[
        "Home"=>"index.md",
        "Tensor"=>"tensor.md",
        "Alternatives"=>"alternatives.md",
    ],
)