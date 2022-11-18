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
    pages=Any[
        "Home"=>"index.md",
        "Alternatives"=>"alternatives.md",
    ],
)