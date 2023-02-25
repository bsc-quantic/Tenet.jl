using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using Tenet

DocMeta.setdocmeta!(Tenet, :DocTestSetup, :(using Tenet); recursive = true)

makedocs(
    modules = [Tenet],
    sitename = "Tenet.jl",
    authors = "Sergio Sánchez Ramírez and contributors",
    pages = Any[
        "Home"=>"index.md",
        "Manual"=>[
            "Tensor" => "tensor.md",
            "Tensor Networks" => "tensor-network.md",
            "Ansatz Tensor Networks" => "ansatz.md",
        ],
        "Examples"=>[
            "Google's Quantum Advantage experiment" => "examples/google-rqc.md",
            "Matrix Product State classifier" => "examples/mps-ml.md",
        ],
        "Alternatives"=>"alternatives.md",
    ],
)