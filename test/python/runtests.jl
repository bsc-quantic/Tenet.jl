using Test
using SafeTestsets

@safetestset "Python" verbose = true begin
    run(`cp $(joinpath(@__DIR__, CondaPkg.toml)) $(joinpath(@__DIR__, "..", "..", CondaPkg.toml))`)
    using Test
    using Tenet
    using CondaPkg
    CondaPkg.update()
    using PythonCall

    include("test_cirq.jl")
    include("test_quimb.jl")
    include("test_qiskit.jl")
    include("test_qibo.jl")
    run(`rm ../CondaPkg.toml`)
end
