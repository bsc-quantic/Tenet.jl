using Test
using Tenet
using SafeTestsets

# add python test dependencies
run(`cp $(joinpath(@__DIR__, "CondaPkg.toml")) $(joinpath(@__DIR__, "..", "..", "CondaPkg.toml"))`)
using CondaPkg
CondaPkg.update()
using PythonCall

@testset "Python" verbose = true begin
    @safetestset "Cirq" begin
        include("test_cirq.jl")
    end

    @safetestset "Quimb" begin
        include("test_quimb.jl")
    end

    @safetestset "Qiskit" begin
        include("test_qiskit.jl")
    end

    @safetestset "Qibo" begin
        include("test_qibo.jl")
    end
end

# cleaning
run(`rm $(joinpath(@__DIR__, "..", "..", "CondaPkg.toml"))`)
