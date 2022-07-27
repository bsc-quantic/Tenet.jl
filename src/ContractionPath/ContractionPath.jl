module ContractionPath

abstract type AbstractPathSolver end

struct Optimal <: AbstractPathSolver end
struct Greedy <: AbstractPathSolver end
struct RandomGreedy <: AbstractPathSolver end

include("ContractionTree.jl")

end