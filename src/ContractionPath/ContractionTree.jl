using Graphs

struct ContractionTree
    tree::Graph
end

# constructor
function ContractionTree() -> ContractionTree
    error("Unimplemented")
end

# contraction path
function contraction_path(tree::ContractionTree) -> Vector{Char}
    error("Unimplemented")
end

elimination_order = contraction_path

# contract
function contract(tree::ContractionTree)
    error("Unimplemented")
end

# cost
function cost(tree::ContractionTree, get::Symbol) -> Integer
    error("Unimplemented")
end

function flops(tree::ContractionTree) -> Integer
    error("Unimplemented")
end

function rank(tree::ContractionTree) -> Integer
    error("Unimplemented")
end

# tree
function leaves(tree::ContractionTree)
    error("Unimplemented")
end

function depth(tree::ContractionTree)
    error("Unimplemented")
end