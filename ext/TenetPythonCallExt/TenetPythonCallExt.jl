module TenetPythonCallExt

using Tenet
using PythonCall
using PythonCall: pyconvert_add_rule

include("pytket.jl")

function __init__()
    init_pytket()
end

end
