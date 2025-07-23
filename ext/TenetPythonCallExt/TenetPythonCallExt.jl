module TenetPythonCallExt

using Tenet
using PythonCall

include("pytket.jl")

function __init__()
    init_pytket()
end

end
