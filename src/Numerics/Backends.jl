abstract type AbstractBackend end

struct OMEinsumBackend <: AbstractBackend end
struct TensorOperationsBackend <: AbstractBackend end

const default_backend::Ref{AbstractBackend} = OMEinsumBackend()

function set_backend!(backend::AbstractBackend)
    default_backend[] = backend
end
