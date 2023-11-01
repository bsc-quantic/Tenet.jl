module TenetFiniteDifferencesExt

using Tenet
using Classes
using FiniteDifferences

function FiniteDifferences.to_vec(x::T) where {T<:absclass(TensorNetwork)}
    x_vec, back = to_vec(x.tensors)
    function TensorNetwork_from_vec(v)
        tensors = back(v)

        # TODO create function fitted for this? or maybe standardize constructors?
        T(map(fieldnames(T)) do fieldname
            if fieldname === :tensors
                tensors
            else
                getfield(x, fieldname)
            end
        end...)
    end

    return x_vec, TensorNetwork_from_vec
end

end
