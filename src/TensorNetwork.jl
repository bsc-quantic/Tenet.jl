abstract type TensorNetwork end

ntensors(_::TensorNetwork) = error("No implementation found")
nindices(_::TensorNetwork) = error("No implementation found")

abstract type TensorNetworkState <: TensorNetwork end

nsites(_::TensorNetworkState) <: TensorNetwork end

abstract type TensorNetworkOperator <: TensorNetwork end
