abstract type AbstractTensorNetwork end

ntensors(_::AbstractTensorNetwork) = error("No implementation found")
nindices(_::AbstractTensorNetwork) = error("No implementation found")

struct TensorNetwork <: AbstractTensorNetwork
end

ntensors(tn::TensorNetwork) = error("No implementation found")
nindices(tn::TensorNetwork) = error("No implementation found")

abstract type TensorNetworkState <: TensorNetwork end

nsites(tn::TensorNetworkState) = error("No mplementation found")

abstract type TensorNetworkOperator <: TensorNetwork end

nsites(_::TensorNetworkOperator) = error("No mplementation found")
