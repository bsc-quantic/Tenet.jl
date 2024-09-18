module TenetAdaptExt

using Tenet
using Adapt

Adapt.adapt_structure(to, x::Tensor) = Tensor(adapt(to, parent(x)), inds(x))
Adapt.adapt_structure(to, x::TensorNetwork) = TensorNetwork(adapt.(Ref(to), tensors(x)))

Adapt.adapt_structure(to, x::Quantum) = Quantum(adapt(to, TensorNetwork(x)), x.sites)
Adapt.adapt_structure(to, x::Ansatz) = Ansatz(adapt(to, Quantum(x)), lattice(x))

end
