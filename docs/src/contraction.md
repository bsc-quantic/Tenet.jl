# Contraction

Contraction path optimization and execution is delegated to the [`EinExprs`](https://github.com/bsc-quantic/EinExprs) library. A `EinExpr` is a lower-level form of a Tensor Network, in which the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. `Expr`) but in which every node represents an Einstein summation expression (aka `einsum`).

```@docs
einexpr(::TensorNetwork)
contract
contract!
```
