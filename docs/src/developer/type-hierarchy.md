# Inheritance and Traits

```@setup kroki
using Kroki
```

Julia (and in general, all modern languages like Rust or Go) implement Object Oriented Programming (OOP) in a rather restricted form compared to popular OOP languages like Java, C++ or Python.
In particular, they forbid _structural inheritance_; i.e. inheriting fields from parent superclass(es).

In recent years, _structural inheritance_ has increasingly been considered a bad practice, favouring _composition_ instead.

Julia design space on this topic is not completely clear. Julia has _abstract types_, which can be "inherited" but do not have fields and can't be instantiated, and _concrete types_, which cannot be inherited from them but have fields and can be instantiated. In this sense, implementing methods with Julia's abstract types act as some kind of polymorphic base class.

As of the time of writing, the type hierarchy of Tenet looks like this:

```@example kroki
mermaid"""graph TD
    id1(AbstractTensorNetwork)
    id2(AbstractQuantum)
    id3(AbstractAnsatz)
    id4(AbstractMPO)
    id5(AbstractMPS)
    id1 -->|inherits| id2 -->|inherits| id3 -->|inherits| id4 -->|inherits| id5
    id1 -->|inherits| TensorNetwork
    id2 -->|inherits| Quantum
    id3 -->|inherits| Ansatz
    id3 -->|inherits| Product
    id4 -->|inherits| MPO
    id5 -->|inherits| MPS
    Ansatz -.->|contains| Quantum -.->|contains| TensorNetwork
    Product -.->|contains| Ansatz
    MPO -.->|contains| Ansatz
    MPS -.->|contains| Ansatz
    style id1 stroke-dasharray: 5 5
    style id2 stroke-dasharray: 5 5
    style id3 stroke-dasharray: 5 5
    style id4 stroke-dasharray: 5 5
    style id5 stroke-dasharray: 5 5
"""
```
