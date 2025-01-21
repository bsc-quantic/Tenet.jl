```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Tenet.jl
  text: The Hackable Tensor Network Library
  tagline: 
  actions:
    - theme: brand
      text: Manual
      link: /manual/design/introduction
    - theme: alt
      text: API Reference 📚
      link: /api/index
    - theme: alt
      text: View on GitHub
      link: https://github.com/bsc-quantic/Tenet.jl
  image:
    src: /logo.svg
    alt: Tenet.jl

features:
  - icon: 🧩
    title: Hackable, Flexible, Extendable!
    details: Tenet proposes a new interface for Tensor Networks which is both easy to use and to extend.
    link: /manual/design/introduction

  - icon: 🚀
    title: Accelerate your code with Reactant.jl
    details: Its deep integration with <a href="https://github.com/EnzymeAD/Reactant.jl">Reactant.jl</a> and carefully developed code, makes it go fast!
    link: /manual/reactant

  - icon: ∂
    title: Built-In MLIR AD
    details: Leverage Enzyme-Powered Automatic Differentiation to Differentiate Tensor Network contraction when used in combination with <a href="https://github.com/EnzymeAD/Reactant.jl">Reactant.jl</a>.
    link: /manual/reactant

  - icon: 🫂
    title: Compatible with friends
    details: We believe in collaboration and interoperation. As such, Tenet.jl integrates with many libraries... Even with Python!
    link: /manual/interop
---
```

Tenet.jl is a Tensor Network library written in Julia and designed to be performant, hackable and intuitive.

!!! info "BSC-Quantic's Registry"
    `Tenet` and some of its dependencies are located in our [own Julia registry](https://github.com/bsc-quantic/Registry).
    In order to download `Tenet`, add our registry to your Julia installation by using the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) in a REPL session,

    ```julia
    using Pkg
    pkg"registry add https://github.com/bsc-quantic/Registry"
    ```

## Features

- Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- Tensor Network slicing/cuttings
- Automatic Differentiation of TN contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
- 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
