```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Tenet.jl
  text: Hackable Tensor Networks
  tagline: 
  actions:
    - theme: brand
      text: Manual
      link: /manual
    - theme: alt
      text: API Reference 📚
      link: /api/api
    - theme: alt
      text: View on GitHub
      link: https://github.com/bsc-quantic/Tenet.jl
  image:
    src: /logo.svg
    alt: Tenet.jl

features:
  - icon: 🚀
    title: Fast & Device Agnostic
    details: Its deep integration with Reactant.jl and carefully developed code, makes it go brrrr
    link: /introduction

  - icon: ∂
    title: Built-In MLIR AD
    details: Leverage Enzyme-Powered Automatic Differentiation to Differentiate MLIR Functions
    link: /introduction

  - icon: 🧩
    title: Carefully crafted
    details: 
    link: /introduction

  - icon: 🫂
    title: Compatible with friends
    details: 
    link: /alternatives
---
```

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
