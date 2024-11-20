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
    alt: Reactant.jl

features:
  - icon: 🚀
    title: Fast & Device Agnostic
    details: Its deep integration with Reactant.jl and carefully developed code, make it go brrrr
    link: /introduction

  - icon: ∂
    title: Built-In MLIR AD
    details: Leverage Enzyme-Powered Automatic Differentiation to Differentiate MLIR Functions
    link: /introduction

  - icon: 🧩
    title: Composable
    details: Executes and optimizes generic Julia code without requiring special rewriting
    link: /introduction

  - icon: 🫂
    title: Compatible with other libraries
    details: 
    link: /introduction
---
```


!!! info "BSC-Quantic's Registry"
    `Tenet` and some of its dependencies are located in our [own Julia registry](https://github.com/bsc-quantic/Registry).
    In order to download `Tenet`, add our registry to your Julia installation by using the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) in a REPL session,

    ```julia
    using Pkg
    pkg"registry add https://github.com/bsc-quantic/Registry"
    ```

<!--
A Julia library for **_Ten_**sor **_Net_**works. `Tenet` can be executed both at local environments and on large supercomputers. Its goals are,

- **Expressiveness** _Simple to use_ 👶
- **Flexibility** _Extend it to your needs_ 🔧
- **Performance** _Goes brr... fast_ 🏎️

A video of its presentation at JuliaCon 2023 can be seen here:

```@raw html
<div class="youtube-video">
<iframe class="youtube-video" width="560" src="https://www.youtube-nocookie.com/embed/8BHGtm6FRMk?si=bPXB6bPtK695HFIR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
```
 -->

## Features

- Optimized Tensor Network contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl)
- Tensor Network slicing/cuttings
- Automatic Differentiation of TN contraction, powered by [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) and [`ChainRules`](https://github.com/JuliaDiff/ChainRulesCore.jl)
- 3D visualization of large networks, powered by [`Makie`](https://github.com/MakieOrg/Makie.jl)
