import{_ as e,c as a,a5 as r,o}from"./chunks/framework.BQp-rUvV.js";const s="/Tenet.jl/previews/PR278/assets/dcxrzmf.BSPoQFMq.png",n="/Tenet.jl/previews/PR278/assets/pfiedop.CAVjj1nl.png",h=JSON.parse('{"title":"Matrix Product States (MPS)","description":"","frontmatter":{},"headers":[],"relativePath":"manual/ansatz/mps.md","filePath":"manual/ansatz/mps.md","lastUpdated":null}'),i={name:"manual/ansatz/mps.md"};function d(c,t,p,u,l,P){return o(),a("div",null,t[0]||(t[0]=[r('<h1 id="Matrix-Product-States-(MPS)" tabindex="-1">Matrix Product States (MPS) <a class="header-anchor" href="#Matrix-Product-States-(MPS)" aria-label="Permalink to &quot;Matrix Product States (MPS) {#Matrix-Product-States-(MPS)}&quot;">​</a></h1><p>Matrix Product States (MPS) are a Quantum Tensor Network ansatz whose tensors are laid out in a 1D chain. Due to this, these networks are also known as <em>Tensor Trains</em> in other mathematical fields. Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions). <img src="'+s+'" alt=""></p><h2 id="Matrix-Product-Operators-(MPO)" tabindex="-1">Matrix Product Operators (MPO) <a class="header-anchor" href="#Matrix-Product-Operators-(MPO)" aria-label="Permalink to &quot;Matrix Product Operators (MPO) {#Matrix-Product-Operators-(MPO)}&quot;">​</a></h2><p>Matrix Product Operators (MPO) are the operator version of <a href="/Tenet.jl/previews/PR278/manual/ansatz/mps#matrix-product-states-mps">Matrix Product State (MPS)</a>. The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output). <img src="'+n+'" alt=""></p><p>In <code>Tenet</code>, the generic <code>MatrixProduct</code> ansatz implements this topology. Type variables are used to address their functionality (<code>State</code> or <code>Operator</code>) and their boundary conditions (<code>Open</code> or <code>Periodic</code>).</p>',5)]))}const M=e(i,[["render",d]]);export{h as __pageData,M as default};