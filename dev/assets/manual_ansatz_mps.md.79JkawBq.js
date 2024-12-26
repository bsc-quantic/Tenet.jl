import{_ as n,c as s,a5 as e,o as t}from"./chunks/framework.Dri5Xj-R.js";const h=JSON.parse('{"title":"Matrix Product States (MPS)","description":"","frontmatter":{},"headers":[],"relativePath":"manual/ansatz/mps.md","filePath":"manual/ansatz/mps.md","lastUpdated":null}'),i={name:"manual/ansatz/mps.md"};function p(o,a,r,d,l,c){return t(),s("div",null,a[0]||(a[0]=[e(`<h1 id="Matrix-Product-States-(MPS)" tabindex="-1">Matrix Product States (MPS) <a class="header-anchor" href="#Matrix-Product-States-(MPS)" aria-label="Permalink to &quot;Matrix Product States (MPS) {#Matrix-Product-States-(MPS)}&quot;">​</a></h1><p>Matrix Product States (MPS) are a Quantum Tensor Network ansatz whose tensors are laid out in a 1D chain. Due to this, these networks are also known as <em>Tensor Trains</em> in other mathematical fields. Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions).</p><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fig = Figure() # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn_open = rand(MatrixProduct{State,Open}, n=10, χ=4) # hide</span></span>
<span class="line"><span>tn_periodic = rand(MatrixProduct{State,Periodic}, n=10, χ=4) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>plot!(fig[1,1], tn_open, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide</span></span>
<span class="line"><span>plot!(fig[1,2], tn_periodic, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1,1, Bottom()], &quot;Open&quot;) # hide</span></span>
<span class="line"><span>Label(fig[1,2, Bottom()], &quot;Periodic&quot;) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig # hide</span></span></code></pre></div><h2 id="Matrix-Product-Operators-(MPO)" tabindex="-1">Matrix Product Operators (MPO) <a class="header-anchor" href="#Matrix-Product-Operators-(MPO)" aria-label="Permalink to &quot;Matrix Product Operators (MPO) {#Matrix-Product-Operators-(MPO)}&quot;">​</a></h2><p>Matrix Product Operators (MPO) are the operator version of <a href="/Tenet.jl/dev/manual/ansatz/mps#matrix-product-states-mps">Matrix Product State (MPS)</a>. The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output).</p><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fig = Figure() # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn_open = rand(MatrixProduct{Operator,Open}, n=10, χ=4) # hide</span></span>
<span class="line"><span>tn_periodic = rand(MatrixProduct{Operator,Periodic}, n=10, χ=4) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>plot!(fig[1,1], tn_open, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide</span></span>
<span class="line"><span>plot!(fig[1,2], tn_periodic, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1,1, Bottom()], &quot;Open&quot;) # hide</span></span>
<span class="line"><span>Label(fig[1,2, Bottom()], &quot;Periodic&quot;) # hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig # hide</span></span></code></pre></div><p>In <code>Tenet</code>, the generic <code>MatrixProduct</code> ansatz implements this topology. Type variables are used to address their functionality (<code>State</code> or <code>Operator</code>) and their boundary conditions (<code>Open</code> or <code>Periodic</code>).</p>`,7)]))}const P=n(i,[["render",p]]);export{h as __pageData,P as default};
