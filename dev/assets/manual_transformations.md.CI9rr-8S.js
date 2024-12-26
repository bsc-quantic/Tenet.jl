import{_ as a,c as s,a5 as e,o as p}from"./chunks/framework.Dri5Xj-R.js";const u=JSON.parse('{"title":"Transformations","description":"","frontmatter":{},"headers":[],"relativePath":"manual/transformations.md","filePath":"manual/transformations.md","lastUpdated":null}'),i={name:"manual/transformations.md"};function t(l,n,o,r,c,d){return p(),s("div",null,n[0]||(n[0]=[e(`<h1 id="transformations" tabindex="-1">Transformations <a class="header-anchor" href="#transformations" aria-label="Permalink to &quot;Transformations&quot;">​</a></h1><p>In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network&#39;s structure locally by permuting, contracting, factoring or truncating tensors.</p><p>A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn&#39;t necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.</p><p>Our approach is based in (Gray and Kourtis, 2021), which can also be found in <a href="https://quimb.readthedocs.io/" target="_blank" rel="noreferrer">quimb</a>.</p><p>In Tenet, we provide a set of predefined transformations which you can apply to your <code>TensorNetwork</code> using both the <code>transform</code>/<code>transform!</code> functions.</p><h2 id="Available-transformations" tabindex="-1">Available transformations <a class="header-anchor" href="#Available-transformations" aria-label="Permalink to &quot;Available transformations {#Available-transformations}&quot;">​</a></h2><h3 id="Hyperindex-converter" tabindex="-1">Hyperindex converter <a class="header-anchor" href="#Hyperindex-converter" aria-label="Permalink to &quot;Hyperindex converter {#Hyperindex-converter}&quot;">​</a></h3><h3 id="Contraction-simplification" tabindex="-1">Contraction simplification <a class="header-anchor" href="#Contraction-simplification" aria-label="Permalink to &quot;Contraction simplification {#Contraction-simplification}&quot;">​</a></h3><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>set_theme!(resolution=(800,200)) # hide</span></span>
<span class="line"><span>fig = Figure() #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l)) #hide</span></span>
<span class="line"><span>B = Tensor(rand(2, 2), (:i, :m)) #hide</span></span>
<span class="line"><span>C = Tensor(rand(2, 2, 2), (:m, :n, :o)) #hide</span></span>
<span class="line"><span>E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j)) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn = TensorNetwork([A, B, C, E]) #hide</span></span>
<span class="line"><span>reduced = transform(tn, Tenet.ContractSimplification) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide</span></span>
<span class="line"><span>graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1, 1, Bottom()], &quot;Original&quot;) #hide</span></span>
<span class="line"><span>Label(fig[1, 2, Bottom()], &quot;Transformed&quot;) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig #hide</span></span></code></pre></div><h3 id="Diagonal-reduction" tabindex="-1">Diagonal reduction <a class="header-anchor" href="#Diagonal-reduction" aria-label="Permalink to &quot;Diagonal reduction {#Diagonal-reduction}&quot;">​</a></h3><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>set_theme!(resolution=(800,200)) # hide</span></span>
<span class="line"><span>fig = Figure() #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>data = zeros(Float64, 2, 2, 2, 2) #hide</span></span>
<span class="line"><span>for i in 1:2 #hide</span></span>
<span class="line"><span>    for j in 1:2 #hide</span></span>
<span class="line"><span>        for k in 1:2 #hide</span></span>
<span class="line"><span>            data[i, i, j, k] = k #hide</span></span>
<span class="line"><span>        end #hide</span></span>
<span class="line"><span>    end #hide</span></span>
<span class="line"><span>end #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>A = Tensor(data, (:i, :j, :k, :l)) #hide</span></span>
<span class="line"><span>B = Tensor(rand(2, 2), (:i, :m)) #hide</span></span>
<span class="line"><span>C = Tensor(rand(2, 2), (:j, :n)) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn = TensorNetwork([A, B, C]) #hide</span></span>
<span class="line"><span>reduced = transform(tn, Tenet.DiagonalReduction) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide</span></span>
<span class="line"><span>graphplot!(fig[1, 2], reduced; layout=Stress(), labels=true) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1, 1, Bottom()], &quot;Original&quot;) #hide</span></span>
<span class="line"><span>Label(fig[1, 2, Bottom()], &quot;Transformed&quot;) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig #hide</span></span></code></pre></div><h3 id="Anti-diagonal-reduction" tabindex="-1">Anti-diagonal reduction <a class="header-anchor" href="#Anti-diagonal-reduction" aria-label="Permalink to &quot;Anti-diagonal reduction {#Anti-diagonal-reduction}&quot;">​</a></h3><h3 id="Dimension-truncation" tabindex="-1">Dimension truncation <a class="header-anchor" href="#Dimension-truncation" aria-label="Permalink to &quot;Dimension truncation {#Dimension-truncation}&quot;">​</a></h3><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>set_theme!(resolution=(800,200)) # hide</span></span>
<span class="line"><span>fig = Figure() #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>data = rand(3, 3, 3) #hide</span></span>
<span class="line"><span>data[:, 1:2, :] .= 0 #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>A = Tensor(data, (:i, :j, :k)) #hide</span></span>
<span class="line"><span>B = Tensor(rand(3, 3), (:j, :l)) #hide</span></span>
<span class="line"><span>C = Tensor(rand(3, 3), (:l, :m)) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn = TensorNetwork([A, B, C]) #hide</span></span>
<span class="line"><span>reduced = transform(tn, Tenet.Truncate) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>graphplot!(fig[1, 1], tn; layout=Spring(C=10), labels=true) #hide</span></span>
<span class="line"><span>graphplot!(fig[1, 2], reduced; layout=Spring(C=10), labels=true) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1, 1, Bottom()], &quot;Original&quot;) #hide</span></span>
<span class="line"><span>Label(fig[1, 2, Bottom()], &quot;Transformed&quot;) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig #hide</span></span></code></pre></div><h3 id="Split-simplification" tabindex="-1">Split simplification <a class="header-anchor" href="#Split-simplification" aria-label="Permalink to &quot;Split simplification {#Split-simplification}&quot;">​</a></h3><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>set_theme!(resolution=(800,200)) # hide</span></span>
<span class="line"><span>fig = Figure() #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>v1 = Tensor([1, 2, 3], (:i,)) #hide</span></span>
<span class="line"><span>v2 = Tensor([4, 5, 6], (:j,)) #hide</span></span>
<span class="line"><span>m1 = Tensor(rand(3, 3), (:k, :l)) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>t1 = contract(v1, v2) #hide</span></span>
<span class="line"><span>tensor = contract(t1, m1)  #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tn = TensorNetwork([ #hide</span></span>
<span class="line"><span>    tensor, #hide</span></span>
<span class="line"><span>    Tensor(rand(3, 3, 3), (:k, :m, :n)), #hide</span></span>
<span class="line"><span>    Tensor(rand(3, 3, 3), (:l, :n, :o)) #hide</span></span>
<span class="line"><span>]) #hide</span></span>
<span class="line"><span>reduced = transform(tn, Tenet.SplitSimplification) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>graphplot!(fig[1, 1], tn; layout=Stress(), labels=true) #hide</span></span>
<span class="line"><span>graphplot!(fig[1, 2], reduced, layout=Spring(C=11); labels=true) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Label(fig[1, 1, Bottom()], &quot;Original&quot;) #hide</span></span>
<span class="line"><span>Label(fig[1, 2, Bottom()], &quot;Transformed&quot;) #hide</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fig #hide</span></span></code></pre></div>`,16)]))}const m=a(i,[["render",t]]);export{u as __pageData,m as default};
