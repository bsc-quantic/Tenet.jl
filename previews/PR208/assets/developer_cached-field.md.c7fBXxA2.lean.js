import{_ as e,c as i,a5 as a,o as t}from"./chunks/framework.BXRFa3aJ.js";const c=JSON.parse('{"title":"Cached field","description":"","frontmatter":{},"headers":[],"relativePath":"developer/cached-field.md","filePath":"developer/cached-field.md","lastUpdated":null}'),n={name:"developer/cached-field.md"};function h(l,s,p,r,o,d){return t(),i("div",null,s[0]||(s[0]=[a(`<h1 id="Cached-field" tabindex="-1">Cached field <a class="header-anchor" href="#Cached-field" aria-label="Permalink to &quot;Cached field {#Cached-field}&quot;">​</a></h1><p>The <code>CachedField</code> mechanism was introduced to alleviate the overhead of repeatedly calling <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Tenet.tensors"><code>tensors</code></a>. Some parts of the code (e.g. the Reactant linearization of a <a href="/Tenet.jl/previews/PR208/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>) require that the no-kwarg <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> method, which returns all the tensors present in an <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Tenet.AbstractTensorNetwork"><code>AbstractTensorNetwork</code></a>, to always return the <a href="/Tenet.jl/previews/PR208/api/tensor#Tensor"><code>Tensor</code></a>s in the same order if it hasn&#39;t been modified.</p><p>To fulfill such requirement, we sort the <a href="/Tenet.jl/previews/PR208/api/tensor#Tensor"><code>Tensor</code></a>s by their <a href="/Tenet.jl/previews/PR208/api/tensor#EinExprs.inds-Tuple{Tensor}"><code>inds</code></a> but the overhead of it is big when their number is in order of 100-1000s and <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Tenet.tensors"><code>tensors</code></a> is called repeatedly.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">A </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:i</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:j</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">B </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:j</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:k</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">C </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:a</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([A,B,C])</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">tensors</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> inds</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>3-element Vector{Vector{Symbol}}:</span></span>
<span class="line"><span> [:a]</span></span>
<span class="line"><span> [:i, :j]</span></span>
<span class="line"><span> [:j, :k]</span></span></code></pre></div><p>In order to avoid the repeated cost, we cache the results of <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> in a field of [<code>TensorNetwork</code>] with type <code>CachedField</code>. Such type just stores the result of the last call and a invalidation flag.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sorted_tensors</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Tenet.CachedField{Vector{Tensor}}(true, Tensor[[0.62091187968517, 0.23244142749888663], [0.3471018050593194 0.5814673752231113; 0.7981433565541086 0.601171390598339], [0.05330463522195017 0.931077531291871; 0.9603296884437661 0.9181724252065666]])</span></span></code></pre></div><p>Calling <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>push!</code></a> or <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Base.pop!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>pop!</code></a> invalidates the <code>CachedField</code>, so the next time <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> is called, it will reconstruct the cache. Because any other method that can modify the result of <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> relies on <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>push!</code></a> or <a href="/Tenet.jl/previews/PR208/api/tensornetwork#Base.pop!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>pop!</code></a>, the cache is always invalidated correctly.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">delete!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn, C)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sorted_tensors</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">isvalid</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>false</span></span></code></pre></div>`,11)]))}const E=e(n,[["render",h]]);export{c as __pageData,E as default};
