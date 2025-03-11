import{_ as e,c as a,o as i,ai as t}from"./chunks/framework.DP0sMrLM.js";const c=JSON.parse('{"title":"Cached field","description":"","frontmatter":{},"headers":[],"relativePath":"developer/cached-field.md","filePath":"developer/cached-field.md","lastUpdated":null}'),n={name:"developer/cached-field.md"};function h(l,s,p,r,o,d){return i(),a("div",null,s[0]||(s[0]=[t(`<h1 id="Cached-field" tabindex="-1">Cached field <a class="header-anchor" href="#Cached-field" aria-label="Permalink to &quot;Cached field {#Cached-field}&quot;">​</a></h1><p>The <code>CachedField</code> mechanism was introduced to alleviate the overhead of repeatedly calling <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Tenet.tensors"><code>tensors</code></a>. Some parts of the code (e.g. the Reactant linearization of a <a href="/Tenet.jl/v0.8.1/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>) require that the no-kwarg <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> method, which returns all the tensors present in an <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Tenet.AbstractTensorNetwork"><code>AbstractTensorNetwork</code></a>, to always return the <a href="/Tenet.jl/v0.8.1/api/tensor#Tensor"><code>Tensor</code></a>s in the same order if it hasn&#39;t been modified.</p><p>To fulfill such requirement, we sort the <a href="/Tenet.jl/v0.8.1/api/tensor#Tensor"><code>Tensor</code></a>s by their <a href="/Tenet.jl/v0.8.1/api/tensor#EinExprs.inds-Tuple{Tensor}"><code>inds</code></a> but the overhead of it is big when their number is in order of 100-1000s and <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Tenet.tensors"><code>tensors</code></a> is called repeatedly.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">A </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:i</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:j</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">B </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:j</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:k</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">C </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,), [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:a</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([A,B,C])</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">tensors</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> inds</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>3-element Vector{Tuple{Symbol, Vararg{Symbol}}}:</span></span>
<span class="line"><span> (:a,)</span></span>
<span class="line"><span> (:i, :j)</span></span>
<span class="line"><span> (:j, :k)</span></span></code></pre></div><p>In order to avoid the repeated cost, we cache the results of <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> in a field of [<code>TensorNetwork</code>] with type <code>CachedField</code>. Such type just stores the result of the last call and a invalidation flag.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sorted_tensors</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Tenet.CachedField{Vector{Tensor}}(true, Tensor[[0.35165852078597026, 0.3346369045095444], [0.46522111874152894 0.39127459952461685; 0.11879132801922143 0.8489514253953275], [0.9270302077426045 0.7756710873563134; 0.19430422850857887 0.7558583096254348]])</span></span></code></pre></div><p>Calling <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>push!</code></a> or <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Base.pop!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>pop!</code></a> invalidates the <code>CachedField</code>, so the next time <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> is called, it will reconstruct the cache. Because any other method that can modify the result of <a href="./@ref"><code>tensors(::AbstractTensorNetwork)</code></a> relies on <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>push!</code></a> or <a href="/Tenet.jl/v0.8.1/api/tensornetwork#Base.pop!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>pop!</code></a>, the cache is always invalidated correctly.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">delete!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn, C)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sorted_tensors</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">isvalid</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>false</span></span></code></pre></div><p>DocumenterMermaid.MermaidScriptBlock([...])</p>`,12)]))}const E=e(n,[["render",h]]);export{c as __pageData,E as default};
