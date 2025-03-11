import{_ as s,c as a,o as t,ai as o}from"./chunks/framework.8ZLnTjfO.js";const f=JSON.parse('{"title":"Unsafe regions","description":"","frontmatter":{},"headers":[],"relativePath":"developer/unsafe-region.md","filePath":"developer/unsafe-region.md","lastUpdated":null}'),n={name:"developer/unsafe-region.md"};function r(i,e,c,d,h,l){return t(),a("div",null,e[0]||(e[0]=[o(`<h1 id="Unsafe-regions" tabindex="-1">Unsafe regions <a class="header-anchor" href="#Unsafe-regions" aria-label="Permalink to &quot;Unsafe regions {#Unsafe-regions}&quot;">​</a></h1><p>In order to avoid inconsistency issues, <a href="/Tenet.jl/previews/PR329/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> checks the index sizes are correct whenever a <a href="/Tenet.jl/previews/PR329/api/tensor#Tensor"><code>Tensor</code></a> is <a href="/Tenet.jl/previews/PR329/api/tensornetwork#Base.push!-Tuple{Tenet.AbstractTensorNetwork, Tensor}"><code>push!</code></a>ed and it already contains some of the its indices. There are cases in which you may want to temporarily avoid index size checks (for performance or for ergonomy) on <code>push!</code> to a <a href="/Tenet.jl/previews/PR329/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>. But mutating a <a href="/Tenet.jl/previews/PR329/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> without checks is dangerous, as it can leave it in a inconsistent state which would lead to hard to trace errors.</p><p>Instead, we developed the <code>@unsafe_region</code> macro. The first argument is the <a href="./@ref"><code>AbstractTensorNetwork</code></a> you want to disable the checks for, and the second argument is the code where you modify the <a href="./@ref"><code>AbstractTensorNetwork</code></a> without checks.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@unsafe_region</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>When the scope of the <code>@unsafe_region</code> ends, it will automatically run a full check on <code>tn</code> to assert that the final state of the <a href="./@ref"><code>AbstractTensorNetwork</code></a> is consistent.</p><p>Note that this only affects disables the checks for one <a href="./@ref"><code>AbstractTensorNetwork</code></a>, but multiple <code>@unsafe_region</code>s can be nested.</p><p>DocumenterMermaid.MermaidScriptBlock([...])</p>`,7)]))}const u=s(n,[["render",r]]);export{f as __pageData,u as default};
