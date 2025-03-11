import{_ as a,c as i,ai as t,o as e}from"./chunks/framework.BjNXNzNQ.js";const n="/Tenet.jl/v0.8.0/assets/rjbkxac.CCFqWZuP.png",h="/Tenet.jl/v0.8.0/assets/udnrgnr.DtEcVRWk.png",E=JSON.parse('{"title":"The Product ansatz","description":"","frontmatter":{},"headers":[],"relativePath":"manual/design/product.md","filePath":"manual/design/product.md","lastUpdated":null}'),r={name:"manual/design/product.md"};function d(p,s,o,l,k,c){return e(),i("div",null,s[0]||(s[0]=[t('<h1 id="The-Product-ansatz" tabindex="-1">The <code>Product</code> ansatz <a class="header-anchor" href="#The-Product-ansatz" aria-label="Permalink to &quot;The `Product` ansatz {#The-Product-ansatz}&quot;">​</a></h1><p>A <a href="/Tenet.jl/v0.8.0/api/product#Product"><code>Product</code></a> is the simplest <a href="/Tenet.jl/v0.8.0/api/ansatz#Ansatz"><code>Ansatz</code></a> Tensor Network, which consists of a <a href="/Tenet.jl/v0.8.0/api/tensor#Tensor"><code>Tensor</code></a> per <a href="/Tenet.jl/v0.8.0/api/quantum#Tenet.Lane"><code>Lane</code></a> without any bonds, so all the sites are unconnected. The <a href="./@ref"><code>Socket</code></a> type of a <a href="/Tenet.jl/v0.8.0/api/product#Product"><code>Product</code></a> (whether it represents a <a href="./@ref"><code>State</code></a> or an <a href="./@ref"><code>Operator</code></a>) depends on the order of the tensors provided in the constructor.</p><h2 id="Product-State" tabindex="-1"><code>Product</code> State <a class="header-anchor" href="#Product-State" aria-label="Permalink to &quot;`Product` State {#Product-State}&quot;">​</a></h2><p>Each tensor is one-dimensional, with the only index being the output physical index.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">qtn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Product</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p><img src="'+n+'" alt=""></p><h2 id="Product-Operator" tabindex="-1"><code>Product</code> Operator <a class="header-anchor" href="#Product-Operator" aria-label="Permalink to &quot;`Product` Operator {#Product-Operator}&quot;">​</a></h2><p>Each tensor is two-dimensional, with the indices being the input and output physical indices.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">qtn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Product</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p><img src="'+h+'" alt=""></p><p>DocumenterMermaid.MermaidScriptBlock([...])</p>',11)]))}const g=a(r,[["render",d]]);export{E as __pageData,g as default};
