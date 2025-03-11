import{_ as s,c as a,a5 as n,o as i}from"./chunks/framework.BqptwCCd.js";const k=JSON.parse('{"title":"Unsafe regions","description":"","frontmatter":{},"headers":[],"relativePath":"developer/unsafe-region.md","filePath":"developer/unsafe-region.md","lastUpdated":null}'),t={name:"developer/unsafe-region.md"};function o(r,e,l,p,d,h){return i(),a("div",null,e[0]||(e[0]=[n(`<h1 id="Unsafe-regions" tabindex="-1">Unsafe regions <a class="header-anchor" href="#Unsafe-regions" aria-label="Permalink to &quot;Unsafe regions {#Unsafe-regions}&quot;">​</a></h1><p>There are cases in which you may want to temporarily avoid index size checks on <code>push!</code> to a <a href="/Tenet.jl/previews/PR262/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@unsafe_region</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div>`,3)]))}const g=s(t,[["render",o]]);export{k as __pageData,g as default};
