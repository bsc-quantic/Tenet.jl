import{_ as l,c as o,j as e,a as i,G as a,a5 as n,B as r,o as p}from"./chunks/framework.dyMqKSEW.js";const V=JSON.parse('{"title":"TensorNetwork","description":"","frontmatter":{},"headers":[],"relativePath":"api/tensornetwork.md","filePath":"api/tensornetwork.md","lastUpdated":null}'),d={name:"api/tensornetwork.md"},k={class:"jldocstring custom-block",open:""},h={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},g={class:"jldocstring custom-block",open:""},u={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""},y={class:"jldocstring custom-block",open:""},T={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},E={class:"jldocstring custom-block",open:""},j={class:"jldocstring custom-block",open:""},w={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},F={class:"jldocstring custom-block",open:""},C={class:"jldocstring custom-block",open:""},B={class:"jldocstring custom-block",open:""},N={class:"jldocstring custom-block",open:""},A={class:"jldocstring custom-block",open:""},D={class:"jldocstring custom-block",open:""},x={class:"jldocstring custom-block",open:""};function L(R,s,q,P,S,O){const t=r("Badge");return p(),o("div",null,[s[63]||(s[63]=e("h1",{id:"tensornetwork",tabindex:"-1"},[i("TensorNetwork "),e("a",{class:"header-anchor",href:"#tensornetwork","aria-label":'Permalink to "TensorNetwork"'},"​")],-1)),e("details",k,[e("summary",null,[s[0]||(s[0]=e("a",{id:"Tenet.TensorNetwork",href:"#Tenet.TensorNetwork"},[e("span",{class:"jlbinding"},"Tenet.TensorNetwork")],-1)),s[1]||(s[1]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[2]||(s[2]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">TensorNetwork</span></span></code></pre></div><p>Graph of interconnected tensors, representing a multilinear equation. Graph vertices represent tensors and graph edges, tensor indices.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L33-L38" target="_blank" rel="noreferrer">source</a></p>',3))]),s[64]||(s[64]=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"Missing docstring."),e("p",null,[i("Missing docstring for "),e("code",null,"inds(::Tenet.TensorNetwork)"),i(". Check Documenter's build log for details.")])],-1)),e("details",h,[e("summary",null,[s[3]||(s[3]=e("a",{id:"Base.size-Tuple{TensorNetwork}",href:"#Base.size-Tuple{TensorNetwork}"},[e("span",{class:"jlbinding"},"Base.size")],-1)),s[4]||(s[4]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[5]||(s[5]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, index)</span></span></code></pre></div><p>Return a mapping from indices to their dimensionalities.</p><p>If <code>index</code> is set, return the dimensionality of <code>index</code>. This is equivalent to <code>size(tn)[index]</code>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L253-L260" target="_blank" rel="noreferrer">source</a></p>`,4))]),s[65]||(s[65]=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"Missing docstring."),e("p",null,[i("Missing docstring for "),e("code",null,"tensors(::Tenet.TensorNetwork)"),i(". Check Documenter's build log for details.")])],-1)),e("details",c,[e("summary",null,[s[6]||(s[6]=e("a",{id:"Base.push!-Tuple{TensorNetwork, Tensor}",href:"#Base.push!-Tuple{TensorNetwork, Tensor}"},[e("span",{class:"jlbinding"},"Base.push!")],-1)),s[7]||(s[7]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[8]||(s[8]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">push!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tensor</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Add a new <code>tensor</code> to the Tensor Network.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.append!-Tuple{TensorNetwork, Union{Tuple{Vararg{var&quot;#s12&quot;}}, AbstractVector{&lt;:var&quot;#s12&quot;}} where var&quot;#s12&quot;&lt;:Tensor}"><code>append!</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.pop!-Tuple{TensorNetwork, Tensor}"><code>pop!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L325-L331" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",g,[e("summary",null,[s[9]||(s[9]=e("a",{id:"Base.pop!-Tuple{TensorNetwork, Tensor}",href:"#Base.pop!-Tuple{TensorNetwork, Tensor}"},[e("span",{class:"jlbinding"},"Base.pop!")],-1)),s[10]||(s[10]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[11]||(s[11]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pop!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tensor</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tensor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pop!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Symbol,AbstractVecOrTuple{Symbol}}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Remove a tensor from the Tensor Network and returns it. If a <code>Tensor</code> is passed, then the first tensor satisfies <em>egality</em> (i.e. <code>≡</code> or <code>===</code>) will be removed. If a <code>Symbol</code> or a list of <code>Symbol</code>s is passed, then remove and return the tensors that contain all the indices.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.push!-Tuple{TensorNetwork, Tensor}"><code>push!</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.delete!-Tuple{TensorNetwork, Any}"><code>delete!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L364-L372" target="_blank" rel="noreferrer">source</a></p>`,4))]),e("details",u,[e("summary",null,[s[12]||(s[12]=e("a",{id:'Base.append!-Tuple{TensorNetwork, Union{Tuple{Vararg{var"#s12"}}, AbstractVector{<:var"#s12"}} where var"#s12"<:Tensor}',href:'#Base.append!-Tuple{TensorNetwork, Union{Tuple{Vararg{var"#s12"}}, AbstractVector{<:var"#s12"}} where var"#s12"<:Tensor}'},[e("span",{class:"jlbinding"},"Base.append!")],-1)),s[13]||(s[13]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[14]||(s[14]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">append!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tensors</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractVecOrTuple{&lt;:Tensor}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Add a list of tensors to a <code>TensorNetwork</code>.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.push!-Tuple{TensorNetwork, Tensor}"><code>push!</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.merge!-Tuple{TensorNetwork, TensorNetwork}"><code>merge!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L355-L361" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",b,[e("summary",null,[s[15]||(s[15]=e("a",{id:"Base.merge!-Tuple{TensorNetwork, TensorNetwork}",href:"#Base.merge!-Tuple{TensorNetwork, TensorNetwork}"},[e("span",{class:"jlbinding"},"Base.merge!")],-1)),s[16]||(s[16]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[17]||(s[17]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">merge!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, others</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, others</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Fuse various <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>s into one.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.append!-Tuple{TensorNetwork, Union{Tuple{Vararg{var&quot;#s12&quot;}}, AbstractVector{&lt;:var&quot;#s12&quot;}} where var&quot;#s12&quot;&lt;:Tensor}"><code>append!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L527-L534" target="_blank" rel="noreferrer">source</a></p>`,4))]),e("details",y,[e("summary",null,[s[18]||(s[18]=e("a",{id:"Base.delete!-Tuple{TensorNetwork, Any}",href:"#Base.delete!-Tuple{TensorNetwork, Any}"},[e("span",{class:"jlbinding"},"Base.delete!")],-1)),s[19]||(s[19]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[20]||(s[20]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">delete!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x)</span></span></code></pre></div><p>Like <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.pop!-Tuple{TensorNetwork, Tensor}"><code>pop!</code></a> but return the <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> instead.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L385-L389" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",T,[e("summary",null,[s[21]||(s[21]=e("a",{id:"Base.replace!",href:"#Base.replace!"},[e("span",{class:"jlbinding"},"Base.replace!")],-1)),s[22]||(s[22]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[23]||(s[23]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">replace!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, old </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> new</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">replace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, old </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> new</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Replace the element in <code>old</code> with the one in <code>new</code>. Depending on the types of <code>old</code> and <code>new</code>, the following behaviour is expected:</p><ul><li><p>If <code>Symbol</code>s, it will correspond to a index renaming.</p></li><li><p>If <code>Tensor</code>s, first element that satisfies <em>egality</em> (<code>≡</code> or <code>===</code>) will be replaced.</p></li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L412-L420" target="_blank" rel="noreferrer">source</a></p>`,4))]),e("details",f,[e("summary",null,[s[24]||(s[24]=e("a",{id:"Base.selectdim",href:"#Base.selectdim"},[e("span",{class:"jlbinding"},"Base.selectdim")],-1)),s[25]||(s[25]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[26]||(s[26]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">selectdim</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, index</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, i)</span></span></code></pre></div><p>Return a copy of the <a href="./@ref"><code>AbstractTensorNetwork</code></a> where <code>index</code> has been projected to dimension <code>i</code>.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.view-Tuple{TensorNetwork}"><code>view</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.slice!"><code>slice!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L570-L576" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",m,[e("summary",null,[s[27]||(s[27]=e("a",{id:"Tenet.slice!",href:"#Tenet.slice!"},[e("span",{class:"jlbinding"},"Tenet.slice!")],-1)),s[28]||(s[28]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[29]||(s[29]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">slice!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, index</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, i)</span></span></code></pre></div><p>In-place projection of <code>index</code> on dimension <code>i</code>.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.selectdim"><code>selectdim</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.view-Tuple{TensorNetwork}"><code>view</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L555-L561" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",E,[e("summary",null,[s[30]||(s[30]=e("a",{id:"Base.view-Tuple{TensorNetwork}",href:"#Base.view-Tuple{TensorNetwork}"},[e("span",{class:"jlbinding"},"Base.view")],-1)),s[31]||(s[31]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[32]||(s[32]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">view</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractTensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, index </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return a copy of the <a href="./@ref"><code>AbstractTensorNetwork</code></a> where each <code>index</code> has been projected to dimension <code>i</code>. It is equivalent to a recursive call of <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.selectdim"><code>selectdim</code></a>.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Base.selectdim"><code>selectdim</code></a>, <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.slice!"><code>slice!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L579-L586" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",j,[e("summary",null,[s[33]||(s[33]=e("a",{id:"Base.copy-Tuple{TensorNetwork}",href:"#Base.copy-Tuple{TensorNetwork}"},[e("span",{class:"jlbinding"},"Base.copy")],-1)),s[34]||(s[34]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[35]||(s[35]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">copy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return a shallow copy of a <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/TensorNetwork.jl#L80-L84" target="_blank" rel="noreferrer">source</a></p>',3))]),s[66]||(s[66]=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"Missing docstring."),e("p",null,[i("Missing docstring for "),e("code",null,"Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)"),i(". Check Documenter's build log for details.")])],-1)),s[67]||(s[67]=e("h2",{id:"transformations",tabindex:"-1"},[i("Transformations "),e("a",{class:"header-anchor",href:"#transformations","aria-label":'Permalink to "Transformations"'},"​")],-1)),e("details",w,[e("summary",null,[s[36]||(s[36]=e("a",{id:"Tenet.transform",href:"#Tenet.transform"},[e("span",{class:"jlbinding"},"Tenet.transform")],-1)),s[37]||(s[37]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[38]||(s[38]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">transform</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, config</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Transformation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">transform</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, configs)</span></span></code></pre></div><p>Return a new <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> where some <code>Transformation</code> has been performed into it.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.transform!"><code>transform!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L10-L17" target="_blank" rel="noreferrer">source</a></p>`,4))]),e("details",v,[e("summary",null,[s[39]||(s[39]=e("a",{id:"Tenet.transform!",href:"#Tenet.transform!"},[e("span",{class:"jlbinding"},"Tenet.transform!")],-1)),s[40]||(s[40]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[41]||(s[41]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">transform!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, config</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Transformation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">transform!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(tn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, configs)</span></span></code></pre></div><p>In-place version of <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.transform"><code>transform</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L20-L25" target="_blank" rel="noreferrer">source</a></p>`,3))]),e("details",F,[e("summary",null,[s[42]||(s[42]=e("a",{id:"Tenet.HyperFlatten",href:"#Tenet.HyperFlatten"},[e("span",{class:"jlbinding"},"Tenet.HyperFlatten")],-1)),s[43]||(s[43]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[44]||(s[44]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HyperFlatten </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Convert hyperindices to COPY-tensors, represented by <code>DeltaArray</code>s. This transformation is always used by default when visualizing a <code>TensorNetwork</code> with <code>plot</code>.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.HyperGroup"><code>HyperGroup</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L39-L46" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",C,[e("summary",null,[s[45]||(s[45]=e("a",{id:"Tenet.HyperGroup",href:"#Tenet.HyperGroup"},[e("span",{class:"jlbinding"},"Tenet.HyperGroup")],-1)),s[46]||(s[46]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[47]||(s[47]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">HyperGroup </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Convert COPY-tensors, represented by <code>DeltaArray</code>s, to hyperindices.</p><p>See also: <a href="/Tenet.jl/previews/PR261/api/tensornetwork#Tenet.HyperFlatten"><code>HyperFlatten</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L77-L83" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",B,[e("summary",null,[s[48]||(s[48]=e("a",{id:"Tenet.ContractSimplification",href:"#Tenet.ContractSimplification"},[e("span",{class:"jlbinding"},"Tenet.ContractSimplification")],-1)),s[49]||(s[49]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[50]||(s[50]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ContractSimplification </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Preemptively contract tensors whose result doesn&#39;t increase in size.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L114-L118" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",N,[e("summary",null,[s[51]||(s[51]=e("a",{id:"Tenet.DiagonalReduction",href:"#Tenet.DiagonalReduction"},[e("span",{class:"jlbinding"},"Tenet.DiagonalReduction")],-1)),s[52]||(s[52]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[53]||(s[53]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">DiagonalReduction </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Reduce the dimension of a <code>Tensor</code> in a <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> when it has a pair of indices that fulfil a diagonal structure.</p><p><strong>Keyword Arguments</strong></p><ul><li><code>atol</code> Absolute tolerance. Defaults to <code>1e-12</code>.</li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L195-L203" target="_blank" rel="noreferrer">source</a></p>',5))]),e("details",A,[e("summary",null,[s[54]||(s[54]=e("a",{id:"Tenet.AntiDiagonalGauging",href:"#Tenet.AntiDiagonalGauging"},[e("span",{class:"jlbinding"},"Tenet.AntiDiagonalGauging")],-1)),s[55]||(s[55]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[56]||(s[56]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AntiDiagonalGauging </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Reverse the order of tensor indices that fulfill the anti-diagonal condition. While this transformation doesn&#39;t directly enhance computational efficiency, it sets up the <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> for other operations that do.</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>atol</code> Absolute tolerance. Defaults to <code>1e-12</code>.</p></li><li><p><code>skip</code> List of indices to skip. Defaults to <code>[]</code>.</p></li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L244-L254" target="_blank" rel="noreferrer">source</a></p>',5))]),e("details",D,[e("summary",null,[s[57]||(s[57]=e("a",{id:"Tenet.Truncate",href:"#Tenet.Truncate"},[e("span",{class:"jlbinding"},"Tenet.Truncate")],-1)),s[58]||(s[58]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[59]||(s[59]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Truncate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Truncate the dimension of a <code>Tensor</code> in a <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> when it contains columns with all elements smaller than <code>atol</code>.</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>atol</code> Absolute tolerance. Defaults to <code>1e-12</code>.</p></li><li><p><code>skip</code> List of indices to skip. Defaults to <code>[]</code>.</p></li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L160-L169" target="_blank" rel="noreferrer">source</a></p>',5))]),e("details",x,[e("summary",null,[s[60]||(s[60]=e("a",{id:"Tenet.SplitSimplification",href:"#Tenet.SplitSimplification"},[e("span",{class:"jlbinding"},"Tenet.SplitSimplification")],-1)),s[61]||(s[61]=i()),a(t,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[62]||(s[62]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SplitSimplification </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Transformation</span></span></code></pre></div><p>Reduce the rank of tensors in the <a href="/Tenet.jl/previews/PR261/api/tensornetwork#TensorNetwork"><code>TensorNetwork</code></a> by decomposing them using the Singular Value Decomposition (SVD). Tensors whose factorization do not increase the maximum rank of the network are left decomposed.</p><p><strong>Keyword Arguments</strong></p><ul><li><code>atol</code> Absolute tolerance. Defaults to <code>1e-10</code>.</li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/2679b5875055333cbc1034f5f792e3efd8919c5a/src/Transformations.jl#L282-L291" target="_blank" rel="noreferrer">source</a></p>',5))])])}const G=l(d,[["render",L]]);export{V as __pageData,G as default};