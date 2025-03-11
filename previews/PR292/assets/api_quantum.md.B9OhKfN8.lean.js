import{_ as l,c as p,j as s,a as t,G as i,a5 as n,B as r,o}from"./chunks/framework.BrNYpkvI.js";const J=JSON.parse('{"title":"Quantum","description":"","frontmatter":{},"headers":[],"relativePath":"api/quantum.md","filePath":"api/quantum.md","lastUpdated":null}'),d={name:"api/quantum.md"},u={class:"jldocstring custom-block",open:""},h={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},k={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""},g={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""},y={class:"jldocstring custom-block",open:""},j={class:"jldocstring custom-block",open:""},T={class:"jldocstring custom-block",open:""},E={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},C={class:"jldocstring custom-block",open:""},F={class:"jldocstring custom-block",open:""},A={class:"jldocstring custom-block",open:""},Q={class:"jldocstring custom-block",open:""},q={class:"jldocstring custom-block",open:""},B={class:"jldocstring custom-block",open:""},w={class:"jldocstring custom-block",open:""},L={class:"jldocstring custom-block",open:""},x={class:"jldocstring custom-block",open:""},S={class:"jldocstring custom-block",open:""},R={class:"jldocstring custom-block",open:""},P={class:"jldocstring custom-block",open:""},D={class:"jldocstring custom-block",open:""};function O(M,e,N,I,V,$){const a=r("Badge");return o(),p("div",null,[e[78]||(e[78]=s("h1",{id:"quantum",tabindex:"-1"},[t("Quantum "),s("a",{class:"header-anchor",href:"#quantum","aria-label":'Permalink to "Quantum"'},"​")],-1)),s("details",u,[s("summary",null,[e[0]||(e[0]=s("a",{id:"Tenet.Lane",href:"#Tenet.Lane"},[s("span",{class:"jlbinding"},"Tenet.Lane")],-1)),e[1]||(e[1]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[2]||(e[2]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lane</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(id)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Lane</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, j, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lane</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;i,j,...&quot;</span></span></code></pre></div><p>Represents the location of a physical index.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Site"><code>Site</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.lanes-Tuple{Tenet.AbstractQuantum}"><code>lanes</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L3-L11" target="_blank" rel="noreferrer">source</a></p>`,4))]),s("details",h,[s("summary",null,[e[3]||(e[3]=s("a",{id:"Tenet.Site",href:"#Tenet.Site"},[s("span",{class:"jlbinding"},"Tenet.Site")],-1)),e[4]||(e[4]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[5]||(e[5]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Site</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(id[; dual </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Site</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, j, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[; dual </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">site</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;i,j,...[&#39;]&quot;</span></span></code></pre></div><p>Represents a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Lane"><code>Lane</code></a> with an annotation of input or output. <code>Site</code> objects are used to label the indices of tensors in a <a href="/Tenet.jl/previews/PR292/api/quantum#Quantum"><code>Quantum</code></a> Tensor Network.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Lane"><code>Lane</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.sites"><code>sites</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.isdual-Tuple{Site}"><code>isdual</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L40-L49" target="_blank" rel="noreferrer">source</a></p>`,4))]),s("details",c,[s("summary",null,[e[6]||(e[6]=s("a",{id:"Base.adjoint-Tuple{Site}",href:"#Base.adjoint-Tuple{Site}"},[s("span",{class:"jlbinding"},"Base.adjoint")],-1)),e[7]||(e[7]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[8]||(e[8]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(site</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Site</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns the adjoint of <code>site</code>, i.e. a new <code>Site</code> object with the same coordinates as <code>site</code> but with the <code>dual</code> flag flipped (so an <em>input</em> site becomes an <em>output</em> site and vice versa).</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L75-L79" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",k,[s("summary",null,[e[9]||(e[9]=s("a",{id:"Tenet.id",href:"#Tenet.id"},[s("span",{class:"jlbinding"},"Tenet.id")],-1)),e[10]||(e[10]=t()),i(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[11]||(e[11]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">id</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lane</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLane</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns the coordinate location of the <code>lane</code>.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.lanes-Tuple{Tenet.AbstractQuantum}"><code>lanes</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L24-L30" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",b,[s("summary",null,[e[12]||(e[12]=s("a",{id:"Tenet.isdual-Tuple{Site}",href:"#Tenet.isdual-Tuple{Site}"},[s("span",{class:"jlbinding"},"Tenet.isdual")],-1)),e[13]||(e[13]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[14]||(e[14]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">isdual</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(site</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Site</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns <code>true</code> if <code>site</code> is a dual site (i.e. is a &quot;input&quot;), <code>false</code> otherwise (i.e. is an &quot;output&quot;).</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Base.adjoint-Tuple{Site}"><code>adjoint(::Site)</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L65-L71" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",g,[s("summary",null,[e[15]||(e[15]=s("a",{id:"Tenet.@lane_str-Tuple{Any}",href:"#Tenet.@lane_str-Tuple{Any}"},[s("span",{class:"jlbinding"},"Tenet.@lane_str")],-1)),e[16]||(e[16]=t()),i(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),e[17]||(e[17]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lane</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;i,j,...&quot;</span></span></code></pre></div><p>Constructs a <code>Lane</code> object with the given coordinates. The coordinates are given as a comma-separated list of integers.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Lane"><code>Lane</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.@site_str-Tuple{Any}"><code>@site_str</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L82-L88" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",m,[s("summary",null,[e[18]||(e[18]=s("a",{id:"Tenet.@site_str-Tuple{Any}",href:"#Tenet.@site_str-Tuple{Any}"},[s("span",{class:"jlbinding"},"Tenet.@site_str")],-1)),e[19]||(e[19]=t()),i(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),e[20]||(e[20]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">site</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;i,j,...[&#39;]&quot;</span></span></code></pre></div><p>Constructs a <code>Site</code> object with the given coordinates. The coordinates are given as a comma-separated list of integers. Optionally, a trailing <code>&#39;</code> can be added to indicate that the site is a dual site (i.e. an &quot;input&quot;).</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Site"><code>Site</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.@lane_str-Tuple{Any}"><code>@lane_str</code></a></p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Site.jl#L100-L106" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",f,[s("summary",null,[e[21]||(e[21]=s("a",{id:"Tenet.AbstractQuantum",href:"#Tenet.AbstractQuantum"},[s("span",{class:"jlbinding"},"Tenet.AbstractQuantum")],-1)),e[22]||(e[22]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[23]||(e[23]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractQuantum</span></span></code></pre></div><p>Abstract type for <code>Quantum</code>-derived types. Its subtypes must implement conversion or extraction of the underlying <code>Quantum</code> by overloading the <code>Quantum</code> constructor.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L31-L36" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",y,[s("summary",null,[e[24]||(e[24]=s("a",{id:"Tenet.Operator",href:"#Tenet.Operator"},[s("span",{class:"jlbinding"},"Tenet.Operator")],-1)),e[25]||(e[25]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[26]||(e[26]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Operator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Socket</span></span></code></pre></div><p>Socket representing an operator; i.e. a Tensor Network with both input and output sites.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L24-L28" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",j,[s("summary",null,[e[27]||(e[27]=s("a",{id:"Tenet.Quantum",href:"#Tenet.Quantum"},[s("span",{class:"jlbinding"},"Tenet.Quantum")],-1)),e[28]||(e[28]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[29]||(e[29]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Quantum</span></span></code></pre></div><p>Tensor Network with a notion of &quot;causality&quot;. This leads to the concept of sites and directionality (input/output).</p><p><strong>Notes</strong></p><ul><li>Indices are referenced by <code>Site</code>s.</li></ul><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L39-L47" target="_blank" rel="noreferrer">source</a></p>',5))]),s("details",T,[s("summary",null,[e[30]||(e[30]=s("a",{id:"Tenet.Scalar",href:"#Tenet.Scalar"},[s("span",{class:"jlbinding"},"Tenet.Scalar")],-1)),e[31]||(e[31]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[32]||(e[32]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Scalar </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Socket</span></span></code></pre></div><p>Socket representing a scalar; i.e. a Tensor Network with no open sites.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L8-L12" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",E,[s("summary",null,[e[33]||(e[33]=s("a",{id:"Tenet.Socket",href:"#Tenet.Socket"},[s("span",{class:"jlbinding"},"Tenet.Socket")],-1)),e[34]||(e[34]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[35]||(e[35]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Socket</span></span></code></pre></div><p>Abstract type representing the socket trait of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L1-L5" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",v,[s("summary",null,[e[36]||(e[36]=s("a",{id:"Tenet.State",href:"#Tenet.State"},[s("span",{class:"jlbinding"},"Tenet.State")],-1)),e[37]||(e[37]=t()),i(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[38]||(e[38]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">State </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Socket</span></span></code></pre></div><p>Socket representing a state; i.e. a Tensor Network with only input sites (or only output sites if <code>dual = true</code>).</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L15-L19" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",C,[s("summary",null,[e[39]||(e[39]=s("a",{id:"Tenet.TensorNetwork-Tuple{Tenet.AbstractQuantum}",href:"#Tenet.TensorNetwork-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Tenet.TensorNetwork")],-1)),e[40]||(e[40]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[41]||(e[41]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TensorNetwork</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the underlying <code>TensorNetwork</code> of an <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L74-L78" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",F,[s("summary",null,[e[42]||(e[42]=s("a",{id:"Base.adjoint-Tuple{Tenet.AbstractQuantum}",href:"#Base.adjoint-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Base.adjoint")],-1)),e[43]||(e[43]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[44]||(e[44]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the adjoint of a <a href="/Tenet.jl/previews/PR292/api/quantum#Quantum"><code>Quantum</code></a> Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L357-L361" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",A,[s("summary",null,[e[45]||(e[45]=s("a",{id:"Base.merge!-Tuple{Tenet.AbstractQuantum, Tenet.AbstractQuantum}",href:"#Base.merge!-Tuple{Tenet.AbstractQuantum, Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Base.merge!")],-1)),e[46]||(e[46]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[47]||(e[47]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">merge!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; reset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Merge in-place multiple <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Networks. If <code>reset=true</code>, then all indices are renamed. If <code>reset=false</code>, then only the indices of the input/output sites are renamed.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Base.merge-Tuple{Vararg{Tenet.AbstractQuantum}}"><code>merge</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.@reindex!"><code>@reindex!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L395-L401" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",Q,[s("summary",null,[e[48]||(e[48]=s("a",{id:"Base.merge-Tuple{Vararg{Tenet.AbstractQuantum}}",href:"#Base.merge-Tuple{Vararg{Tenet.AbstractQuantum}}"},[s("span",{class:"jlbinding"},"Base.merge")],-1)),e[49]||(e[49]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[50]||(e[50]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(a</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, b</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; reset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Merge multiple <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Networks. If <code>reset=true</code>, then all indices are renamed. If <code>reset=false</code>, then only the indices of the input/output sites are renamed.</p><p>See also: <a href="/Tenet.jl/previews/PR292/api/quantum#Base.merge!-Tuple{Tenet.AbstractQuantum, Tenet.AbstractQuantum}"><code>merge!</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.@reindex!"><code>@reindex!</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L385-L391" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",q,[s("summary",null,[e[51]||(e[51]=s("a",{id:"LinearAlgebra.adjoint!-Tuple{Tenet.AbstractQuantum}",href:"#LinearAlgebra.adjoint!-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"LinearAlgebra.adjoint!")],-1)),e[52]||(e[52]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[53]||(e[53]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">LinearAlgebra</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjoint!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Like <a href="/Tenet.jl/previews/PR292/api/quantum#Base.adjoint-Tuple{Site}"><code>adjoint</code></a>, but in-place.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L364-L368" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",B,[s("summary",null,[e[54]||(e[54]=s("a",{id:"LinearAlgebra.norm",href:"#LinearAlgebra.norm"},[s("span",{class:"jlbinding"},"LinearAlgebra.norm")],-1)),e[55]||(e[55]=t()),i(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[56]||(e[56]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">LinearAlgebra</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">norm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, p</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the Lp-norm of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Only L2-norm is implemented yet.</p></div><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L423-L431" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",w,[s("summary",null,[e[57]||(e[57]=s("a",{id:"Tenet.isconnectable-Tuple{Any, Any}",href:"#Tenet.isconnectable-Tuple{Any, Any}"},[s("span",{class:"jlbinding"},"Tenet.isconnectable")],-1)),e[58]||(e[58]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[59]||(e[59]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">isconnectable</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(a</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, b</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return <code>true</code> if two <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Networks can be connected. This means:</p><ol><li><p>The outputs of <code>a</code> are a superset of the inputs of <code>b</code>.</p></li><li><p>The outputs of <code>a</code> and <code>b</code> are disjoint except for the sites that are connected.</p></li></ol><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L324-L331" target="_blank" rel="noreferrer">source</a></p>',4))]),s("details",L,[s("summary",null,[e[60]||(e[60]=s("a",{id:"Tenet.lanes-Tuple{Tenet.AbstractQuantum}",href:"#Tenet.lanes-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Tenet.lanes")],-1)),e[61]||(e[61]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[62]||(e[62]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lanes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the lanes of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L267-L271" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",x,[s("summary",null,[e[63]||(e[63]=s("a",{id:"Tenet.nlanes-Tuple{Tenet.AbstractQuantum}",href:"#Tenet.nlanes-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Tenet.nlanes")],-1)),e[64]||(e[64]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[65]||(e[65]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nlanes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the number of lanes of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L274-L278" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",S,[s("summary",null,[e[66]||(e[66]=s("a",{id:"Tenet.nsites-Tuple{Tenet.AbstractQuantum}",href:"#Tenet.nsites-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Tenet.nsites")],-1)),e[67]||(e[67]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[68]||(e[68]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nsites</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the number of sites of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L255-L259" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",R,[s("summary",null,[e[69]||(e[69]=s("a",{id:"Tenet.sites",href:"#Tenet.sites"},[s("span",{class:"jlbinding"},"Tenet.sites")],-1)),e[70]||(e[70]=t()),i(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[71]||(e[71]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">sites</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the sites of a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.AbstractQuantum"><code>AbstractQuantum</code></a> Tensor Network.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L245-L249" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",P,[s("summary",null,[e[72]||(e[72]=s("a",{id:"Tenet.socket-Tuple{Tenet.AbstractQuantum}",href:"#Tenet.socket-Tuple{Tenet.AbstractQuantum}"},[s("span",{class:"jlbinding"},"Tenet.socket")],-1)),e[73]||(e[73]=t()),i(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[74]||(e[74]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">socket</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractQuantum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Return the socket of a <a href="/Tenet.jl/previews/PR292/api/quantum#Quantum"><code>Quantum</code></a> Tensor Network; i.e. whether it is a <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Scalar"><code>Scalar</code></a>, <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.State"><code>State</code></a> or <a href="/Tenet.jl/previews/PR292/api/quantum#Tenet.Operator"><code>Operator</code></a>.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L339-L343" target="_blank" rel="noreferrer">source</a></p>',3))]),s("details",D,[s("summary",null,[e[75]||(e[75]=s("a",{id:"Tenet.@reindex!",href:"#Tenet.@reindex!"},[s("span",{class:"jlbinding"},"Tenet.@reindex!")],-1)),e[76]||(e[76]=t()),i(a,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),e[77]||(e[77]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@reindex!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> a </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b reset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span></span></code></pre></div><p>Rename in-place the indices of the input/output sites of two <a href="/Tenet.jl/previews/PR292/api/quantum#Quantum"><code>Quantum</code></a> Tensor Networks to be able to connect between them. If <code>reset=true</code>, then all indices are renamed. If <code>reset=false</code>, then only the indices of the input/output sites are renamed.</p><p><a href="https://github.com/bsc-quantic/Tenet.jl/blob/f72b20be48cd7e19ff49abcaa8df1779ed6f7818/src/Quantum.jl#L222-L227" target="_blank" rel="noreferrer">source</a></p>',3))])])}const U=l(d,[["render",O]]);export{J as __pageData,U as default};
