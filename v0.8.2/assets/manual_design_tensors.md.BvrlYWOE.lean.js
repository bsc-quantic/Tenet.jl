import{_ as l,c as i,o as t,j as s,ai as e,a as n}from"./chunks/framework.Brb4zMhR.js";const S=JSON.parse('{"title":"Tensors","description":"","frontmatter":{},"headers":[],"relativePath":"manual/design/tensors.md","filePath":"manual/design/tensors.md","lastUpdated":null}'),h={name:"manual/design/tensors.md"},p={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"1.593ex",height:"1.532ex",role:"img",focusable:"false",viewBox:"0 -677 704 677","aria-hidden":"true"},T={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.357ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 600 453","aria-hidden":"true"},d={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},r={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.357ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 600 453","aria-hidden":"true"},o={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},g={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.072ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.876ex",height:"1.618ex",role:"img",focusable:"false",viewBox:"0 -683 829 715","aria-hidden":"true"},m={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.072ex"},xmlns:"http://www.w3.org/2000/svg",width:"31.017ex",height:"2.207ex",role:"img",focusable:"false",viewBox:"0 -943.3 13709.6 975.3","aria-hidden":"true"},c={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},E={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"49.273ex",height:"2.7ex",role:"img",focusable:"false",viewBox:"0 -943.3 21778.8 1193.3","aria-hidden":"true"},F={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},C={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.357ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 600 453","aria-hidden":"true"},u={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},x={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.697ex"},xmlns:"http://www.w3.org/2000/svg",width:"14.944ex",height:"4.847ex",role:"img",focusable:"false",viewBox:"0 -950 6605.1 2142.2","aria-hidden":"true"},H={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},w={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.885ex"},xmlns:"http://www.w3.org/2000/svg",width:"9.208ex",height:"2.903ex",role:"img",focusable:"false",viewBox:"0 -891.7 4070.1 1283","aria-hidden":"true"},f={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},B={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-3.006ex"},xmlns:"http://www.w3.org/2000/svg",width:"16.345ex",height:"5.155ex",role:"img",focusable:"false",viewBox:"0 -950 7224.6 2278.6","aria-hidden":"true"},L={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},b={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.697ex"},xmlns:"http://www.w3.org/2000/svg",width:"12.75ex",height:"4.847ex",role:"img",focusable:"false",viewBox:"0 -950 5635.5 2142.2","aria-hidden":"true"},D={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},v={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.697ex"},xmlns:"http://www.w3.org/2000/svg",width:"15.536ex",height:"4.847ex",role:"img",focusable:"false",viewBox:"0 -950 6866.8 2142.2","aria-hidden":"true"};function M(A,a,j,V,Z,_){return t(),i("div",null,[a[35]||(a[35]=s("h1",{id:"tensors",tabindex:"-1"},[n("Tensors "),s("a",{class:"header-anchor",href:"#tensors","aria-label":'Permalink to "Tensors"'},"​")],-1)),a[36]||(a[36]=s("p",null,[n("If you have reached here, you probably know what a tensor is, and probably have heard many jokes about "),s("em",null,"what a tensor is"),s("sup",{class:"footnote-ref"},[s("a",{href:"#fn1",id:"fnref1"},"[1]")]),n(". Nevertheless, we are gonna give a brief remainder.")],-1)),s("p",null,[a[8]||(a[8]=n("A tensor ")),s("mjx-container",p,[(t(),i("svg",Q,a[0]||(a[0]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D447",d:"M40 437Q21 437 21 445Q21 450 37 501T71 602L88 651Q93 669 101 677H569H659Q691 677 697 676T704 667Q704 661 687 553T668 444Q668 437 649 437Q640 437 637 437T631 442L629 445Q629 451 635 490T641 551Q641 586 628 604T573 629Q568 630 515 631Q469 631 457 630T439 622Q438 621 368 343T298 60Q298 48 386 46Q418 46 427 45T436 36Q436 31 433 22Q429 4 424 1L422 0Q419 0 415 0Q410 0 363 1T228 2Q99 2 64 0H49Q43 6 43 9T45 27Q49 40 55 46H83H94Q174 46 189 55Q190 56 191 56Q196 59 201 76T241 233Q258 301 269 344Q339 619 339 625Q339 630 310 630H279Q212 630 191 624Q146 614 121 583T67 467Q60 445 57 441T43 437H40Z",style:{"stroke-width":"3"}})])])],-1)]))),a[1]||(a[1]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"T")])],-1))]),a[9]||(a[9]=n(" of order")),a[10]||(a[10]=s("sup",{class:"footnote-ref"},[s("a",{href:"#fn2",id:"fnref2"},"[2]")],-1)),a[11]||(a[11]=n()),s("mjx-container",T,[(t(),i("svg",k,a[2]||(a[2]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D45B",d:"M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z",style:{"stroke-width":"3"}})])])],-1)]))),a[3]||(a[3]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"n")])],-1))]),a[12]||(a[12]=n(" is a multilinear")),a[13]||(a[13]=s("sup",{class:"footnote-ref"},[s("a",{href:"#fn3",id:"fnref3"},"[3]")],-1)),a[14]||(a[14]=n(" function between ")),s("mjx-container",d,[(t(),i("svg",r,a[4]||(a[4]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D45B",d:"M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z",style:{"stroke-width":"3"}})])])],-1)]))),a[5]||(a[5]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"n")])],-1))]),a[15]||(a[15]=n(" vector spaces over a field ")),s("mjx-container",o,[(t(),i("svg",g,a[6]||(a[6]=[e("",1)]))),a[7]||(a[7]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")])])],-1))]),a[16]||(a[16]=n("."))]),s("mjx-container",m,[(t(),i("svg",y,a[17]||(a[17]=[e("",1)]))),a[18]||(a[18]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"T"),s("mo",null,":"),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"dim"),s("mo",{"data-mjx-texclass":"NONE"},"⁡"),s("mo",{stretchy:"false"},"("),s("mn",null,"1"),s("mo",{stretchy:"false"},")")])]),s("mo",null,"×"),s("mo",null,"⋯"),s("mo",null,"×"),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"dim"),s("mo",{"data-mjx-texclass":"NONE"},"⁡"),s("mo",{stretchy:"false"},"("),s("mi",null,"n"),s("mo",{stretchy:"false"},")")])]),s("mo",{stretchy:"false"},"↦"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")])])],-1))]),a[37]||(a[37]=s("p",null,"In layman's terms, you can view a tensor as a linear function that maps a set of vectors to a scalar.",-1)),s("mjx-container",c,[(t(),i("svg",E,a[19]||(a[19]=[e("",1)]))),a[20]||(a[20]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"T"),s("mo",{stretchy:"false"},"("),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{mathvariant:"bold"},"v")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mo",{stretchy:"false"},"("),s("mn",null,"1"),s("mo",{stretchy:"false"},")")])]),s("mo",null,","),s("mo",null,"…"),s("mo",null,","),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{mathvariant:"bold"},"v")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mo",{stretchy:"false"},"("),s("mi",null,"n"),s("mo",{stretchy:"false"},")")])]),s("mo",{stretchy:"false"},")"),s("mo",null,"="),s("mi",null,"c"),s("mo",null,"∈"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")]),s("mstyle",{scriptlevel:"0"},[s("mspace",{width:"2em"})]),s("mstyle",{scriptlevel:"0"},[s("mspace",{width:"2em"})]),s("mi",{mathvariant:"normal"},"∀"),s("mi",null,"i"),s("mo",null,","),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{mathvariant:"bold"},"v")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mo",{stretchy:"false"},"("),s("mi",null,"i"),s("mo",{stretchy:"false"},")")])]),s("mo",null,"∈"),s("msup",null,[s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"F")]),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"dim"),s("mo",{"data-mjx-texclass":"NONE"},"⁡"),s("mo",{stretchy:"false"},"("),s("mi",null,"i"),s("mo",{stretchy:"false"},")")])])])],-1))]),s("p",null,[a[23]||(a[23]=n("Just like with matrices and vectors, ")),s("mjx-container",F,[(t(),i("svg",C,a[21]||(a[21]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D45B",d:"M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z",style:{"stroke-width":"3"}})])])],-1)]))),a[22]||(a[22]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"n")])],-1))]),a[24]||(a[24]=n("-dimensional arrays of numbers can be used to represent tensors. Furthermore, scalars, vectors and matrices can be viewed as tensors of order 0, 1 and 2, respectively."))]),a[38]||(a[38]=s("p",null,"The dimensions of the tensors are usually identified with labels and known as tensor indices or just indices. By appropeately fixing the indices in a expression, a lot of different linear algebra operations can be described.",-1)),a[39]||(a[39]=s("p",null,"For example, the trace operation...",-1)),s("mjx-container",u,[(t(),i("svg",x,a[25]||(a[25]=[e("",1)]))),a[26]||(a[26]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"t"),s("mi",null,"r"),s("mo",{stretchy:"false"},"("),s("mi",null,"A"),s("mo",{stretchy:"false"},")"),s("mo",null,"="),s("munder",null,[s("mo",{"data-mjx-texclass":"OP"},"∑"),s("mi",null,"i")]),s("msub",null,[s("mi",null,"A"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"i"),s("mi",null,"i")])])])],-1))]),a[40]||(a[40]=s("p",null,"... a tranposition of dimensions...",-1)),s("mjx-container",H,[(t(),i("svg",w,a[27]||(a[27]=[e("",1)]))),a[28]||(a[28]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("msub",null,[s("mi",null,"A"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"j"),s("mi",null,"i")])]),s("mo",null,"="),s("msubsup",null,[s("mi",null,"A"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"i"),s("mi",null,"j")]),s("mi",null,"T")])])],-1))]),a[41]||(a[41]=s("p",null,"... or a matrix multiplication.",-1)),s("mjx-container",f,[(t(),i("svg",B,a[29]||(a[29]=[e("",1)]))),a[30]||(a[30]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("msub",null,[s("mi",null,"C"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"i"),s("mi",null,"k")])]),s("mo",null,"="),s("munder",null,[s("mo",{"data-mjx-texclass":"OP"},"∑"),s("mi",null,"j")]),s("msub",null,[s("mi",null,"A"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"i"),s("mi",null,"j")])]),s("msub",null,[s("mi",null,"B"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mi",null,"j"),s("mi",null,"k")])])])],-1))]),a[42]||(a[42]=e("",17)),s("mjx-container",L,[(t(),i("svg",b,a[31]||(a[31]=[e("",1)]))),a[32]||(a[32]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("msub",null,[s("mi",null,"X"),s("mi",null,"j")]),s("mo",null,"="),s("munder",null,[s("mo",{"data-mjx-texclass":"OP"},"∑"),s("mi",null,"i")]),s("msub",null,[s("mi",null,"A"),s("mi",null,"i")]),s("mi",null,"j")])],-1))]),a[43]||(a[43]=e("",3)),s("mjx-container",D,[(t(),i("svg",v,a[33]||(a[33]=[e("",1)]))),a[34]||(a[34]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("msub",null,[s("mi",null,"Y"),s("mi",null,"j")]),s("mo",null,"="),s("munder",null,[s("mo",{"data-mjx-texclass":"OP"},"∑"),s("mi",null,"i")]),s("msub",null,[s("mi",null,"A"),s("mi",null,"i")]),s("mi",null,"j"),s("msub",null,[s("mi",null,"B"),s("mi",null,"j")]),s("mi",null,"i")])],-1))]),a[44]||(a[44]=e("",16))])}const O=l(h,[["render",M]]);export{S as __pageData,O as default};
