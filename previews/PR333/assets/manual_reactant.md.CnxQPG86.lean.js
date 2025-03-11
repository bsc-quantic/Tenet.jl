import{_ as e,c as t,o as n,j as s,ai as l,a}from"./chunks/framework.CX2rM8VP.js";const F=JSON.parse('{"title":"Acceleration with Reactant.jl","description":"","frontmatter":{},"headers":[],"relativePath":"manual/reactant.md","filePath":"manual/reactant.md","lastUpdated":null}'),h={name:"manual/reactant.md"},p={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.464ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.473ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -694 651 899","aria-hidden":"true"},r={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},o={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"2.009ex",height:"1.545ex",role:"img",focusable:"false",viewBox:"0 -683 888 683","aria-hidden":"true"};function d(E,i,c,g,y,u){return n(),t("div",null,[i[7]||(i[7]=s("h1",{id:"Acceleration-with-Reactant.jl",tabindex:"-1"},[a("Acceleration with Reactant.jl "),s("a",{class:"header-anchor",href:"#Acceleration-with-Reactant.jl","aria-label":'Permalink to "Acceleration with Reactant.jl {#Acceleration-with-Reactant.jl}"'},"​")],-1)),i[8]||(i[8]=s("p",null,[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl",target:"_blank",rel:"noreferrer"},"Reactant.jl"),a(" is a new MLIR & XLA frontend for the Julia language. It's similar to JAX, in the sense that it traces some code, compiles array operations using the XLA compiler and can run the final compiled function in CPU, GPU or TPU.")],-1)),i[9]||(i[9]=s("p",null,[a("Tenet.jl has top-class integration with "),s("a",{href:"https://github.com/EnzymeAD/Reactant.jl",target:"_blank",rel:"noreferrer"},"Reactant.jl"),a(". Let's dive in on how to combine both Tenet.jl and Reactant.jl with a simple example: compute an expectation value between a MPS and a MPO.")],-1)),s("p",null,[i[4]||(i[4]=a("Let's first initialize the state ")),s("mjx-container",p,[(n(),t("svg",k,i[0]||(i[0]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D713",d:"M161 441Q202 441 226 417T250 358Q250 338 218 252T187 127Q190 85 214 61Q235 43 257 37Q275 29 288 29H289L371 360Q455 691 456 692Q459 694 472 694Q492 694 492 687Q492 678 411 356Q329 28 329 27T335 26Q421 26 498 114T576 278Q576 302 568 319T550 343T532 361T524 384Q524 405 541 424T583 443Q602 443 618 425T634 366Q634 337 623 288T605 220Q573 125 492 57T329 -11H319L296 -104Q272 -198 272 -199Q270 -205 252 -205H239Q233 -199 233 -197Q233 -192 256 -102T279 -9Q272 -8 265 -8Q106 14 106 139Q106 174 139 264T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Q21 299 34 333T82 404T161 441Z",style:{"stroke-width":"3"}})])])],-1)]))),i[1]||(i[1]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"ψ")])],-1))]),i[5]||(i[5]=a(" and the operator ")),s("mjx-container",r,[(n(),t("svg",o,i[2]||(i[2]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D43B",d:"M228 637Q194 637 192 641Q191 643 191 649Q191 673 202 682Q204 683 219 683Q260 681 355 681Q389 681 418 681T463 682T483 682Q499 682 499 672Q499 670 497 658Q492 641 487 638H485Q483 638 480 638T473 638T464 637T455 637Q416 636 405 634T387 623Q384 619 355 500Q348 474 340 442T328 395L324 380Q324 378 469 378H614L615 381Q615 384 646 504Q674 619 674 627T617 637Q594 637 587 639T580 648Q580 650 582 660Q586 677 588 679T604 682Q609 682 646 681T740 680Q802 680 835 681T871 682Q888 682 888 672Q888 645 876 638H874Q872 638 869 638T862 638T853 637T844 637Q805 636 794 634T776 623Q773 618 704 340T634 58Q634 51 638 51Q646 48 692 46H723Q729 38 729 37T726 19Q722 6 716 0H701Q664 2 567 2Q533 2 504 2T458 2T437 1Q420 1 420 10Q420 15 423 24Q428 43 433 45Q437 46 448 46H454Q481 46 514 49Q520 50 522 50T528 55T534 64T540 82T547 110T558 153Q565 181 569 198Q602 330 602 331T457 332H312L279 197Q245 63 245 58Q245 51 253 49T303 46H334Q340 38 340 37T337 19Q333 6 327 0H312Q275 2 178 2Q144 2 115 2T69 2T48 1Q31 1 31 10Q31 12 34 24Q39 43 44 45Q48 46 59 46H65Q92 46 125 49Q139 52 144 61Q147 65 216 339T285 628Q285 635 228 637Z",style:{"stroke-width":"3"}})])])],-1)]))),i[3]||(i[3]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"H")])],-1))]),i[6]||(i[6]=a(" randomly:"))]),i[10]||(i[10]=l("",11))])}const C=e(h,[["render",d]]);export{F as __pageData,C as default};
