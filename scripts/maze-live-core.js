(function(root){'use strict';const moves=[[-1,0],[0,1],[1,0],[0,-1]];
 function next(maze,p,a){const r=Math.floor(p/4)+moves[a][0],c=p%4+moves[a][1],n=r*4+c;return r<0||r>3||c<0||c>3||maze.walls.includes(n)?p:n;}
 function erf(x){const sign=x<0?-1:1,t=1/(1+.3275911*Math.abs(x));return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));}
 function infer(maze,position,w,mask){let x=Array.from({length:48},(_,i)=>i<16?Number(maze.walls.includes(i)):i<32?Number(i-16===position):Number(i-32===maze.goal));for(const name of ['0','2','4']){x=w[name+'.weight'].map((row,i)=>row.reduce((s,v,j)=>s+v*x[j],w[name+'.bias'][i]));if(name!=='4')x=x.map(v=>.5*v*(1+erf(v/Math.SQRT2)));}if(mask)x=x.map((v,a)=>next(maze,position,a)===position?-Infinity:v);const a=x.indexOf(Math.max(...x));return {action:a,logits:x,next:next(maze,position,a)};}
 root.MazeLive={next,infer};if(typeof module!=='undefined')module.exports=root.MazeLive;
})(typeof self!=='undefined'?self:globalThis);
