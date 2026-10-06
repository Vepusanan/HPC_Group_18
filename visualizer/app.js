'use strict';
// No third-party chart dependency: SVG charts work offline.
const $ = id => document.getElementById(id);
let cpu = null, gpu = null;
const positive = n => typeof n === 'number' && Number.isFinite(n) && n > 0;
function validateDataset(data, kind) {
  if (!data || data.schemaVersion !== 1 || data.implementation !== kind || data.iterations !== 100 ||
      data.seed !== 42 || data.boundary !== 'dead-exterior' || data.initializer !== 'lcg32-msb-v1' || data.rule !== 'B3/S23' ||
      !Number.isInteger(data.runs) || data.runs < 3 || !Array.isArray(data.results) || !data.results.length || data.results.length > 30 ||
      !data.environment || typeof data.environment.hardware !== 'string') throw Error('Incompatible JSON. Use this project’s current benchmark runner (100 iterations, seed 42).');
  const seen = new Set();
  for (const r of data.results) {
    if (!Number.isInteger(r.gridSize) || r.gridSize < 1 || r.gridSize > 8192 || seen.has(r.gridSize) || r.cells !== r.gridSize ** 2 ||
        !positive(r.executionTimeMs) || !Number.isFinite(r.stddevMs) || r.stddevMs < 0 ||
        !Array.isArray(r.samplesMs) || r.samplesMs.length !== data.runs || !r.samplesMs.every(positive) ||
        !/^[a-f0-9]{64}$/.test(r.finalGridSHA256)) throw Error('Invalid grid, timing, repetitions or result fingerprint.');
    const mean = r.samplesMs.reduce((a,b)=>a+b,0)/data.runs;
    if (Math.abs(mean-r.executionTimeMs) > Math.max(1e-6, mean*1e-6)) throw Error('Mean does not match timing samples.');
    if (kind === 'CUDA' && (r.validation !== 'exact-cpu-reference' || !positive(r.totalTimeMs))) throw Error('CUDA result lacks reference validation or total timing.');
    seen.add(r.gridSize);
  }
  if (!Array.isArray(data.gridSizes) || data.gridSizes.length !== seen.size || new Set(data.gridSizes).size !== seen.size || !data.gridSizes.every(n=>seen.has(n))) throw Error('Grid-size metadata does not match results.');
  return data;
}
function comparisons() {
  const sizes = [...new Set([...(cpu?.results||[]),...(gpu?.results||[])].map(r=>r.gridSize))].sort((a,b)=>a-b);
  return sizes.map(n=>{
    const c=cpu?.results.find(r=>r.gridSize===n), g=gpu?.results.find(r=>r.gridSize===n);
    const match=!!(c&&g&&c.finalGridSHA256===g.finalGridSHA256);
    return {n,c,g,match,speed:match?c.executionTimeMs/g.executionTimeMs:null,total:match?c.executionTimeMs/g.totalTimeMs:null};
  });
}
function node(tag, text, parent) {const e=document.createElement(tag); e.textContent=text; parent?.append(e);return e;}
const fmt = n => n==null?'—':n.toLocaleString(undefined,{maximumFractionDigits:3});
function chart(id, rows, series, log=false) {
  const host=$(id);host.replaceChildren();
  const values=series.flatMap(s=>rows.map(s.value)).filter(positive);
  if(!values.length){node('p','Upload compatible CUDA results to see this chart.',host);return;}
  const NS='http://www.w3.org/2000/svg'; const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 560 310');svg.setAttribute('role','img');svg.setAttribute('aria-label',id==='timeChart'?'Execution time by grid size':'Speedup by grid size');host.append(svg);
  function el(tag,attrs,text){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;svg.append(e);return e;}
  const lower=log?Math.floor(Math.log10(Math.min(...values))):0;
  const upper=log?Math.max(lower+1,Math.ceil(Math.log10(Math.max(...values)))):Math.max(1,...values)*1.15;
  const x=i=>65+i*460/Math.max(1,rows.length-1), y=v=>250-205*((log?Math.log10(v):v)-lower)/(upper-lower);
  for(let i=0;i<=4;i++){const v=lower+(upper-lower)*i/4, yy=250-205*i/4;el('line',{x1:65,x2:525,y1:yy,y2:yy,stroke:'#293b51'});el('text',{x:58,y:yy+4,fill:'#a3b3c7','text-anchor':'end','font-size':11},fmt(log?10**v:v));}
  rows.forEach((r,i)=>el('text',{x:x(i),y:274,fill:'#a3b3c7','text-anchor':'middle','font-size':11},String(r.n)));
  series.forEach((s,j)=>{el('text',{x:65+j*150,y:18,fill:s.color,'font-size':12},s.name);let segment=[];const flush=()=>{if(segment.length)el('polyline',{points:segment.join(' '),fill:'none',stroke:s.color,'stroke-width':2});segment=[];};rows.forEach((r,i)=>{const v=s.value(r);if(!positive(v)){flush();return;}segment.push(`${x(i)},${y(v)}`);const dot=el('circle',{cx:x(i),cy:y(v),r:4,fill:s.color});const title=document.createElementNS(NS,'title');title.textContent=`${r.n} × ${r.n}: ${fmt(v)}`;dot.append(title);});flush();});
}
function render(){
  const rows=comparisons(), matched=rows.filter(r=>r.match), peak=matched.reduce((a,b)=>!a||b.speed>a.speed?b:a,null);
  $('metrics').replaceChildren();
  for(const [label,value] of [['Iterations','100'],['CPU grids',cpu?.results.length||'—'],['CUDA grids',gpu?.results.length||'—'],['Peak speedup',peak?fmt(peak.speed)+'×':'—']]){const box=node('div','',$('metrics'));box.className='metric';node('small',label,box);node('strong',String(value),box);}
  $('environment').replaceChildren();
  for(const [name,dataset] of [['Local CPU',cpu],['Colab CUDA',gpu]]){const box=node('div','',$('environment'));node('h3',name,box);node('p',dataset?dataset.environment.hardware:'Awaiting results',box);if(dataset)node('p',`${dataset.runs} runs · ${dataset.createdAt||'date unavailable'}`,box);}
  $('rows').replaceChildren();
  rows.forEach(r=>{const tr=node('tr','',$('rows'));[`${r.n} × ${r.n} / ${(r.n*r.n).toLocaleString()}`,100,r.c?`${fmt(r.c.executionTimeMs)} ± ${fmt(r.c.stddevMs)}`:'—',r.g?`${fmt(r.g.executionTimeMs)} ± ${fmt(r.g.stddevMs)}`:'—',fmt(r.g?.totalTimeMs),r.match?fmt(r.speed)+'×':'—',r.match?fmt(r.total)+'×':'—',r.match?'SHA-256 match':r.c&&r.g?'MISMATCH — excluded':'Awaiting counterpart'].forEach(v=>node('td',String(v),tr));});
  chart('timeChart',rows,[{name:'CPU',color:'#78b6ff',value:r=>r.c?.executionTimeMs},{name:'CUDA simulation',color:'#70e0ba',value:r=>r.g?.executionTimeMs},{name:'CUDA total',color:'#e5b75e',value:r=>r.g?.totalTimeMs}],true);
  chart('speedChart',rows,[{name:'Simulation',color:'#70e0ba',value:r=>r.speed},{name:'Including transfers',color:'#e5b75e',value:r=>r.total}]);
  $('insights').replaceChildren(); const insight=t=>node('li',t,$('insights'));
  if(!cpu)insight('Run python3 run_benchmarks.py on your Mac, then reload this page.');
  if(cpu&&!gpu)insight('CPU results are ready. Upload cuda_results.json from Colab to calculate speedup.');
  if(rows.some(r=>r.c&&r.g&&!r.match))insight('Some final-grid fingerprints differ. Their speedups are withheld; rerun correctness checks.');
  if(peak){insight(`Highest observed simulation speedup: ${fmt(peak.speed)}× at ${peak.n} × ${peak.n}.`);const significant=matched.find(r=>r.speed>=2);insight(significant?`First tested grid reaching 2× simulation speedup: ${significant.n} × ${significant.n}. Here “significant” means a descriptive 2× threshold, not statistical significance.`:'No matched grid reaches the descriptive 2× simulation-speedup threshold.');
    if(matched.length>1){const first=matched[0],last=matched.at(-1);insight(`From ${first.n} to ${last.n}, cells increase ${fmt((last.n/first.n)**2)}×; CPU time grows ${fmt(last.c.executionTimeMs/first.c.executionTimeMs)}× and CUDA simulation time grows ${fmt(last.g.executionTimeMs/first.g.executionTimeMs)}×.`);insight(matched.every((r,i)=>!i||r.speed>=matched[i-1].speed)?'Speedup is nondecreasing across matched tested sizes.':'Speedup is not monotonic across matched tested sizes.');}
    const small=matched[0];insight(`At the smallest matched grid, CUDA total is ${fmt(small.g.totalTimeMs)} ms versus ${fmt(small.g.executionTimeMs)} ms simulation time. The difference includes transfers and host/synchronization overhead; these timings do not isolate its individual causes.`);
    if(small.g.totalTimeMs<small.g.executionTimeMs)insight('Total timing is below simulation timing in this dataset; investigate measurement noise or timing definitions.');
  }
  $('download').disabled=!rows.length;
}
function status(text,error=false){$('status').textContent=text;$('status').className=error?'error':'';}
async function upload(file,kind){try{if(file.size>2e6)throw Error('File exceeds 2 MB.');const data=validateDataset(JSON.parse(await file.text()),kind);if(kind==='CPU')cpu=data;else gpu=data;render();status(`${kind} results loaded. ${comparisons().filter(r=>r.match).length} matched grid sizes.`);}catch(e){status(e.message,true);}}
$('cpuFile').onchange=e=>e.target.files[0]&&upload(e.target.files[0],'CPU');
$('gpuFile').onchange=e=>e.target.files[0]&&upload(e.target.files[0],'CUDA');
$('clear').onclick=()=>{gpu=null;$('gpuFile').value='';render();status('CUDA cleared. CPU results retained.');};
$('download').onclick=()=>{const rows=[['gridSize','cells','iterations','cpuTimeMs','gpuKernelTimeMs','gpuTotalTimeMs','speedup','speedupTotal','fingerprintMatch'],...comparisons().map(r=>[r.n,r.n*r.n,100,r.c?.executionTimeMs??'',r.g?.executionTimeMs??'',r.g?.totalTimeMs??'',r.speed??'',r.total??'',r.match])];const url=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'}));const a=node('a','');a.href=url;a.download='comparison.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
fetch('../results/cpu_results.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('No local CPU results yet. Run python3 run_benchmarks.py, then reload.');return r.json();}).then(d=>{cpu=validateDataset(d,'CPU');render();status('Local CPU results loaded. Upload CUDA JSON when ready.');}).catch(e=>status(e.message,true));
render();
// Small educational simulation, independent from measured C++/CUDA timings.
const size=48,ctx=$('life').getContext('2d');let grid=new Uint8Array(size*size),next=new Uint8Array(size*size),generation=0,timer=null;
function draw(){ctx.fillStyle='#08131c';ctx.fillRect(0,0,480,480);ctx.fillStyle='#70e0ba';grid.forEach((v,i)=>{if(v)ctx.fillRect(i%size*10,Math.floor(i/size)*10,9,9);});$('generation').textContent=`Generation ${generation}`;}
function pause(){clearInterval(timer);timer=null;}
function initialize(seed){pause();for(let i=0;i<grid.length;i++){seed=(Math.imul(1664525,seed)+1013904223)>>>0;grid[i]=seed>>>31;}generation=0;draw();}
function step(){for(let r=0;r<size;r++)for(let c=0;c<size;c++){let count=0;for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const y=r+dr,x=c+dc;if((dr||dc)&&y>=0&&y<size&&x>=0&&x<size)count+=grid[y*size+x];}next[r*size+c]=+(count===3||(grid[r*size+c]&&count===2));}[grid,next]=[next,grid];generation++;draw();}
$('start').onclick=()=>{if(!timer)timer=setInterval(step,120);};$('pause').onclick=pause;$('step').onclick=()=>{pause();step();};$('reset').onclick=()=>initialize(42);$('random').onclick=()=>initialize(Math.random()*2**32>>>0);initialize(42);
