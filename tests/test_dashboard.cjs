// Synthetic fixtures are used only in memory; never exported as benchmark results.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('visualizer/app.js','utf8').split("$('cpuFile').onchange")[0];
function element(){return {children:[],textContent:'',append(e){this.children.push(e)},replaceChildren(){this.children=[]},setAttribute(){}};}
const elements={};
const context=vm.createContext({document:{getElementById:id=>elements[id]??=element(),createElement:element,createElementNS:element}});
vm.runInContext(source,context);
const cpu=JSON.parse(fs.readFileSync('results/cpu_results.json','utf8'));
context.fixture=cpu;
vm.runInContext("cpu=validateDataset(fixture,'CPU');render()",context);
assert.equal(vm.runInContext('comparisons().length',context),5);
assert.equal(vm.runInContext('comparisons().filter(r=>r.match).length',context),0);
const gpu=structuredClone(cpu);gpu.implementation='CUDA';
for(const r of gpu.results){r.executionTimeMs/=2;r.samplesMs=r.samplesMs.map(n=>n/2);r.stddevMs/=2;r.validation='exact-cpu-reference';r.totalTimeMs=r.executionTimeMs*1.2;}
context.fixture=gpu;
vm.runInContext("gpu=validateDataset(fixture,'CUDA');render()",context);
assert.equal(vm.runInContext('comparisons()[0].speed',context),2);
assert.equal(vm.runInContext('comparisons().filter(r=>r.match).length',context),5);
vm.runInContext("gpu.results[0].finalGridSHA256='0'.repeat(64);render()",context);
assert.equal(vm.runInContext('comparisons()[0].speed',context),null);
const invalid=structuredClone(gpu);invalid.iterations=99;context.fixture=invalid;
assert.throws(()=>vm.runInContext("validateDataset(fixture,'CUDA')",context));
invalid.iterations=100;invalid.results[0].executionTimeMs=0;
assert.throws(()=>vm.runInContext("validateDataset(fixture,'CUDA')",context));
invalid.results[0].executionTimeMs=123456;
assert.throws(()=>vm.runInContext("validateDataset(fixture,'CUDA')",context));
console.log('Dashboard tests PASSED: CPU-only, pairing, ratios, fingerprint mismatch and invalid uploads.');
