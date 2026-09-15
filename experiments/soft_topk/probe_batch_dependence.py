#!/usr/bin/env python3
"""Bounded CPU probe of a single source file. No model, dataset, GPU, or edits."""
from pathlib import Path
import importlib.util,json,hashlib,math
import torch
SRC=Path(__file__).resolve().parent/'historical_thretopk.py'
OUT=Path(__file__).resolve().parents[2]/'build/diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('mergenet_thretopk_cpu_probe',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def safe(v):
    if isinstance(v,list):return [safe(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    return v

def evaluate(label,rows,dtype):
    x=torch.tensor(rows,dtype=dtype,device='cpu',requires_grad=True)
    y=m.ThreTopK(x,2,temperature=30.0)
    # Loss uses only sample 0; gradient in another row proves sample coupling.
    loss=(y[0]*torch.tensor([1.,2.,3.,4.],dtype=dtype)).sum()
    grad=torch.autograd.grad(loss,x)[0]
    return dict(case=label,input=rows,k=2,N=4,k_over_N=0.5,temperature=30.0,dtype=str(dtype),x_range=float(x.detach().max()-x.detach().min()),output=safe(y.detach().tolist()),gradient_of_first_sample_loss=safe(grad.detach().tolist()),outputs_finite=bool(torch.isfinite(y).all()),gradients_finite=bool(torch.isfinite(grad).all()),other_sample_gradient_max_abs=float(grad[1:].detach().abs().max()) if len(rows)>1 and torch.isfinite(grad[1:]).all() else None)

cases=[];comparisons=[]
for dtype in [torch.float64,torch.float32]:
    first=[1.,2.,3.,4.]
    values=[evaluate('alone',[first],dtype),evaluate('same_range_companion',[first,[1.2,2.2,3.2,3.8]],dtype),evaluate('wide_range_companion',[first,[10.,20.,30.,40.]],dtype),evaluate('all_ones',[[1.]*4,[1.]*4],dtype),evaluate('all_zeros',[[0.]*4,[0.]*4],dtype)]
    comparisons.append(dict(dtype=str(dtype),first_sample_output_max_abs_change=max(abs(x-y) for x,y in zip(values[1]['output'][0],values[2]['output'][0])),first_sample_alone_vs_same_range_max_abs_change=max(abs(x-y) for x,y in zip(values[0]['output'][0],values[1]['output'][0])),cross_sample_gradient_max_abs=values[2]['other_sample_gradient_max_abs']))
    cases.extend(values)
result=dict(scope='single-function synthetic CPU probe; no checkpoint/full model/accuracy experiment',torch_version=torch.__version__,source_file=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),release_commit='4b4945afd93986d94ebb8bf4f44b2e0dba83d230',cases=cases,comparisons=comparisons)
(OUT/'batch_dependence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'comparisons':comparisons,'constant_cases':[{'case':r['case'],'dtype':r['dtype'],'outputs_finite':r['outputs_finite'],'gradients_finite':r['gradients_finite']} for r in cases if r['case'].startswith('all_') ]},indent=2))
