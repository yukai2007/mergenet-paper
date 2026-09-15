#!/usr/bin/env python3
"""Final bounded CPU verification of the independent correctness reference."""
from pathlib import Path
import json,math
import torch
from thretopk_bisection_reference import thretopk_bisection_reference as reference
OUT=Path(__file__).resolve().parents[2]/'build/diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(615)
cases=[]
for dtype in [torch.float32,torch.float64]:
    for n in [4,16,64,392]:
        templates={
            'random_positive':torch.rand(2,n,dtype=dtype)+0.1,
            'constant_zero':torch.zeros(2,n,dtype=dtype),
            'constant_one':torch.ones(2,n,dtype=dtype),
            'near_constant':torch.ones(2,n,dtype=dtype)+torch.arange(n,dtype=dtype)[None,:]*torch.finfo(dtype).eps*0.1,
            'cluster_plus_outlier':torch.cat([torch.ones(2,n-1,dtype=dtype),torch.full((2,1),4.,dtype=dtype)],dim=1),
        }
        for kind,template in templates.items():
            for k in sorted({1,n//2,n-1}):
                x=template.clone().requires_grad_(True);y=reference(x,k)
                g=torch.autograd.grad((y*torch.linspace(.1,1.,n,dtype=dtype)).sum(),x)[0]
                mass_error=float((y.sum(-1)-k).abs().max());tol=5e-4 if dtype==torch.float32 else 1e-6
                finite=bool(torch.isfinite(y).all() and torch.isfinite(g).all())
                cases.append(dict(dtype=str(dtype),N=n,k=k,input_kind=kind,outputs_finite=bool(torch.isfinite(y).all()),gradients_finite=bool(torch.isfinite(g).all()),sum_per_row=y.sum(-1).detach().tolist(),mass_max_abs_error=mass_error,mass_tolerance=tol,max_gradient_abs=float(g.abs().max()),pass_checks=finite and mass_error<=tol))
invariance=[]
for dtype in [torch.float32,torch.float64]:
    ys=[];gs=[]
    for second in [[1.2,2.2,3.2,3.8],[10.,20.,30.,40.]]:
        x=torch.tensor([[1.,2.,3.,4.],second],dtype=dtype,requires_grad=True);y=reference(x,2)
        g=torch.autograd.grad((y[0]*torch.arange(1,5,dtype=dtype)).sum(),x)[0];ys.append(y.detach());gs.append(g.detach())
    invariance.append(dict(dtype=str(dtype),output_change=float((ys[0][0]-ys[1][0]).abs().max()),first_gradient_change=float((gs[0][0]-gs[1][0]).abs().max()),cross_sample_gradient=float(torch.stack([g[1].abs().max() for g in gs]).max())))
gx=torch.tensor([[-1.2,-.33,.71,1.99,3.1],[.14,.82,1.2,2.34,4.56]],dtype=torch.float64,requires_grad=True)
gradchecks=[]
for k in [1,2,4]:
    passed=torch.autograd.gradcheck(lambda v:reference(v,k), (gx,), eps=1e-6,atol=1e-5,rtol=1e-3,raise_exception=False)
    gradchecks.append(dict(N=5,k=k,input=gx.detach().tolist(),eps=1e-6,atol=1e-5,rtol=1e-3,passed=passed))
summary=dict(cases=len(cases),passed=sum(c['pass_checks'] for c in cases),failures=[c for c in cases if not c['pass_checks']],max_mass_error_fp32=max(c['mass_max_abs_error'] for c in cases if c['dtype']=='torch.float32'),max_mass_error_fp64=max(c['mass_max_abs_error'] for c in cases if c['dtype']=='torch.float64'),invariance=invariance,gradchecks=gradchecks)
result=dict(scope='independent correctness reference; no training, checkpoints, full-model eval, GPU, performance claim, or release modifications',torch_version=torch.__version__,bisection_dtype='float64',iterations=60,cases=cases,summary=summary)
(OUT/'thretopk_bisection_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
if summary['failures'] or not all(g['passed'] for g in gradchecks) or any(r['output_change']!=0 or r['first_gradient_change']!=0 or r['cross_sample_gradient']!=0 for r in invariance):
    raise SystemExit(1)
