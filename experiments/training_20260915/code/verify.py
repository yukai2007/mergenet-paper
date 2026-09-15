"""Numerical selector and candidate-mask checks before any long training."""
import os,sys,pathlib,json,time,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']='7';os.environ['OPENTOME_SKIP_OPTIONAL_NLP']='1'
sys.path[:0]=[str(ROOT/'runtime/imagenet_longtrain_v1'),'/liziqing/yukai/.deps_mergenet_resize20260810','/liziqing/yukai/mergenet-paper/experiments/soft_topk']
import torch
from selector import rowwise_topk
from thretopk_bisection_reference import thretopk_bisection_reference as reference
from interventions import ReceiverPermutation
import opentome.timm.dtem as dtem
torch.set_num_threads(4);torch.manual_seed(42)
result={'cases':[],'gradchecks':[],'mask_checks':[],'environment':{'torch':torch.__version__,'device':torch.cuda.get_device_name()},'complete':False}
for dtype in [torch.float32,torch.float64]:
 for n in [4,16,64,392,784]:
  for k in sorted(set([1,n//2,n-1])):
   for distribution in ['random','constant','near_constant','outlier']:
    x=torch.randn(2,n,dtype=dtype,device='cuda')
    if distribution=='constant':x.fill_(1.)
    elif distribution=='near_constant':x=1+x*1e-6
    elif distribution=='outlier':x[1,0]=1e3
    x.requires_grad_();refx=x.detach().clone().requires_grad_();y=rowwise_topk(x,k);ref=reference(refx,k)
    dy=torch.linspace(-1,1,n,device='cuda',dtype=dtype).expand_as(x)
    (y*dy).sum().backward();(ref*dy).sum().backward()
    delta=(y-ref).abs().max().item();mass=(y.sum(-1)-k).abs().max().item()
    finite=bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
    row={'dtype':str(dtype),'n':n,'k':k,'distribution':distribution,'max_output_error':delta,'max_mass_error':mass,'finite_forward_backward':finite}
    assert finite and delta<(3e-6 if dtype==torch.float32 else 1e-10) and mass<(5e-4 if dtype==torch.float32 else 1e-8),row
    result['cases'].append(row)
for k in [1,2,4]:
 x=torch.tensor([[.1,.7,-.4,1.2,2.3],[.4,1.8,.1,-.2,.9]],dtype=torch.float64,requires_grad=True)
 assert torch.autograd.gradcheck(lambda z:rowwise_topk(z,k),(x,),eps=1e-6,atol=2e-5,rtol=2e-4)
 result['gradchecks'].append(k)
x=torch.randn(2,392,device='cuda',requires_grad=True);base=rowwise_topk(x,65);other=x.detach().clone();other[1]*=100
assert torch.equal(base[0],rowwise_topk(other,65)[0]);base[0].square().sum().backward();assert torch.count_nonzero(x.grad[1])==0
for dtype in [torch.float16,torch.bfloat16]:
 x=torch.randn(2,392,device='cuda',dtype=dtype,requires_grad=True)
 with torch.autocast('cuda',dtype=dtype):y=rowwise_topk(x,65)
 y.float().square().mean().backward();assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
 result.setdefault('mixed_precision',[]).append({'dtype':str(dtype),'max_mass_error':float((y.float().sum(-1)-65).abs().max()),'finite':True})
# Mask checks with actual random and alternating partitions, varying sample count.
adj=dtem.build_patch_spatial_adjacency((28,28),3.,'euclidean',device=torch.device('cuda'))
control=ReceiverPermutation(dtem.gather_patch_spatial_mask)
for kind in ['alternating','random']:
 for batch in [1,2,8]:
  idx=torch.arange(784,device='cuda').expand(batch,-1) if kind=='alternating' else torch.rand(batch,784,device='cuda').argsort(-1)
  aa,bb=(idx[:,::2]+1,idx[:,1::2]+1) if kind=='alternating' else (idx[:,:392]+1,idx[:,392:]+1)
  original=dtem.gather_patch_spatial_mask(adj,aa,bb);state=torch.cuda.get_rng_state();cpu_state=torch.get_rng_state()
  changed=control(adj,aa,bb)
  assert torch.equal(original.sum(-1),changed.sum(-1))
  assert torch.equal(original.sum(-2).sort(-1).values,changed.sum(-2).sort(-1).values)
  assert torch.equal(state,torch.cuda.get_rng_state()) and torch.equal(cpu_state,torch.get_rng_state())
  assert torch.equal(changed,control(adj,aa,bb))
  overlap=(changed&original).sum().float()/original.sum()
  assert overlap<.2
  result['mask_checks'].append({'partition':kind,'batch':batch,'row_degree_exact':True,'receiver_degree_multiset_exact':True,'rng_unchanged':True,'deterministic':True,'edge_overlap_fraction':float(overlap)})
result['complete']=True;result['cross_sample_dependency']=0
(ROOT/'validation').mkdir(exist_ok=True);(ROOT/'validation/numerical.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(result['cases']),'selector cases',len(result['gradchecks']),'gradchecks',len(result['mask_checks']),'degree checks',flush=True)
