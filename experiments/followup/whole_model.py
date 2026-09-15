"""Isolated synthetic whole-model inference timing, exact patch budgets, no weights."""
import argparse, os, sys, json, hashlib, pathlib, subprocess, time, statistics, gc
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True)
p.add_argument('--deps',type=pathlib.Path,required=True)
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
p.add_argument('--parity-only',action='store_true')
p.add_argument('--loss-scale',type=float,default=1.)
p.add_argument('--parity-backend',choices=['sparse','dense'],default='sparse')
a=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
os.environ['OPENTOME_SKIP_OPTIONAL_NLP']='1';os.environ['TIMM_FUSED_ATTN']='1'
sys.path[:0]=[str(a.source.resolve()),str(a.deps.resolve())]
import torch, timm, flash_attn
import opentome.models
from opentome.timm.tome import tome_apply_patch
from opentome.timm import bias_local_attn as bla
from torch.nn import functional as F
# Identical mathematical attention; changes only the initial qkv layout.
class BNHDLocalAttention(bla.LocalAttention):
    def forward(self,x):
        B,N,C=x.shape
        q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,1,3,4).unbind(0)
        q,k=self.q_norm(q),self.k_norm(k)
        assert N>self.num_heads
        out=bla.unbiased_local_attention(q,k,v,local_window=self.local_window,dropout_p=self.attn_drop.p,training=self.training)
        if self.cls_global and self.local_window>=0 and N>1:
            cls=F.scaled_dot_product_attention(q[:,:1].transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=self.attn_drop.p if self.training else 0.,is_causal=False)
            out=out.clone();out[:,:1]=cls.transpose(1,2)
        return self.proj_drop(self.proj(out.reshape(B,N,C)))

def layout(m,new):
    n=0
    for mod in m.modules():
        if type(mod) in (bla.LocalAttention,BNHDLocalAttention):mod.__class__=BNHDLocalAttention if new else bla.LocalAttention;n+=1
    assert n==6,n

torch.set_num_threads(4)
a.out.parent.mkdir(parents=True,exist_ok=True)
uid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
def pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return sorted(int(s.split(',')[1]) for s in raw.splitlines() if uid in s)
assert pids()==[],pids()
probe=torch.empty(1,device='cuda');torch.cuda.synchronize();selfpids=pids();assert len(selfpids)==1
result={'loss_scale':a.loss_scale,'spatial_backend':a.parity_backend,'scope':'random-initialized architecture inference benchmark; no ImageNet accuracy evaluated; not a trained-checkpoint frontier','gpu':torch.cuda.get_device_name(),'gpu_uuid':uid,'torch':torch.__version__,'timm':timm.__version__,'flash_attn':flash_attn.__version__,'source':str(a.source),'source_hashes':{f:hashlib.sha256((a.source/f).read_bytes()).hexdigest() for f in ['opentome/models/mergenet/model.py','opentome/timm/tome.py','opentome/timm/dtem.py','opentome/timm/bias_local_attn.py']},'batch':64,'precision':'float32 weights and inputs, fp16 autocast','warmup':10,'iterations':30,'rounds':3,'dropout':0,'seed':42,'timings':[],'parity':[],'guards':[]}
def save():a.out.write_text(json.dumps(result,indent=2)+'\n')
def guard():
    current=pids();assert current==selfpids,(current,selfpids)
    result['guards'].append({'time':time.time(),'pids':current})
def build(res,variant):
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if variant.startswith('mn'):
        radius=5.143 if 'r5' in variant else 3.
        m=timm.create_model('mergenet_small_cls',pretrained=False,num_classes=1000,img_size=res,dtem_spatial_radius=radius,dtem_spatial_backend=a.parity_backend)
        if variant.endswith('bnhd'):layout(m,True)
    else:
        m=timm.create_model('deit_small_patch16_224',pretrained=False,num_classes=1000,img_size=res,patch_size=8)
        if variant.startswith('tome'):
            remove=(res//8)**2//2
            tome_apply_patch(m,trace_source=False,prop_attn=variant=='tome_prop_true',r=remove//12)
            m._tome_info['total_merge']=remove
    return m.cuda().eval()
def logits(out):
    return out[0] if isinstance(out,(tuple,list)) else out
def diff(x,y):
    z=(x-y).float();return {'max_abs':z.abs().max().item(),'relative_l2':(z.norm()/x.float().norm().clamp_min(1e-12)).item()}
def backward(out,x,m):
    out.float().square().mean().mul(a.loss_scale).backward()
    for param in m.parameters():
        if param.grad is not None:param.grad.div_(a.loss_scale)
    x.grad.div_(a.loss_scale)
# Whole-model training-graph parity, same input/weights/random grouping, zero dropout.
for res in [224,384]:
    guard();m=build(res,'mn_r3');m.train()
    x=torch.randn(2,3,res,res,device='cuda',requires_grad=True)
    state=torch.cuda.get_rng_state();cpu_state=torch.get_rng_state()
    with torch.autocast('cuda',dtype=torch.float16):y=logits(m(x))
    backward(y,x,m)
    grads={n:v.grad.detach().clone() for n,v in m.named_parameters() if v.grad is not None};xg=x.grad.clone();old=y.detach().clone()
    m.zero_grad(set_to_none=True);x.grad=None
    torch.cuda.set_rng_state(state);torch.set_rng_state(cpu_state)
    with torch.autocast('cuda',dtype=torch.float16):repeat=logits(m(x))
    backward(repeat,x,m)
    repeated={'output':diff(old,repeat.detach()),'input_gradient':diff(xg,x.grad),'parameter_gradients':{n:diff(v,dict(m.named_parameters())[n].grad) for n,v in grads.items()}}
    m.zero_grad(set_to_none=True);x.grad=None;layout(m,True)
    torch.cuda.set_rng_state(state);torch.set_rng_state(cpu_state)
    with torch.autocast('cuda',dtype=torch.float16):z=logits(m(x))
    backward(z,x,m)
    gd={n:diff(v,dict(m.named_parameters())[n].grad) for n,v in grads.items()}
    row={'resolution':res,'mode':'train; RNG restored; fresh per-sample routing; dropout zero','original_repeat':repeated,'output':diff(old,z.detach()),'input_gradient':diff(xg,x.grad),'parameter_gradients':gd}
    row['pass']=row['output']['relative_l2']<.002 and row['input_gradient']['relative_l2']<.003 and all(v['relative_l2']<.003 for v in gd.values())
    result['parity'].append(row);save() # Report failed thresholds; never relabel a failing parity test.
    print('PARITY',res,row['output'],row['input_gradient'],flush=True)
    del m,x,y,z,repeat,grads,xg,old;gc.collect();torch.cuda.empty_cache()
if a.parity_only:
    result['complete']=True;save();sys.exit(0)
assert a.parity_backend=='sparse', 'Timing contract requires sparse release backend'
for res in [224,384]:
    variants=['dense','tome_prop_true','tome_prop_false','mn_r3','mn_r3_bnhd']+(['mn_r5','mn_r5_bnhd'] if res==384 else [])
    for rnd in range(3):
        for variant in variants if rnd%2==0 else list(reversed(variants)):
            guard();m=build(res,variant);x=torch.randn(64,3,res,res,device='cuda')
            seen=[]
            def hook(mod,ins,out):
                t=out[0] if isinstance(out,(tuple,list)) else out
                if torch.is_tensor(t) and t.ndim==3:seen.append([type(mod).__name__,int(t.shape[1])])
            hooks=[b.register_forward_hook(hook) for b in m.modules() if type(b).__name__.endswith('Block')]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):y=logits(m(x))
            for h in hooks:h.remove()
            assert seen,variant
            expected=(res//8)**2 if variant=='dense' else (res//8)**2//2
            assert seen[-1][1]-1==expected,(variant,seen,expected)
            assert torch.isfinite(y).all()
            del y
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                for _ in range(10):m(x)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(30)]
                for s,e in events:s.record();m(x);e.record()
                torch.cuda.synchronize()
            vals=[s.elapsed_time(e) for s,e in events]
            row={'resolution':res,'variant':variant,'round':rnd,'patches':expected,'observed_blocks':seen,'hooks_removed_before_timing':True,'median_ms':statistics.median(vals),'samples_ms':vals,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'parameters':sum(p.numel() for p in m.parameters())}
            result['timings'].append(row);guard();save();print('TIME',res,rnd,variant,row['median_ms'],flush=True)
            del m,x;gc.collect();torch.cuda.empty_cache()
result['complete']=True;save()
