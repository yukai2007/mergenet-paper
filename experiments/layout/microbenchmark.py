"""A100 single LocalAttention module only. No checkpoints, optimizer, or repo edits."""
import os, sys, json, pathlib, hashlib, importlib.util, subprocess, statistics, time, copy
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu',required=True,type=int,help='Physical index of an idle GPU')
parser.add_argument('--source',required=True,type=pathlib.Path,help='Release opentome/timm/bias_local_attn.py')
parser.add_argument('--deps',type=pathlib.Path,help='Optional Python dependency overlay')
parser.add_argument('--outdir',required=True,type=pathlib.Path)
args=parser.parse_args()
GPU=str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES']=GPU
if args.deps:sys.path.insert(0,str(args.deps.resolve()))
import atexit
import torch
import torch.nn.functional as F
import flash_attn
SRC=args.source.resolve()
OUT=args.outdir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('release_bias_local_attn',SRC)
bla=importlib.util.module_from_spec(spec);spec.loader.exec_module(bla)
torch.set_num_threads(4);torch.manual_seed(42)

ALLOWED_PIDS = set()
def snapshot(establish_self=False):
    g=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout
    a=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout
    gpu=[l for l in g.splitlines() if l.split(',')[0].strip()==GPU][0]
    uid=gpu.split(',')[1].strip(); current={int(l.split(',')[1]) for l in a.splitlines() if uid in l}
    if establish_self:
        assert len(current)==1, ('Cannot establish isolated self context',current)
        ALLOWED_PIDS.update(current)
    if current != ALLOWED_PIDS:raise RuntimeError('GPU'+GPU+' process set changed: '+str(current)+' expected '+str(ALLOWED_PIDS))
    return dict(time=time.time(),gpu=gpu,apps=[l for l in a.splitlines() if uid in l])

class BNHDLocalAttention(bla.LocalAttention):
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,1,3,4)
        q,k,v=qkv.unbind(0)
        q,k=self.q_norm(q),self.k_norm(k)
        # The release unbiased wrapper does not accept input_layout. Here N>H
        # is explicitly asserted; its existing inference selects BNHD.
        assert N>self.num_heads
        out=bla.unbiased_local_attention(q,k,v,local_window=self.local_window,dropout_p=self.attn_drop.p,training=self.training)
        if self.cls_global and self.local_window>=0 and N>1:
            cls=F.scaled_dot_product_attention(q[:,:1].transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=None,dropout_p=self.attn_drop.p if self.training else 0.0,is_causal=False)
            out=out.clone();out[:,:1]=cls.transpose(1,2)
        return self.proj_drop(self.proj(out.reshape(B,N,C)))


def model_pair(cls_global=True,qk_norm=False):
    kw=dict(dim=384,num_heads=6,qkv_bias=True,qk_norm=qk_norm,local_window=16,cls_global=cls_global)
    a=bla.LocalAttention(**kw).cuda();b=BNHDLocalAttention(**kw).cuda();b.load_state_dict(a.state_dict());return a,b

def delta(a,b):
    a=a.detach().float();b=b.detach().float();d=a-b
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.norm().clamp_min(1e-12)),bitwise_equal=torch.equal(a,b))

def run_parity(n,cls,qk):
    a,b=model_pair(cls,qk);a.train();b.train();x=torch.randn(2,n,384,device='cuda');xx=x.clone().requires_grad_();yy=x.clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.float16):u=a(xx);v=b(yy)
    dout=torch.randn_like(u.float())/(u.numel()**.5)
    (u.float()*dout).sum().backward();(v.float()*dout).sum().backward()
    grads={name:delta(p.grad,dict(b.named_parameters())[name].grad) for name,p in a.named_parameters() if p.grad is not None}
    r=dict(batch=2,tokens_including_cls=n,cls_global=cls,qk_norm=qk,output=delta(u,v),input_gradient=delta(xx.grad,yy.grad),parameter_gradients=grads)
    # Compare relative L2 to avoid claiming universal elementwise fp16 parity.
    r['passes_stated_relative_l2_tolerance']=r['output']['relative_l2']<.002 and r['input_gradient']['relative_l2']<.003 and all(z['relative_l2']<.003 for z in grads.values())
    assert r['passes_stated_relative_l2_tolerance'],r
    del a,b,x,xx,yy,u,v,dout;torch.cuda.empty_cache();return r

def timed(m,x,backward):
    m.train(backward)
    def one():
        if backward:
            m.zero_grad(set_to_none=True);x.grad=None
            with torch.autocast('cuda',dtype=torch.float16):out=m(x)
            out.float().square().mean().backward()
        else:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):m(x)
    for _ in range(5):one()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s,e in events:s.record();one();e.record()
    torch.cuda.synchronize()
    ms=[s.elapsed_time(e) for s,e in events]
    return dict(samples_ms=ms,median_ms=statistics.median(ms),q1_ms=statistics.quantiles(ms,n=4,method='inclusive')[0],q3_ms=statistics.quantiles(ms,n=4,method='inclusive')[2],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)

result=dict(scope='single LocalAttention module; synthetic random weights, no optimizer or ImageNet',source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),torch=torch.__version__,flash_attn=flash_attn.__version__,dtype='float32 params/input with fp16 autocast',dropout=0,dim=384,heads=6,window_radius_1d=16,precision_note='No bitwise parity assumed for backward; relative L2 tolerance forward .002, gradients .003',snapshots=[snapshot()],parity=[],timings=[])
atexit.register(lambda: (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n'))
result['host_pid_guard_note']='nvidia-smi reports host PIDs; require empty preflight, then one stable PID immediately after creating only our CUDA context'
result['process_status_NSpid']=[l for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith(('Pid:', 'NSpid:'))]
torch.cuda.init()
context_probe=torch.empty(1,device='cuda')
torch.cuda.synchronize()
result['snapshots'].append(snapshot(establish_self=True))
for n,cls,qk in [(785,True,False),(2305,True,False),(785,False,False),(785,True,True)]:
    result['parity'].append(run_parity(n,cls,qk));(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('parity',n,cls,qk,'OK',flush=True)
for batch,n in [(64,785),(16,2305)]:
    a,b=model_pair();x=torch.randn(batch,n,384,device='cuda',requires_grad=True)
    for backward in (False,True):
        for round_no in range(3):
            result['snapshots'].append(snapshot())
            order=[('original',a),('BNHD',b)] if round_no%2==0 else [('BNHD',b),('original',a)]
            for name,m in order:
                cell=dict(batch=batch,tokens_including_cls=n,backward=backward,round=round_no,variant=name,**timed(m,x,backward));result['timings'].append(cell)
                print('timing',batch,n,backward,round_no,name,cell['median_ms'],flush=True)
    del a,b,x;torch.cuda.empty_cache()
result['complete']=True
result['snapshots'].append(snapshot());result['elapsed_seconds']=time.time()-result['snapshots'][0]['time']
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('DONE',result['elapsed_seconds'],flush=True)
