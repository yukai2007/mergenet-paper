"""Per-sample soft cardinality with a stable threshold and implicit gradient.

CUDA: one Triton program per row, 40 FP32 bisection iterations. CPU/FP64:
PyTorch bisection. Normalize in FP32 for half inputs. Historical selector is
not modified. No model parameter is introduced. Not a change to hard top-k.
"""
import math
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _solve(Z,Y,H,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
        row=tl.program_id(0);i=tl.arange(0,BLOCK);mask=i<N
        z=tl.load(Z+row*N+i,mask=mask,other=0).to(tl.float32)
        margin=tl.log(2.*N)
        lo=tl.min(tl.where(mask,z,float('inf')),0)-margin
        hi=tl.max(tl.where(mask,z,float('-inf')),0)+margin
        for _ in range(40):
            t=(lo+hi)*.5;delta=z-t;tail=.5*tl.exp(-tl.abs(delta))
            h=tl.where(delta>0,1.-tail,tail)
            if K <= N//2:
                above=tl.sum(tl.where(mask,h,0),0)>K
            else:
                deficit=tl.where(delta>0,tail,1.-tail)
                above=tl.sum(tl.where(mask,deficit,0),0)<(N-K)
            lo=tl.where(above,t,lo);hi=tl.where(above,hi,t)
        delta=z-(lo+hi)*.5;hp=.5*tl.exp(-tl.abs(delta))
        y=tl.where(delta>0,1.-hp,hp)
        tl.store(Y+row*N+i,y,mask=mask);tl.store(H+row*N+i,hp,mask=mask)

class _Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,k):
        n=z.shape[-1]
        if z.is_cuda and z.dtype==torch.float32 and triton is not None:
            z=z.contiguous();y=torch.empty_like(z);hp=torch.empty_like(z)
            _solve[(z.shape[0],)](z,y,hp,n,k,triton.next_power_of_2(n))
        else:
            margin=math.log(2*n);lo=z.amin(-1,keepdim=True)-margin;hi=z.amax(-1,keepdim=True)+margin
            for _ in range(60 if z.dtype==torch.float64 else 40):
                t=(lo+hi)*.5;delta=z-t;tail=.5*torch.exp(-delta.abs())
                y=torch.where(delta>0,1-tail,tail);above=y.sum(-1,keepdim=True)>k
                lo=torch.where(above,t,lo);hi=torch.where(above,hi,t)
            delta=z-(lo+hi)*.5;hp=.5*torch.exp(-delta.abs());y=torch.where(delta>0,1-hp,hp)
        ctx.save_for_backward(hp);return y
    @staticmethod
    def backward(ctx,dy):
        hp,=ctx.saved_tensors
        mean=(dy*hp).sum(-1,keepdim=True)/hp.sum(-1,keepdim=True).clamp_min(torch.finfo(hp.dtype).tiny)
        return hp*(dy-mean),None

def rowwise_topk(x,k,temperature=30.):
    """Finite floating inputs [B,N] (or [N]); row mass k before downstream masks."""
    if x.ndim==1:x=x.unsqueeze(0)
    if x.ndim!=2 or not x.is_floating_point():raise ValueError('Expected floating [B,N]')
    k=int(k);n=x.shape[-1]
    if not 0<=k<=n:raise ValueError('k outside [0,N]')
    if k==0:return x*0
    if k==n:return x*0+1
    work=x if x.dtype==torch.float64 else x.float()
    lo=work.amin(-1,keepdim=True);span=work.amax(-1,keepdim=True)-lo
    z=(work-lo)/span.clamp_min(torch.finfo(work.dtype).eps)*temperature
    return _Threshold.apply(z,k).to(x.dtype)
