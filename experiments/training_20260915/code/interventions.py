"""Install explicit per-process interventions into the frozen CIFAR runtime."""
import hashlib
from pathlib import Path
import torch
from selector import rowwise_topk

class ReceiverPermutation:
    """Permute receiver columns of each realized R3 bipartite mask.

    Preserves each donor's eligible count and the receiver-degree multiset,
    including empty rows, at each routing call. Column permutation preserves
    correlations between donors. It is NOT a uniform random graph control.
    Fixed receiver-ID priorities use their own CPU generator; never consume
    the model's initialization/dropout/partition RNG stream. No mutable RNG
    state must be resumed. Receiver IDs and current partition determine mask.
    """
    def __init__(self,original,patches=784,seed=20260915):
        self.original=original
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.priority=torch.randperm(patches,generator=generator)
        self.device_priority={};self.calls=0;self.checked=0
    def __call__(self,adjacency,a_orig_idx,b_orig_idx,num_prefix_tokens=1,validate=True):
        allowed=self.original(adjacency,a_orig_idx,b_orig_idx,num_prefix_tokens=num_prefix_tokens,validate=validate)
        device=b_orig_idx.device
        if device not in self.device_priority:self.device_priority[device]=self.priority.to(device)
        priorities=self.device_priority[device][b_orig_idx.long()-num_prefix_tokens]
        perm=priorities.argsort(dim=-1)
        control=allowed.gather(-1,perm.unsqueeze(1).expand(-1,allowed.shape[1],-1))
        if self.checked<6:
            assert torch.equal(allowed.sum(-1),control.sum(-1))
            assert torch.equal(allowed.sum(-2).sort(-1).values,control.sum(-2).sort(-1).values)
            self.checked+=1
        self.calls+=1
        return control

def install(selector,geometry):
    import opentome.models.mergenet.model as model
    import opentome.timm.dtem as dtem
    assert selector in ('historical','rowwise') and geometry in ('global','flat8','r3','degree')
    result={'selector':selector,'geometry':geometry,'selector_implementation':'40-step FP32 Triton bisection; per-row normalization; implicit VJP' if selector=='rowwise' else 'unchanged frozen ThreTopK','support_seed':20260915 if geometry=='degree' else None}
    if selector=='rowwise':model.ThreTopK=dtem.ThreTopK=rowwise_topk
    if geometry=='degree':
        intervention=ReceiverPermutation(dtem.gather_patch_spatial_mask)
        dtem.gather_patch_spatial_mask=intervention
        result['control']='receiver-column permutation of actual R3 candidate mask; exact per-donor degree and receiver-degree multiset; fixed ID priority; not independent random edges'
    result['code_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    return result
