"""Replay existing local CIFAR EMA weights and measure batch composition effects."""
import argparse, pathlib, sys, os, json, csv, hashlib, time, subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--campaign',type=pathlib.Path,required=True);p.add_argument('--deps',type=pathlib.Path,required=True)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--outdir',type=pathlib.Path,required=True)
p.add_argument('--data',type=pathlib.Path,required=True)
p.add_argument('--reference-only',action='store_true')
a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['OPENTOME_SKIP_OPTIONAL_NLP']='1';os.environ['OPENTOME_MERGENET_IMPL']='new';os.environ['TIMM_FUSED_ATTN']='1'
runtime=a.campaign/'runtime/imagenet_longtrain_v1'
sys.path[:0]=[str(runtime),str(a.deps),str(a.campaign/'evaluation/runtime/v1/cifar_dtem_spatial_20260814')]
import eval_common as common
import torch, timm, torchvision, numpy as np
import opentome.models
import opentome.models.mergenet.model as model_module
import opentome.timm.dtem as dtem_module
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'soft_topk'))
from thretopk_bisection_reference import thretopk_bisection_reference
from torch.utils.data import DataLoader, Subset
torch.set_num_threads(4);torch.manual_seed(42)
a.outdir.mkdir(parents=True,exist_ok=True)
uid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
def pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return sorted(int(s.split(',')[1]) for s in raw.splitlines() if uid in s)
assert pids()==[]
probe=torch.empty(1,device='cuda');torch.cuda.synchronize();selfpids=pids();assert len(selfpids)==1
protocol,experiments=common.load_protocol(common.default_protocol(a.campaign))
data_evidence=common.validate_data(a.data)
transform=timm.data.create_transform(input_size=(3,224,224),is_training=False,interpolation='bicubic',mean=common.CIFAR100_MEAN,std=common.CIFAR100_STD,crop_pct=.9)
dataset=torchvision.datasets.CIFAR100(root=str(a.data),train=False,download=False,transform=transform)
labels=torch.tensor(dataset.targets)
result={'scope':'local CIFAR-100 held-out test replay only; not company ImageNet weights','runtime':str(runtime),'torch':torch.__version__,'timm':timm.__version__,'gpu':torch.cuda.get_device_name(),'data':data_evidence,'resolution':224,'precision':'FP32 weights; FP16 autocast','dataset_samples':len(dataset),'models':[],'complete':False}
def save():(a.outdir/('reference_eval.json' if a.reference_only else 'results.json')).write_text(json.dumps(result,indent=2)+'\n')
def extract_logits(out):
    return out[0] if isinstance(out,(tuple,list)) else out
original=dtem_module.ThreTopK
for variant in ['global','spatial_r3']:
    assert pids()==selfpids
    exp=next(e for e in experiments if e.model_id=='mn_l2' and e.variant_id==variant)
    task=common.EvalTask(exp,224);evidence=common.read_checkpoint_evidence(a.campaign,task)
    m=timm.create_model('mergenet_small_cls',**common.model_kwargs(task,protocol))
    common.load_ema_checkpoint(task,evidence,m,torch);m.cuda().eval()
    row={'variant':variant,'checkpoint':evidence,'model_kwargs':common.model_kwargs(task,protocol),'evaluations':[]};result['models'].append(row);save()
    # Same shape and same first 16 images; companions alone are replaced.
    small=torch.stack([dataset[i][0] for i in range(112)]).cuda()
    pairs=[small[:64],torch.cat([small[:16],small[64:112]])]
    row['fixed_first16_companion_probe']={}
    for selector in ([] if a.reference_only else ['historical','rowwise_bisection_reference']):
        replacement=original if selector=='historical' else thretopk_bisection_reference
        dtem_module.ThreTopK=model_module.ThreTopK=replacement
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            u=extract_logits(m(pairs[0])).float().cpu();v=extract_logits(m(pairs[1])).float().cpu();repeat=extract_logits(m(pairs[0])).float().cpu()
        row['fixed_first16_companion_probe'][selector]={'max_abs_logit_difference':(u[:16]-v[:16]).abs().max().item(),'mean_abs_logit_difference':(u[:16]-v[:16]).abs().mean().item(),'changed_top1':int((u[:16].argmax(-1)!=v[:16].argmax(-1)).sum()),'repeat_max_abs':(u-repeat).abs().max().item(),'finite':bool(torch.isfinite(u).all() and torch.isfinite(v).all())}
        save()
    dtem_module.ThreTopK=model_module.ThreTopK=original
    del small,pairs
    reference=torch.from_numpy(np.load(a.outdir/f'{variant}_sequential_b200.npz')['logits']) if a.reference_only else None
    perm=np.random.default_rng(20260915).permutation(len(dataset))
    jobs=[('rowwise_b200',200,np.arange(len(dataset)))] if a.reference_only else [('sequential_b200',200,np.arange(len(dataset))),('sequential_b64',64,np.arange(len(dataset))),('permuted_b200',200,perm)]
    if a.reference_only:dtem_module.ThreTopK=model_module.ThreTopK=thretopk_bisection_reference
    for name,batch,order in jobs:
        start=time.time();loader=DataLoader(Subset(dataset,order.tolist()),batch_size=batch,shuffle=False,num_workers=8,pin_memory=True)
        logits=torch.empty(len(dataset),100);cursor=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for x,y in loader:
                pred=extract_logits(m(x.cuda(non_blocking=True))).float().cpu()
                assert torch.isfinite(pred).all()
                inds=order[cursor:cursor+len(x)];logits[inds]=pred;cursor+=len(x)
        assert cursor==10000 and pids()==selfpids
        pred=logits.argmax(-1);correct=int((pred==labels).sum())
        if reference is None:reference=logits.clone()
        refpred=reference.argmax(-1);refok=refpred==labels;ok=pred==labels
        item={'name':name,'batch':batch,'order_seed':20260915 if name.startswith('permuted') else None,'samples':cursor,'correct':correct,'top1':correct/100.,'seconds':time.time()-start,'max_abs_logit_difference_vs_b200':(logits-reference).abs().max().item(),'changed_top1_vs_b200':int((pred!=refpred).sum()),'became_correct':int((~refok & ok).sum()),'became_wrong':int((refok & ~ok).sum())}
        np.savez_compressed(a.outdir/f'{variant}_{name}.npz',logits=logits.numpy(),labels=labels.numpy(),order=order)
        row['evaluations'].append(item);save();print(variant,item,flush=True)
    del m,reference;torch.cuda.empty_cache()
result['complete']=True;save()
