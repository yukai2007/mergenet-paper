"""Run the frozen trainer with explicitly recorded selector/support interventions."""
import argparse,os,sys,pathlib,json,hashlib,time,math,importlib.util,itertools
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=pathlib.Path);p.add_argument('--smoke-steps',type=int,default=0)
a=p.parse_args();job=json.loads(a.job.read_text());out=pathlib.Path(job['directory']);out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('OPENTOME_SKIP_OPTIONAL_NLP','1');os.environ['OPENTOME_MERGENET_IMPL']='new';os.environ['TIMM_FUSED_ATTN']='1'
sys.path[:0]=[str(ROOT/'runtime/imagenet_longtrain_v1'),'/liziqing/yukai/.deps_mergenet_resize20260810']
import torch
from interventions import install
torch.set_num_threads(4)
intervention=install(job['selector'],job['geometry'])
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
health={'job_id':job['job_id'],'started':time.time(),'smoke_steps':a.smoke_steps,'intervention':intervention,'epochs':[],'steps':0,'optimizer_updates':0,'scale_decreases':0,'nonfinite_loss':0,'complete':False,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
atomic(out/'health.json',health)
trainer_path=ROOT/'runtime/imagenet_longtrain_v1/trainer/classification/in1k_trainer.py'
spec=importlib.util.spec_from_file_location('campaign_frozen_trainer',trainer_path);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
OriginalScaler=trainer.NativeScaler
class ObservedScaler(OriginalScaler):
    def __call__(self,loss,optimizer,**kw):
        if not bool(torch.isfinite(loss)):
            health['nonfinite_loss']+=1;atomic(out/'health.json',health);raise FloatingPointError('Nonfinite training loss')
        before=float(self._scaler.get_scale());n=health['optimizer_updates']
        super().__call__(loss,optimizer,**kw)
        after=float(self._scaler.get_scale());health['steps']+=1
        health['scale_decreases']+=int(after<before);health['last_loss']=float(loss.detach());health['loss_scale']=after;health['last_step_time']=time.time()
        if health['steps']==25 or health['steps']%250==0:atomic(out/'health.json',health)
trainer.NativeScaler=ObservedScaler
original_train=trainer.train_one_epoch
class LimitedLoader:
    def __init__(self,loader,n):self.loader=loader;self.n=n
    def __len__(self):return min(len(self.loader),self.n)
    def __iter__(self):return itertools.islice(iter(self.loader),len(self))
    def __getattr__(self,name):return getattr(self.loader,name)
    @property
    def mixup_enabled(self):return self.loader.mixup_enabled
    @mixup_enabled.setter
    def mixup_enabled(self,value):self.loader.mixup_enabled=value

def counted_train(epoch,model,loader,optimizer,*args,**kw):
    if not getattr(optimizer,'_campaign_counted',False):
        def update(opt,arguments,keywords):health['optimizer_updates']+=1
        optimizer.register_step_post_hook(update);optimizer._campaign_counted=True
    start=time.time();health['epoch_started']=start;prev=health['optimizer_updates'];skipped=health['scale_decreases']
    if a.smoke_steps:loader=LimitedLoader(loader,a.smoke_steps)
    value=original_train(epoch,model,loader,optimizer,*args,**kw)
    metric=value[0];assert all(math.isfinite(float(v)) for v in metric.values() if isinstance(v,(int,float))),metric
    completed=health['optimizer_updates']-prev
    assert completed>0,'No actual optimizer updates'
    if epoch>0:assert completed>=len(loader)*.8,'More than20% optimizer updates skipped'
    health['epochs'].append({'epoch':epoch,'train_seconds':time.time()-start,'batches':len(loader),'optimizer_updates':completed,'scale_decreases':health['scale_decreases']-skipped,'metrics':dict(metric)})
    atomic(out/'health.json',health);return value
trainer.train_one_epoch=counted_train
original_save=trainer.CheckpointSaver.save_checkpoint
def observed_save(self,epoch,metric=None):
    if metric is not None:assert math.isfinite(float(metric)),metric
    value=original_save(self,epoch,metric)
    health['last_saved_epoch']=epoch;health['last_ema_metric']=metric;health['last_save_time']=time.time()
    health['epochs'][-1]['epoch_total_seconds']=time.time()-health['epoch_started']
    atomic(out/'health.json',health)
    if a.smoke_steps:
        assert health['optimizer_updates']>=a.smoke_steps//2
        health['complete']=True;health['finished']=time.time();atomic(out/'health.json',health);raise SystemExit(0)
    return value
trainer.CheckpointSaver.save_checkpoint=observed_save
# Extra intervention tags are not understood by the upstream parser: retain
# exact training args and bind these to the job manifest and checkpoint sidecar.
sys.argv=[str(trainer_path)]+job['trainer_args']
try:
    trainer.main()
    health['complete']=health.get('last_saved_epoch')==199
    assert health['complete'],'Trainer returned without complete200-epoch schedule'
    health['finished']=time.time();atomic(out/'health.json',health)
except BaseException as exc:
    if not (isinstance(exc,SystemExit) and exc.code==0 and health['complete']):
        health['error']=repr(exc);health['failed']=time.time();atomic(out/'health.json',health)
    raise
