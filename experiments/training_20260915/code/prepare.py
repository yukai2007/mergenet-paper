"""Freeze a24-job experiment contract before reading new training results."""
import pathlib,json,hashlib,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
source=pathlib.Path('/liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_dtem_spatial_20260814/runs/mn_l2/spatial_r3/r224/seed42/manifest.json')
base=json.loads(source.read_text())['command'][3:]
def setarg(args,key,value):
    if key in args:args[args.index(key)+1]=str(value)
    else:args.extend([key,str(value)])
def remove(args,key):
    if key in args:i=args.index(key);del args[i:i+2]
configs=[(s,g) for s in ['historical','rowwise'] for g in ['global','flat8','r3','degree']]
protocol={'created_date':'2026-09-15','dataset':'CIFAR100','train_samples':50000,'eval_samples':10000,'resolution':224,'patch_size':8,'local_depth':6,'latent_depth':6,'lambda':2,'epochs':200,'seeds':[42,43,44],'microbatch':200,'validation_batch':200,'accumulation':1,'configs':[{'selector':s,'geometry':g} for s,g in configs],'num_runs':24,'queue':'seed42 all8configs, then43, then44;one independent run per GPU','primary_endpoint':'epoch199 EMA top1; best EMA secondary only','primary_contrasts':['R3-global','R3-flat8','R3-degree at fixed selector','rowwise-historical at fixed geometry'],'summaries':'paired per-seed differences; mean and sample std across3seeds; report all cells including negative outcomes','scope':'new local training from random initialization; no company ImageNet weights; dense historical spatial backend','source_manifest':str(source),'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'degree_control':'permute columns of realized R3 donor/receiver mask using fixed receiver-ID priorities; preserve per-donor degree and receiver-degree multiset on every call; not independent random edges;support seed20260915 fixed across training seeds','selector':'FP32 per-row normalization +40-step Triton bisection +implicit VJP; FP64 CPU reference; historical function unchanged','excluded_changes':['No BNHD layout optimization','No ImageNet training','No new learning-rate search','No geometry-specific batch or schedule changes'],'smoke':'24 training minibatches on realCIFAR plus full raw/EMA validation; auxiliary weight0.05 fromstep1 and warmup LR0.001 as deliberate stress conditions; fresh full runs restart seed and use original200epoch schedule','stopping':'failed numerical or any smoke gate prevents all full jobs; full jobs abort nonfinite losses/metrics, zero optimizer updates, or >20% skipped updates afterepoch0; failures retained, never silent reruns','resume':'no automatic restart; completed checkpoints preserved; manual resume must be recorded as non-bitwise continuation because upstream trainer does not save complete RNG state','retention':'each run keeps last, best, and top3 checkpoint files via native hardlinks; logs and configs preserved','changes_from_source':['seed by plan','geometry selector intervention explicitly recorded','checkpoint_hist1 toreduce redundant weights','workers4 for8simultaneous jobs','log_interval50 unchanged']}
protocol['retention']='each run keeps last and best plus top1 checkpoint via native hardlinks; logs and configs preserved'
(ROOT/'jobs').mkdir(exist_ok=True);(ROOT/'smoke_jobs').mkdir(exist_ok=True)
for seed in protocol['seeds']:
 for s,g in configs:
  name=f'{s}_{g}_s{seed}';directory=ROOT/'runs'/name;args=base.copy()
  for key,val in [('--seed',seed),('--output',directory.parent),('--experiment',directory.name),('--workers',4),('--checkpoint_hist',1)]:setarg(args,key,val)
  if g in ['global','flat8']:
   remove(args,'--dtem_spatial_radius');remove(args,'--dtem_spatial_metric')
  setarg(args,'--dtem_window_size',8 if g=='flat8' else 0)
  job={'job_id':name,'selector':s,'geometry':g,'seed':seed,'directory':str(directory),'trainer_args':args}
  (ROOT/'jobs'/f'{name}.json').write_text(json.dumps(job,indent=2)+'\n')
  if seed==42:
   smoke=json.loads(json.dumps(job));smoke['job_id']='smoke_'+name;d=ROOT/'smoke'/name;smoke['directory']=str(d)
   for key,val in [('--output',d.parent),('--experiment',d.name),('--warmup_lr',.001),('--soft_topk_aux_start_epoch',0),('--soft_topk_aux_ramp_epochs',0),('--log_interval',5)]:setarg(smoke['trainer_args'],key,val)
   (ROOT/'smoke_jobs'/f'{name}.json').write_text(json.dumps(smoke,indent=2)+'\n')
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
print('Prepared24 full jobs and8 smoke jobs')
