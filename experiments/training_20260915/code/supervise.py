"""Durable single-host8GPU scheduler: all-smoke gate, then24 full jobs.

No automatic failed-job retries. Never adopts or terminates foreign processes.
A STOP file stops new dispatch; CANCEL additionally terminates our live jobs.
"""
import os,sys,pathlib,json,subprocess,time,signal,fcntl,csv,math,hashlib,statistics,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.write(str(os.getpid()));lock.flush()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def read(path):return json.loads(path.read_text())
def gpu_info():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines()
    return {int(x.split(',')[0]):{'uuid':x.split(',')[1].strip(),'name':x.split(',')[2].strip()} for x in lines}
def apps():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    by={}
    for line in out.splitlines():
        if ',' in line:uid,pid=line.split(',')[:2];by.setdefault(uid.strip(),set()).add(int(pid))
    return by
G=gpu_info();assert all(i in G for i in range(8))
assert all(not apps().get(G[i]['uuid']) for i in range(8)),'An assigned GPU is busy'
assert read(ROOT/'validation/numerical.json')['complete'],'Numerical gate incomplete'
# Verify runtime and freeze this exact wrapper/intervention code before launch.
for name,digest in read(ROOT/'runtime_manifest.json')['files'].items():assert hashlib.sha256((ROOT/'runtime/imagenet_longtrain_v1'/name).read_bytes()).hexdigest()==digest,name
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').glob('*.py')}
atomic(ROOT/'code_manifest.json',hashes)
state={'started':time.time(),'pid':os.getpid(),'phase':'smoke','jobs':{},'gpus':G,'complete':False,'errors':[],'code_hashes':hashes}
active={};logs={}
configs=[(s,g) for s in ['historical','rowwise'] for g in ['global','flat8','r3','degree']]
full=[ROOT/'jobs'/f'{s}_{g}_s{seed}.json' for seed in [42,43,44] for s,g in configs]
smoke=[ROOT/'smoke_jobs'/f'{s}_{g}_s42.json' for s,g in configs]
queue=smoke.copy()
def publish():
    atomic(ROOT/'state.json',state)
    lines=['# MergeNet 八卡本地实验状态','',f"阶段：{state['phase']}；更新：{time.strftime('%Y-%m-%d %H:%M:%S')}；数值检查：通过。",'', '| Job | GPU | 状态 | 完成epoch | EMA top1 | 最近epoch秒 |','| --- | ---: | --- | ---: | ---: | ---: |']
    for key,row in state['jobs'].items():
        path=pathlib.Path(row['directory'])/'health.json'
        if path.exists():
            try:
                h=read(path);row['last_saved_epoch']=h.get('last_saved_epoch',-1);row['optimizer_updates']=h.get('optimizer_updates',0);row['last_ema_metric']=h.get('last_ema_metric');row['last_step_time']=h.get('last_step_time');row['epoch_seconds']=[e['epoch_total_seconds'] for e in h['epochs'] if 'epoch_total_seconds' in e]
            except (ValueError,KeyError):pass
        metric=row.get('last_ema_metric');secs=row.get('epoch_seconds',[])
        lines.append(f"| {key} | {row['gpu']} | {row['status']} | {row.get('last_saved_epoch',-1)+1} | {metric if metric is not None else '—'} | {round(secs[-1],1) if secs else '—'} |")
    lines+=['','正式矩阵：8配置 × seeds42/43/44 ×200epochs；主终点为epoch199 EMA。短跑结果与正式结果分开。','没有公司ImageNet权重；这些是独立CIFAR训练。', '', '停止新任务：在campaign根目录创建STOP；终止本调度器的训练：创建CANCEL。不要删除原始run。']
    (ROOT/'STATUS.zh-CN.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'state.json',state)
def launch(path,gpu,smoke_run):
    job=read(path);out=pathlib.Path(job['directory']);assert not (out/'health.json').exists(),f'Existing run requires explicit review: {out}'
    out.mkdir(parents=True,exist_ok=True);log=(out/'train.log').open('w')
    env=os.environ.copy()
    for key in ['WORLD_SIZE','RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=G[gpu]['uuid'],PYTHONPATH=f"{ROOT/'runtime/imagenet_longtrain_v1'}:/liziqing/yukai/.deps_mergenet_resize20260810",PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENTOME_SKIP_OPTIONAL_NLP='1',OPENTOME_MERGENET_IMPL='new',TIMM_FUSED_ATTN='1')
    cmd=['/usr/bin/python','-S',str(ROOT/'code/train_entry.py'),'--job',str(path)]
    if smoke_run:cmd+=['--smoke-steps','24']
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    key=job['job_id'];active[gpu]=(proc,key);logs[key]=log
    row={'status':'running','gpu':gpu,'gpu_uuid':G[gpu]['uuid'],'pid':proc.pid,'directory':str(out),'started':time.time(),'command':cmd,'job_manifest':str(path),'job_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    state['jobs'][key]=row;atomic(out/'launch.json',row);print('LAUNCH',key,'GPU',gpu,'pid',proc.pid,flush=True)
def completion(key,smoke_run):
    row=state['jobs'][key];out=pathlib.Path(row['directory']);h=read(out/'health.json');assert h['complete'] and h['nonfinite_loss']==0
    rows=list(csv.DictReader((out/'summary.csv').open()));expected=1 if smoke_run else 200
    assert len(rows)==expected and [int(r['epoch']) for r in rows]==list(range(expected))
    assert all(math.isfinite(float(r['eval_top1'])) and math.isfinite(float(r['train_loss'])) for r in rows)
    ckpt=out/'last.pth.tar';assert ckpt.is_file()
    receipt={'job_id':key,'completed':time.time(),'epochs':expected,'final_ema_top1':float(rows[-1]['eval_top1']),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'checkpoint_path':str(ckpt),'summary_sha256':hashlib.sha256((out/'summary.csv').read_bytes()).hexdigest(),'code_manifest':str(ROOT/'code_manifest.json'),'smoke':smoke_run}
    atomic(out/'completion.json',receipt);row['status']='complete';row['receipt']=receipt

def cancel_owned():
    for proc,key in active.values():
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
try:
    while True:
        if (ROOT/'CANCEL').exists():cancel_owned();state['phase']='cancelled';publish();break
        for gpu,(proc,key) in list(active.items()):
            ret=proc.poll()
            if ret is not None:
                logs[key].close();state['jobs'][key]['exit_code']=ret;state['jobs'][key]['ended']=time.time()
                try:
                    assert ret==0,f'exit{ret}';completion(key,state['phase']=='smoke')
                    print('COMPLETE',key,flush=True)
                except Exception as exc:
                    state['jobs'][key]['status']='failed';state['errors'].append({'job_id':key,'error':repr(exc)});print('FAILED',key,repr(exc),flush=True)
                del active[gpu]
        if state['phase']=='smoke' and not queue and not active:
            if state['errors']:state['phase']='smoke_failed';publish();break
            state['phase']='training';queue=full.copy();atomic(ROOT/'smoke_gate.json',{'pass':True,'completed':time.time(),'jobs':[r['receipt'] for r in state['jobs'].values()]});print('ALL8 SMOKE PASS; START FULL MATRIX',flush=True)
        if not state['errors'] and not (ROOT/'STOP').exists():
            occupied=apps()
            for gpu in range(8):
                if gpu not in active and queue and not occupied.get(G[gpu]['uuid']):launch(queue.pop(0),gpu,state['phase']=='smoke')
        if not queue and not active:
            state['complete']=not state['errors'];state['phase']='complete' if state['complete'] else 'failed';state['finished']=time.time();publish();break
        if state['errors'] and not active:state['phase']='failed';publish();break
        publish();time.sleep(10)
except BaseException:
    state['errors'].append({'supervisor':traceback.format_exc()});state['phase']='supervisor_failed';cancel_owned();publish();raise
