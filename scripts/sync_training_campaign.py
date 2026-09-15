"""Archive bounded live-campaign source and clearly labeled partial progress.

Never launches training, copies weights, or changes manuscript result tables.
"""
from pathlib import Path
import json,hashlib,shutil,csv,datetime
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/liziqing/yukai/mergenet_local_campaign_20260915')
DEST=ROOT/'experiments/training_20260915'
DEST.mkdir(exist_ok=True)
def copy(rel):
    src=SOURCE/rel;dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
for name in ['protocol.json','runtime_manifest.json','code_manifest.json','environment.json','smoke_gate.json','README.zh-CN.md','report.py']:
    if (SOURCE/name).exists():copy(Path(name))
for directory in ['code','jobs','smoke_jobs','validation','runtime/imagenet_longtrain_v1']:
    for p in (SOURCE/directory).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p.relative_to(SOURCE))
# Static runtime/script hashes must match the actual campaign freeze.
for name,digest in json.loads((DEST/'code_manifest.json').read_text()).items():
    assert hashlib.sha256((DEST/'code'/name).read_bytes()).hexdigest()==digest,name
for name,digest in json.loads((DEST/'runtime_manifest.json').read_text())['files'].items():
    assert hashlib.sha256((DEST/'runtime/imagenet_longtrain_v1'/name).read_bytes()).hexdigest()==digest,name
state=json.loads((SOURCE/'state.json').read_text())
rows=list(csv.DictReader((SOURCE/'reports/progress.csv').open()))
smoke=[]
for path in sorted((SOURCE/'smoke').glob('*/health.json')):
    h=json.loads(path.read_text());e=h['epochs'][0]
    smoke.append({'job_id':h['job_id'],'complete':h['complete'],'training_steps':h['steps'],'optimizer_updates':h['optimizer_updates'],'amp_scale_decreases':h['scale_decreases'],'nonfinite_loss':h['nonfinite_loss'],'final_loss_scale':h['loss_scale'],'train_seconds':e['train_seconds'],'full_smoke_seconds':e['epoch_total_seconds'],'checkpoint_selection_metric':h['last_ema_metric'],'actual_ema_top1':float(list(csv.DictReader((path.parent/'summary.csv').open()))[-1]['eval_top1']),'health_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
snapshot={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'live_campaign':str(SOURCE),'phase':state['phase'],'complete':state['complete'],'planned_runs':24,'completed_runs':sum(r['status']=='complete' for r in rows),'smoke':smoke,'progress':rows,'errors':state['errors'],'warning':'Partial new CIFAR training only; no new formal endpoint until200epochs; no companyImageNet weights; immutable source archive, do not launch jobs here because saved job directories point at the active campaign.'}
(DEST/'status_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(DEST/'README.md').write_text('''# New local training campaign (in progress)

Source/evidence snapshot only. The active campaign is `/liziqing/yukai/mergenet_local_campaign_20260915`; read its `PROGRESS.zh-CN.md` for current progress. `status_snapshot.json` is timestamped and may be stale. Original run directories must not be relaunched or overwritten.

The frozen matrix is historical/rowwise selector ×global/flat8/R3/receiver-degree-preserving permutation ×seeds42/43/44, each200epochs on CIFAR-100 resized to224. Eight independent GPU jobs share the same batch200 and training recipe. Formal endpoints and seed statistics remain pending until the relevant runs finish.

`code/` and `runtime/` preserve exactly the executable source identities from the campaign. Runtime licensing and upstream headers remain intact. `jobs/` and `smoke_jobs/` record resolved commands; their output paths refer to the live campaign, so do not run them from this archive. `report.py` demonstrates the aggregation logic but reads a live campaign's state/logs. `validation/` and `smoke_gate.json` contain completed preflight evidence. Large weights and full machine logs are not copied.

`README.zh-CN.md` is the original campaign guide; its live relative progress links refer to the campaign directory, not this static archive. To refresh the source/progress snapshot only, run `python scripts/sync_training_campaign.py` from the paper project. It never launches training or changes paper result tables.
''')
print('Snapshot:',snapshot['phase'],snapshot['completed_runs'],'of24 complete;',len(smoke),'smoke passes')
