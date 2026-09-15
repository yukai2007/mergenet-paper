#!/usr/bin/env python3
"""Recompute manuscript tables/figures from the curated campaign records (CPU)."""
from pathlib import Path
import csv, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIG=ROOT/'figures'; TAB=ROOT/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
C={'dense':'#356080','mn':'#CB6B35','tome':'#238C7A','pitome':'#8270A5','global':'#8D8D8D','flat':'#BDA35E'}
protocol=json.loads((DATA/'launch_protocol.json').read_text())
runs=[];curves={}
for name,p in protocol.items():
 f=DATA/'runs'/name/'summary.csv'
 if not f.exists():continue
 d=pd.read_csv(f); assert d.epoch.tolist()==list(range(len(d))),name
 assert np.isfinite(d.eval_top1).all(),name
 ix=d.eval_top1.idxmax();b=d.loc[ix];last=d.iloc[-1]
 assert abs(float(p['best_top1'])-b.eval_top1)<1e-8,name
 assert abs(float(p['final_top1'])-last.eval_top1)<1e-8,name
 q=dict(p);q.update(best_top1=float(b.eval_top1),final_top1=float(last.eval_top1),best_epoch=int(b.epoch),epochs_done=len(d));runs.append(q);curves[name]=d
run=pd.DataFrame(runs).set_index('run');run.to_csv(DATA/'results.csv')
def val(n,k='best_top1'):return float(run.loc[n,k])
def table(name,cols,header,rows):
 lines=['\\begin{tabular}{'+cols+'}','\\toprule',header+r' \\',r'\midrule']
 for r in rows:lines.append(' & '.join(str(s) for s in r)+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}']
 (TAB/(name+'.tex')).write_text('\n'.join(lines)+'\n')
main=[('DeiT-S/16',224,'300','s3_deit_p16_300e'),('DeiT-S/8',224,'300','s3_deit_p8_300e'),('MergeNet, $R=3$',224,'300','s3_mn_r3_lr75_300e'),('DeiT-S/8',384,'300+30','s3_deit_p8_ft384_30e'),('MergeNet, $R=3$',384,'300+30','s3_mn_r3_ft384_30e'),('MergeNet, $R=5.143$',384,'300+30','s3_mn_r5_ft384_30e')]
table('main','llrrrr','Model & Pixels & Epochs & Best (\\%) & Final (\\%) & Best epoch',[[a,b,c,f'{val(n):.3f}',f'{val(n,"final_top1"):.3f}',int(run.loc[n,'best_epoch'])] for a,b,c,n in main])
geom=[('Global','$5.0$','s2_mn_global_150e'),('Flat window 8','$5.0$','s2_mn_flat_w8_150e'),('Spatial $R=2$','$5.0$','s2_mn_r2_150e'),('Spatial $R=3$','$5.0$','s2_mn_r3_150e'),('Spatial $R=3$','$7.5$','s2_mn_r3_lr75_150e'),('Spatial $R=3$, warmup 20','$7.5$','s4_mn_warm20_150e')]
table('geometry','lrrr','Routing / probe & LR ($10^{-4}$) & Best (\\%) & Final (\\%)',[[a,b,f'{val(n):.3f}',f'{val(n,"final_top1"):.3f}'] for a,b,n in geom])
# Recompute the sweep from latest successful full-validation records, never max score.
records=[json.loads(l) for l in (DATA/'testtime_records.jsonl').read_text().splitlines()]
latest={}
for r in sorted(records,key=lambda r:r['ts']):
 z=r['result'];assert z['n_images']==50000 and z['is_final']
 assert not z['load'].get('missing') and not z['load'].get('skipped_shape')
 latest[r['key']]=r
sweep=[]
for r in latest.values():
 c=r['cell'];z=r['result']
 sweep.append({'key':r['key'],'resolution':c['resolution'],'ft384':r['key'].endswith('|ft384'),'model':c['model'],'method':c['method'],'patch_tokens':z['achieved_tokens'],'top1':z['top1']})
sweep=pd.DataFrame(sweep);sweep.to_csv(DATA/'sweep_latest.csv',index=False)
matched=pd.read_csv(DATA/'testtime_accuracy_curves.csv')
for _,r in matched.iterrows():
 for meth,col in [('mergenet','mergenet_top1'),('tome','tome_top1'),('pitome','pitome_top1')]:
  s=sweep[(sweep.resolution==r.resolution)&(sweep.ft384==(r.checkpoint_set=='ft384'))&(sweep.method==meth)&(sweep.patch_tokens==r.final_patch_tokens)]
  assert len(s)==1 and abs(float(s.iloc[0].top1)-r[col])<1e-8,(r.to_dict(),meth,s.to_dict())
assert len(matched)==20 and (matched.tome_minus_mergenet_pp>0).all()
table('sweep_main','lrrrrr','Checkpoint / pixels & Patches & MergeNet & ToMe & PiToMe & $\\Delta$ ToMe',[[('224 trained / 224' if r.resolution==224 else '384 tuned / 384'),int(r.final_patch_tokens),f'{r.mergenet_top1:.3f}',f'{r.tome_top1:.3f}',f'{r.pitome_top1:.3f}',f'{r.tome_minus_mergenet_pp:+.3f}'] for _,r in matched[(matched.resolution==224)|(matched.checkpoint_set=='ft384')].iterrows()])
# Efficiency tables keep timings separate from accuracy and use exact raw seconds.
eff=pd.read_csv(DATA/'publishable_efficiency.csv');e=eff[eff.source.str.endswith('bench_clean_2026-09-05.json')]
erows=[]
for n,lab in [('deit_p16','DeiT-S/16'),('deit_p8','DeiT-S/8'),('mergenet','MergeNet, $R=3$')]:
 t=e[(e.model==n)&(e['mode']=='train_fwd_bwd')].iloc[0];i=e[(e.model==n)&(e['mode']=='inference')].iloc[0]
 erows.append([lab,f'{i.parameters_all/1e6:.2f}',f'{t.median_seconds*1000:.2f}',f'{i.median_seconds*1000:.2f}',f'{i.iqr_seconds*1000:.3f}',f'{t.peak_allocated_gib:.3f}',f'{i.peak_allocated_gib:.3f}'])
table('efficiency224','lrrrrrr','Model & Params (M) & Train (ms) & Infer (ms) & IQR (ms) & Train GiB & Infer GiB',erows)
lat=pd.read_csv(DATA/'matched_latency_caveated.csv')
t=lat[lat.method=='tome'].reset_index(drop=True);m=lat[lat.method=='mergenet'].reset_index(drop=True)
table('latency384','rrrrrr','ToMe patches & MN patches & ToMe ms & MN ms & ToMe GiB & MN GiB',[[int(a.actual_patch_tokens),int(b.actual_patch_tokens),f'{a.median_seconds*1000:.2f}',f'{b.median_seconds*1000:.2f}',f'{a.peak_reserved_gib:.3f}',f'{b.peak_reserved_gib:.3f}'] for (_,a),(_,b) in zip(t.iterrows(),m.iterrows())])
# Main learning curves.
fig,axes=plt.subplots(1,2,figsize=(6.5,2.8))
for ax,items in zip(axes,[[('s3_deit_p8_300e','DeiT-S/8',C['dense']),('s3_mn_r3_lr75_300e','MergeNet R3',C['mn'])],[('s3_deit_p8_ft384_30e','DeiT-S/8',C['dense']),('s3_mn_r3_ft384_30e','MergeNet R3',C['mn']),('s3_mn_r5_ft384_30e','MergeNet R5.143','#B69A52')]]):
 for name,label,col in items:
  d=curves[name];ax.plot(d.epoch,d.eval_top1,label=label,color=col,lw=1.6)
 ax.set(xlabel='Epoch index',ylabel='EMA top-1 (%)');ax.grid(alpha=.17);ax.legend(frameon=False,fontsize=8)
axes[0].set(title='224 px: 300-epoch training',ylim=(50,83),xlim=(40,299));axes[1].set(title='384 px: 30-epoch fine-tuning',ylim=(81.3,83.15),xlim=(0,29))
fig.tight_layout();fig.savefig(FIG/'learning_curves.pdf');plt.close(fig)
# Accuracy-only curves. No latency axis: harness configs differ.
fig,axes=plt.subplots(1,2,figsize=(6.5,2.8))
for ax,res,ft in zip(axes,[224,384],[False,True]):
 for meth,lab,col in [('mergenet','MergeNet',C['mn']),('tome','DeiT-S/8 + ToMe',C['tome']),('pitome','DeiT-S/8 + PiToMe',C['pitome'])]:
  d=sweep[(sweep.resolution==res)&(sweep.ft384==ft)&(sweep.method==meth)].sort_values('patch_tokens')
  ax.plot(d.patch_tokens,d.top1,'o-',label=lab,color=col,ms=4)
 ax.set(xlabel='Final patch tokens',ylabel='Top-1 (%)',title=('224 px, trained at 224' if not ft else '384 px, fine-tuned (MN R5.143)'));ax.grid(alpha=.17);ax.legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(FIG/'accuracy_sweep.pdf');plt.close(fig)
# Architecture diagram sized for the manuscript's actual column width.
fig,ax=plt.subplots(figsize=(6.5,3.0));ax.set(xlim=(0,6.5),ylim=(0,3.0));ax.axis('off')
boxes=[(.1,2.05,1.5,'Patch-8 stem\n785 tokens',C['dense']),(2.02,2.05,1.9,'6 local blocks\n785 tokens each',C['dense']),(4.38,2.05,2.0,'6 soft routing steps\n785 slots retained',C['dense']),(4.38,.60,2.0,'Hard top-k gather\n785 → 393 tokens',C['mn']),(2.02,.60,1.9,'Global recovery\n393 queries, 785 K/V',C['mn']),(.1,.60,1.5,'6 latent blocks\n393 tokens + head',C['mn'])]
for x,y,w,lab,col in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,.68,boxstyle='round,pad=0.035',facecolor=col,edgecolor=col));ax.text(x+w/2,y+.34,lab,ha='center',va='center',color='white',fontsize=8.5)
def arrow(a,b,col='#38434C'):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=col,lw=1.3))
arrow((1.65,2.39),(1.95,2.39));arrow((3.97,2.39),(4.31,2.39))
arrow((5.38,2.00),(5.38,1.33));arrow((4.31,.94),(3.97,.94));arrow((1.95,.94),(1.65,.94))
arrow((2.97,2.00),(2.97,1.33),C['dense'])
ax.text(2.82,1.68,'Final local features\nas recovery K/V',ha='right',va='center',fontsize=8,color=C['dense'])
ax.text(5.22,1.67,'Original-grid\nR3 support',ha='right',va='center',fontsize=8,color=C['dense'])
ax.text(3.25,.20,'Full slots (blue) → physical gather → compressed tokens (orange)',ha='center',fontsize=8)
fig.savefig(FIG/'architecture.pdf');plt.close(fig)
# Token trajectory and MAC decomposition are code/analytic quantities.
traj=pd.read_csv(DATA/'token_trajectories.csv');fig,axes=plt.subplots(1,2,figsize=(6.5,2.8))
for model,col in [('ToMe',C['tome']),('MergeNet',C['mn'])]:
 d=traj[(traj.resolution==224)&(traj.checkpoint_set=='e300_224')&(traj.final_patch_tokens==392)&(traj.model==model)]
 if len(d):axes[0].step(d.layer,d.attention_input_patches,where='mid',color=col,label=model,lw=1.7)
axes[0].set(xlabel='Transformer block',ylabel='Attention-input patches',xticks=[1,3,6,9,12],title='Layerwise token trajectories');axes[0].legend(frameon=False);axes[0].grid(alpha=.17)
a=json.loads((DATA/'analytic_cost.json').read_text())['analytic_MAC_budget'];dm=a['deit_s8'];mm=a['mergenet']
cats=[('Projection / MLP',C['dense']),('Attention matmuls',C['tome']),('Routing / recovery',C['mn']),('Stem / head','#999999')]
dense=[sum(dm[k] for k in ['attn_qkv','attn_proj','mlp']),dm['attn_matmul'],0,dm['patch_embed']+dm['head']]
merge=[sum(v for k,v in mm.items() if ('attn_qkv' in k or 'attn_proj' in k or 'mlp' in k)),sum(v for k,v in mm.items() if ('matmul' in k and not k.startswith('xattn'))),sum(v for k,v in mm.items() if k.startswith(('dtem','xattn'))),mm['patch_embed']+mm['head']]
assert abs(sum(merge)-a['mergenet_total_MACs'])<1
base=np.zeros(2)
for i,(name,col) in enumerate(cats):
 v=np.array([dense[i],merge[i]])/1e9;axes[1].bar(['DeiT-S/8','MergeNet'],v,bottom=base,label=name,color=col,width=.55);base+=v
axes[1].set(ylabel='Estimated GMAC / image',title='224 px arithmetic estimate',ylim=(0,29));axes[1].legend(frameon=False,fontsize=7.5,loc='upper right')
fig.tight_layout();fig.savefig(FIG/'cost_and_tokens.pdf');plt.close(fig)
# Complete source-index table (compact run IDs are preserved in data/results.csv).
allrows=[]
for n,p in protocol.items():
 status={'COMPLETE':'Complete','PARTIAL_SNAPSHOT':'Partial','NO_COMPLETED_EPOCH':'No epoch'}[p['status']]
 allrows.append([r'\texttt{'+n.replace('_',r'\_')+'}',str(p['epochs_done'])+'/'+str(p['epochs_target']),status,(f'{val(n):.3f}' if n in run.index else '--')])
table('run_inventory','lrlr','Run ID & Epochs & Status & Best top-1',allrows)
checks={'summary_rows':sum(len(v) for v in curves.values()),'runs_with_metrics':len(curves),'complete_runs':sum(p['status']=='COMPLETE' for p in protocol.values()),'valid_compression_triplets':len(matched),'tome_accuracy_wins':int((matched.tome_minus_mergenet_pp>0).sum()),'gaps_pp':{'r3_minus_global':val('s2_mn_r3_150e')-val('s2_mn_global_150e'),'r3_minus_flat':val('s2_mn_r3_150e')-val('s2_mn_flat_w8_150e'),'mn_minus_dense_224':val('s3_mn_r3_lr75_300e')-val('s3_deit_p8_300e'),'mn_minus_dense_384':val('s3_mn_r3_ft384_30e')-val('s3_deit_p8_ft384_30e')},'warning':'No accuracy-latency Pareto curve is generated: proportional-attention settings differ.'}
(ROOT/'research/numerical_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
