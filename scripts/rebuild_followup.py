"""Validate archived independent follow-up results and regenerate paper tables."""
import pathlib, json, csv, gzip, collections, statistics
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
D=ROOT/'data/followup'
def read(name):return json.loads((D/name).read_text())
whole=read('whole_model.json');assert whole['complete']
assert len(whole['timings'])==36
stats=collections.defaultdict(list)
for c in whole['timings']:
    assert c['hooks_removed_before_timing'] and len(c['samples_ms'])==30
    assert all(np.isfinite(v) and v>0 for v in c['samples_ms'])
    assert statistics.median(c['samples_ms'])==c['median_ms']
    assert c['observed_blocks'][-1][1]-1==c['patches']
    assert c['patches']==(c['resolution']//8)**2//(1 if c['variant']=='dense' else 2)
    stats[c['resolution'],c['variant']].append(c['median_ms'])
summary=[]
for (res,variant),values in stats.items():
    assert len(values)==3
    summary.append(dict(resolution=res,variant=variant,median_ms=statistics.median(values),min_round_median_ms=min(values),max_round_median_ms=max(values)))
with (D/'whole_model_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
by={(r['resolution'],r['variant']):r for r in summary}
rows=[r'\begin{tabular}{lrr}',r'\toprule',r'Model & 224 px (ms) & 384 px (ms) \\',r'\midrule']
for variant,label in [('dense','Dense DeiT-S/8'),('tome_prop_true','ToMe, proportional attention on'),('tome_prop_false','ToMe, proportional attention off'),('mn_r3','MergeNet $R=3$'),('mn_r3_bnhd','MergeNet $R=3$, BNHD'),('mn_r5','MergeNet $R=5.143$'),('mn_r5_bnhd','MergeNet $R=5.143$, BNHD')]:
    vals=[f"{by[res,variant]['median_ms']:.2f}" if (res,variant) in by else '--' for res in [224,384]]
    rows.append(label+' & '+' & '.join(vals)+r' \\')
rows.extend([r'\bottomrule',r'\end{tabular}']);(ROOT/'tables/followup_latency.tex').write_text('\n'.join(rows)+'\n')
with gzip.open(D/'cifar_predictions.csv.gz','rt') as f:preds=list(csv.DictReader(f))
assert len(preds)==10000 and [int(r['index']) for r in preds]==list(range(10000))
labels=np.array([int(r['label']) for r in preds]);results={}
for document in ['cifar_batch.json','cifar_reference.json']:
    doc=read(document);assert doc['complete']
    for model in doc['models']:
        assert model['checkpoint']['strict_state_dict_load']
        for e in model['evaluations']:
            key=model['variant']+'_'+e['name'];p=np.array([int(r[key]) for r in preds]);assert int((p==labels).sum())==e['correct']
            assert e['samples']==10000 and e['top1']==e['correct']/100
            base=np.array([int(r[model['variant']+'_sequential_b200']) for r in preds])
            assert int((p!=base).sum())==e['changed_top1_vs_b200']
            results[key]=e
rng=np.random.default_rng(20260915);paired=[]
rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Evaluation & Global (\%) & $R=3$ (\%) & Gap (pp) & Changed G/R3 \\',r'\midrule']
for name,label in [('sequential_b200','Original order, $B=200$'),('sequential_b64','Original order, $B=64$'),('permuted_b200','Permuted order, $B=200$'),('rowwise_b200','Rowwise reference, $B=200$')]:
    ka,kb='global_'+name,'spatial_r3_'+name
    aa=np.array([int(r[ka]) for r in preds])==labels;bb=np.array([int(r[kb]) for r in preds])==labels
    diff=bb.astype(int)-aa.astype(int);counts=np.bincount(diff+1,minlength=3)
    samples=rng.multinomial(10000,counts/10000,size=20000);ci=np.quantile((samples[:,2]-samples[:,0])/100,[.025,.975])
    paired.append({'evaluation':name,'gap_pp':int(diff.sum())/100,'global_only_correct':int(counts[0]),'r3_only_correct':int(counts[2]),'fixed_prediction_bootstrap95_pp':ci.tolist(),'bootstrap_seed':20260915,'resamples':20000,'scope':'conditional on these two checkpoints and fixed recorded predictions; not seed or rebatching uncertainty'})
    a,b=results[ka],results[kb]
    rows.append(f"{label} & {a['top1']:.2f} & {b['top1']:.2f} & {b['top1']-a['top1']:+.2f} & {a['changed_top1_vs_b200']}/{b['changed_top1_vs_b200']}"+r' \\')
rows.extend([r'\bottomrule',r'\end{tabular}']);(ROOT/'tables/followup_cifar.tex').write_text('\n'.join(rows)+'\n')
(D/'paired_statistics.json').write_text(json.dumps(paired,indent=2)+'\n')
for name in ['dense_parity.json','sparse_scaled_parity.json']:
    doc=read(name);assert doc['complete'] and len(doc['parity'])==2 and all(r['pass'] for r in doc['parity'])
validation={'whole_model_cells':36,'cuda_event_samples':1080,'prediction_rows':10000,'evaluations':8,'image_evaluations':80000,'fresh_local_checkpoints':2,'dense_and_loss_scaled_sparse_parity_passes':4,'unscaled_sparse_parity_failures_preserved':sum(not r['pass'] for r in whole['parity']),'status':'PASS'}
(ROOT/'research/followup_validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation))
