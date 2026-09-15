#!/usr/bin/env python3
"""Check the historical CIFAR appendix against its audited endpoint records."""
from pathlib import Path
import csv, json
root=Path(__file__).resolve().parents[1]
read=lambda n:list(csv.DictReader((root/'data/cifar'/n).open()))
runs=read('all-runs.csv');pairs=read('r3-paired-results.csv');means=read('radius-summary.csv')
assert len(runs)==70 and len(pairs)==10
assert all(r['epochs']=='200' and r['endpoint']=='epoch199_ema_top1' and r['audit_pass']=='True' for r in runs)
idx={(r['model'],r['resize'],r['variant']):float(r['top1']) for r in runs}
for p in pairs:
 for field,var in [('global_top1','global'),('historical_flat_w8_top1','flat_w8_historical'),('r3_top1','spatial_r3')]:
  assert abs(float(p[field])-idx[p['model'],p['resize'],var])<1e-8
 assert abs(float(p['r3_minus_global_pp'])-(float(p['r3_top1'])-float(p['global_top1'])))<1e-8
for m in means:
 vals=[float(r['top1']) for r in runs if r['variant']==m['variant']]
 assert len(vals)==10 and abs(sum(vals)/len(vals)-float(m['mean_top1']))<1e-8
result={'runs':len(runs),'pairs':len(pairs),'r3_mean':sum(float(p['r3_top1']) for p in pairs)/10,'mean_gain_over_global':sum(float(p['r3_minus_global_pp']) for p in pairs)/10,'r3_global_wins':sum(float(p['r3_minus_global_pp'])>0 for p in pairs),'scope':'Audited endpoint-table consistency; no fresh checkpoint evaluation or raw-run replay.'}
(root/'research/cifar_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
