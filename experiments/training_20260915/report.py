"""Read-only aggregation of completed endpoints; partial progress stays separate."""
import pathlib,json,csv,statistics,time,argparse,os
ROOT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
def write_csv(path,rows,fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def once():
    state=json.loads((ROOT/'state.json').read_text());rows=[]
    for jobfile in sorted((ROOT/'jobs').glob('*.json')):
        job=json.loads(jobfile.read_text());directory=pathlib.Path(job['directory']);summary=directory/'summary.csv';records=[]
        if summary.exists():
            with summary.open() as f:
                records=[r for r in csv.DictReader(f) if r.get('epoch') is not None and r.get('eval_top1')]
        done=(directory/'completion.json').exists();health={}
        if (directory/'health.json').exists():health=json.loads((directory/'health.json').read_text())
        assert not done or len(records)==200
        completed_seconds=[x['epoch_total_seconds'] for x in health.get('epochs',[])[1:] if 'epoch_total_seconds' in x]
        recent=statistics.median(completed_seconds[-5:]) if completed_seconds else None
        rows.append({'job_id':job['job_id'],'selector':job['selector'],'geometry':job['geometry'],'seed':job['seed'],'status':state['jobs'].get(job['job_id'],{}).get('status','queued'),'recorded_epochs':len(records),'latest_ema_top1':records[-1]['eval_top1'] if records else '', 'best_so_far_ema_top1':max(float(r['eval_top1']) for r in records) if records else '', 'final_epoch199_ema_top1':float(records[-1]['eval_top1']) if done else '', 'median_recent_epoch_seconds':recent if recent is not None else '', 'remaining_hours_estimate':(200-len(records))*recent/3600 if recent is not None else '', 'optimizer_updates':health.get('optimizer_updates',''),'amp_scale_decreases':health.get('scale_decreases','')})
    out=ROOT/'reports';out.mkdir(exist_ok=True);write_csv(out/'progress.csv',rows,list(rows[0]))
    completed=[r for r in rows if r['final_epoch199_ema_top1']!=''];write_csv(out/'completed_endpoints.csv',completed,list(rows[0]))
    contrasts=[]
    for selector in ['historical','rowwise']:
        for baseline in ['global','flat8','degree']:
            paired=[]
            for seed in [42,43,44]:
                got={r['geometry']:r for r in completed if r['selector']==selector and r['seed']==seed}
                if 'r3' in got and baseline in got:paired.append(got['r3']['final_epoch199_ema_top1']-got[baseline]['final_epoch199_ema_top1'])
            if len(paired)==3:contrasts.append({'contrast':selector+':r3-'+baseline,'n_seeds':3,'mean_delta_pp':statistics.mean(paired),'std_delta_pp':statistics.stdev(paired),'seed42_delta':paired[0],'seed43_delta':paired[1],'seed44_delta':paired[2]})
    for geometry in ['global','flat8','r3','degree']:
        paired=[]
        for seed in [42,43,44]:
            got={r['selector']:r for r in completed if r['geometry']==geometry and r['seed']==seed}
            if len(got)==2:paired.append(got['rowwise']['final_epoch199_ema_top1']-got['historical']['final_epoch199_ema_top1'])
        if len(paired)==3:contrasts.append({'contrast':geometry+':rowwise-historical','n_seeds':3,'mean_delta_pp':statistics.mean(paired),'std_delta_pp':statistics.stdev(paired),'seed42_delta':paired[0],'seed43_delta':paired[1],'seed44_delta':paired[2]})
    write_csv(out/'three_seed_contrasts.csv',contrasts,['contrast','n_seeds','mean_delta_pp','std_delta_pp','seed42_delta','seed43_delta','seed44_delta'])
    text=['# MergeNet 八卡训练进展','',f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；阶段：{state['phase']}；完整终点：{len(completed)}/24。",'','| 配置 | seed | 状态 | epochs | 最新EMA top1 | 最近epoch秒 |','| --- | ---: | --- | ---: | ---: | ---: |']
    for row in rows:
        sec=row['median_recent_epoch_seconds'];sec=round(sec,1) if sec!='' else '—'
        text.append(f"| {row['selector']}/{row['geometry']} | {row['seed']} | {row['status']} | {row['recorded_epochs']}/200 | {row['latest_ema_top1'] or '—'} | {sec} |")
    text+=['','这里的EMA top1来自原始summary.csv。epoch199才是正式终点；早期精度不得当作最终比较。','上游saver在epoch0–49对选优分数减1000（即使lambda已等于2）；health.json的last_ema_metric和原调度器STATUS里相应值实际是这个选优分数，不是真实top1。本报告使用CSV纠正显示，不更改训练或历史数据。','最近epoch耗时排除首次编译/预热epoch；只供排期估算，不用于模型效率论文比较。','新增正式任务将持续自动排队；若任何任务失败，停止派发后续任务，保留已有日志与checkpoint。']
    (ROOT/'PROGRESS.zh-CN.md').write_text('\n'.join(text)+'\n')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),state['phase'],'completed',len(completed),'epochs',sum(r['recorded_epochs'] for r in rows),flush=True)
    return state['phase'] in ['complete','failed','smoke_failed','supervisor_failed','cancelled']
while True:
    done=once()
    if done or not a.watch:break
    time.sleep(60)
