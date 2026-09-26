#!/usr/bin/env python3
"""Build the submission's completed-result tables from local source records."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
TAB = ROOT / 'tables'
PROBE = ROOT / 'experiments/local_gpu_probe_20260925_full_synthetic_gpu2'


def table(name, columns, header, rows):
    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule', header + r' \\', r'\midrule']
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def best(run, epochs):
    data = pd.read_csv(DATA / 'runs' / run / 'summary.csv')
    assert len(data) == epochs and data.epoch.tolist() == list(range(epochs)), run
    return float(data.eval_top1.max())


def main():
    curves = pd.read_csv(DATA / 'testtime_accuracy_curves.csv')
    assert len(curves) == 20 and (curves.n_images == 50000).all()
    records = [json.loads(line) for line in (DATA / 'testtime_records.jsonl').read_text().splitlines()]
    latest = {record['key']: record for record in sorted(records, key=lambda r: r['ts'])}
    for row in curves.itertuples():
        for method, field in [('mergenet', 'mergenet_top1'), ('tome', 'tome_top1'), ('pitome', 'pitome_top1')]:
            matches = [r for r in latest.values() if r['cell']['resolution'] == row.resolution
                       and r['cell']['method'] == method
                       and r['cell'].get('ckpt_set', 'e300_224') == row.checkpoint_set
                       and r['result']['achieved_tokens'] == row.final_patch_tokens]
            assert len(matches) == 1, (row.resolution, row.checkpoint_set, method, row.final_patch_tokens)
            result = matches[0]['result']
            assert result['is_final'] and result['n_images'] == 50000
            assert abs(result['top1'] - getattr(row, field)) < 1e-8
    default = curves[(curves.resolution == 224) & (curves.final_patch_tokens == 392)].iloc[0]
    table('main', 'llrr', r'Model & Evaluation & Final patches & Top-1 (\%)', [
        ['DeiT-S/16', 'dense', 196, f'{best("s3_deit_p16_300e", 300):.3f}'],
        ['DeiT-S/8', 'dense', 784, f'{best("s3_deit_p8_300e", 300):.3f}'],
        ['DeiT-S/8 + ToMe', 'post-training', 392, f'{default.tome_top1:.3f}'],
        ['DeiT-S/8 + PiToMe', 'post-training', 392, f'{default.pitome_top1:.3f}'],
        [r'\mn{} ($R=3$)', 'trained bottleneck', 392, f'{default.mergenet_top1:.3f}'],
    ])
    table('training_budget', 'lrrr', r'Model & 150 epochs & 300 epochs & $\Delta$ (pp)', [
        [name, f'{best(short, 150):.3f}', f'{best(long, 300):.3f}', f'{best(long, 300)-best(short, 150):+.3f}']
        for name, short, long in [
            ('DeiT-S/16', 's2_deit_p16_150e', 's3_deit_p16_300e'),
            ('DeiT-S/8', 's2_deit_p8_150e', 's3_deit_p8_300e'),
            (r'\mn{} ($R=3$)', 's2_mn_r3_lr75_150e', 's3_mn_r3_lr75_300e'),
        ]
    ])
    resolution_rows = []
    for pixels, schedule, radius, dense, mn, epochs in [
        (224, '300', '3', 's3_deit_p8_300e', 's3_mn_r3_lr75_300e', 300),
        (384, '300+30', '3', 's3_deit_p8_ft384_30e', 's3_mn_r3_ft384_30e', 30),
        (384, '300+30', '5.143', 's3_deit_p8_ft384_30e', 's3_mn_r5_ft384_30e', 30),
        (512, '300+30+15', '6.857', 's4_deit_ft512_15e', 's4_mn_ft512_15e', 15),
    ]:
        d, m = best(dense, epochs), best(mn, epochs)
        resolution_rows.append([pixels, schedule, radius, f'{d:.3f}', f'{m:.3f}', f'{m-d:+.3f}'])
    table('resolution', 'llrrrr', r'Pixels & Epochs & MN radius & DeiT-S/8 & MergeNet & Gap (pp)', resolution_rows)
    transfer = curves[curves.transfer_mode == 'zero_shot_res_transfer']
    assert len(transfer) == 10
    table('resolution_transfer', 'rrrrr', r'Pixels & Patches & MergeNet & ToMe & PiToMe', [
        [int(row.resolution), int(row.final_patch_tokens), f'{row.mergenet_top1:.3f}',
         f'{row.tome_top1:.3f}', f'{row.pitome_top1:.3f}']
        for row in transfer.itertuples()
    ])
    eager = pd.read_csv(PROBE / 'summary_eager.csv').set_index('method')
    compiled = pd.read_csv(PROBE / 'summary_compiled.csv').set_index('method')
    rerun = json.loads((PROBE / 'mergenet_b1_rerun.json').read_text())['latency_median_ms']
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Method & Patches & B1 (ms) & B64 (ms) & B64 (im/s) & Alloc. & Reserv. \\',
             r'\midrule']
    for mode, data, names in [('Eager', eager, ['dense', 'tome', 'pitome', 'mergenet']),
                              ('Compiled', compiled, ['tome', 'mergenet'])]:
        if mode == 'Compiled': lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{l}{\textit{' + mode + r'}} \\')
        for name in names:
            row = data.loc[name]
            assert int(row.patches) == (784 if name == 'dense' else 392)
            label = {'dense':'Dense DeiT-S/8', 'tome':'ToMe', 'pitome':'PiToMe', 'mergenet':r'\mn{} ($R=3$)'}[name]
            b1 = (f'{min(row.b1_median_ms,rerun):.2f}--{max(row.b1_median_ms,rerun):.2f}'
                  if mode == 'Eager' and name == 'mergenet' else f'{row.b1_median_ms:.2f}')
            values = [label, str(int(row.patches)), b1, f'{row.b64_median_ms:.2f}',
                      f'{row.b64_throughput_images_s:.1f}', f'{row.b64_peak_allocated_gib:.3f}',
                      f'{row.b64_peak_reserved_gib:.3f}']
            lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (TAB / 'followup_latency.tex').write_text('\n'.join(lines)+'\n')
    h20 = pd.read_csv(DATA / 'publishable_efficiency.csv')
    rows=[]
    for batch in [64,128]:
        source = 'bench_clean_384_b64_2026-09-13' if batch == 64 else 'bench_clean_384_2026-09-13'
        for model,label in [('deit_p8','Dense DeiT-S/8'),('mergenet',r'\mn{} ($R=3$)')]:
            subset=h20[h20.source.str.contains(source)&(h20.model==model)&(h20.batch==batch)]
            assert len(subset)==2 and subset.isolation_clean.all()
            train=subset[subset['mode']=='train_fwd_bwd'].iloc[0]
            infer=subset[subset['mode']=='inference'].iloc[0]
            rows.append([batch,label,f'{1000*train.median_seconds:.2f}',f'{1000*infer.median_seconds:.2f}',
                         f'{train.peak_allocated_gib:.3f}',f'{infer.peak_allocated_gib:.3f}'])
    table('highres_efficiency','rlrrrr',r'B & Model & Train (ms) & Infer (ms) & Train GiB & Infer GiB',rows)
    print('Verified 60 full-validation accuracy values; rebuilt six completed-result tables.')


if __name__ == '__main__':
    main()
