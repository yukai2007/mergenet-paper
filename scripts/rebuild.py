#!/usr/bin/env python3
"""Recompute manuscript tables/figures from the curated campaign records (CPU)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
})
C = {
    'dense': '#356080',
    'mn': '#CB6B35',
    'tome': '#238C7A',
    'pitome': '#8270A5',
    'global': '#8D8D8D',
    'flat': '#BDA35E',
    'degree': '#6B7B3A',
}

protocol = json.loads((DATA / 'launch_protocol.json').read_text())
runs = []
curves = {}
for name, p in protocol.items():
    f = DATA / 'runs' / name / 'summary.csv'
    if not f.exists():
        continue
    d = pd.read_csv(f)
    assert d.epoch.tolist() == list(range(len(d))), name
    assert np.isfinite(d.eval_top1).all(), name
    report = d.eval_top1.astype(float)
    if 'eval_top1_full_compression' in d.columns:
        full = d.eval_top1_full_compression.astype(float)
        if (full > 0).any():
            report = np.where(full > 0, full, report)
            d = d.copy()
            d['report_top1'] = report
            compressed = d[full > 0]
            ix = compressed.report_top1.idxmax()
            b = d.loc[ix]
            last = d.iloc[-1]
            best_top1 = float(b.report_top1)
            final_top1 = float(last.report_top1)
        else:
            ix = d.eval_top1.idxmax()
            b = d.loc[ix]
            last = d.iloc[-1]
            best_top1 = float(b.eval_top1)
            final_top1 = float(last.eval_top1)
    else:
        ix = d.eval_top1.idxmax()
        b = d.loc[ix]
        last = d.iloc[-1]
        best_top1 = float(b.eval_top1)
        final_top1 = float(last.eval_top1)
    q = dict(p)
    q.update(
        best_top1=best_top1,
        final_top1=final_top1,
        best_epoch=int(b.epoch),
        epochs_done=len(d),
    )
    runs.append(q)
    curves[name] = d

run = pd.DataFrame(runs).set_index('run')
run.to_csv(DATA / 'results.csv')


def val(n, k='best_top1'):
    return float(run.loc[n, k])


def table(name, cols, header, rows):
    lines = ['\\begin{tabular}{' + cols + '}', '\\toprule', header + r' \\', r'\midrule']
    for r in rows:
        lines.append(' & '.join(str(s) for s in r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / (name + '.tex')).write_text('\n'.join(lines) + '\n')


schedules = [
    ('DeiT-S/16', 224, '300', 's3_deit_p16_300e'),
    ('DeiT-S/8', 224, '300', 's3_deit_p8_300e'),
    ('MergeNet, $R=3$', 224, '300', 's3_mn_r3_lr75_300e'),
    ('DeiT-S/8', 384, '300+30', 's3_deit_p8_ft384_30e'),
    ('MergeNet, $R=3$', 384, '300+30', 's3_mn_r3_ft384_30e'),
    ('MergeNet, $R=5.143$', 384, '300+30', 's3_mn_r5_ft384_30e'),
    ('DeiT-S/8', 512, '300+30+15', 's4_deit_ft512_15e'),
    ('MergeNet, $R=6.857$', 512, '300+30+15', 's4_mn_ft512_15e'),
]
table(
    'schedules',
    'llrrrr',
    r'Model & Pixels & Epochs & Best (\%) & Final (\%) & Best epoch',
    [
        [a, b, c, f'{val(n):.3f}', f'{val(n, "final_top1"):.3f}', int(run.loc[n, 'best_epoch'])]
        for a, b, c, n in schedules
    ],
)

comparison = pd.read_csv(DATA / 'testtime_accuracy_curves.csv')
comparison_224 = comparison[
    (comparison.resolution == 224)
    & (comparison.checkpoint_set == 'e300_224')
    & (comparison.final_patch_tokens == 392)
].iloc[0]
table(
    'main',
    'llrr',
    r'Method & Checkpoint / path & Final patches & Top-1 (\%)',
    [
        ['DeiT-S/8', 'trained dense', '784', f'{val("s3_deit_p8_300e"):.3f}'],
        ['DeiT-S/8 + ToMe', 'post-training', '392', f'{comparison_224.tome_top1:.3f}'],
        ['DeiT-S/8 + PiToMe', 'post-training', '392', f'{comparison_224.pitome_top1:.3f}'],
        [r'\mn{} ($R=3$)', 'trained bottleneck', '392', f'{comparison_224.mergenet_top1:.3f}'],
        ['DTEM-p8', 'common-recipe adaptation', '392', r'\pendingvalue'],
    ],
)

geom = [
    ('Global', '$5.0$', 's2_mn_global_150e'),
    ('Flat window 8', '$5.0$', 's2_mn_flat_w8_150e'),
    ('Spatial $R=2$', '$5.0$', 's2_mn_r2_150e'),
    ('Spatial $R=3$', '$5.0$', 's2_mn_r3_150e'),
]
table(
    'geometry',
    'lrrr',
    r'Routing / probe & LR ($10^{-4}$) & Best (\%) & Final (\%)',
    [[a, b, f'{val(n):.3f}', f'{val(n, "final_top1"):.3f}'] for a, b, n in geom],
)

base150 = val('s2_mn_r3_lr75_150e')
probes = [
    (r'Spatial $R=3$ (selected)', '$7.5$', 5, 0.10, 0.05, 's2_mn_r3_lr75_150e', 'Complete'),
    (r'Warmup 20', '$7.5$', 20, 0.10, 0.05, 's4_mn_warm20_150e', 'Complete'),
    (r'$\lambda$ 1.5$\to$2 curriculum', '$7.5$', 5, 0.10, 0.05, 's4_mn_lamcurr_150e', 'Complete'),
    (r'Learning rate $10^{-3}$', '$10$', 5, 0.10, 0.05, 's5_mn_lr1e3_150e', 'Complete'),
    (r'Drop-path 0.07', '$7.5$', 5, 0.07, 0.05, 's5_mn_dp07_150e', 'Complete'),
    (r'Weight decay 0.03', '$7.5$', 5, 0.10, 0.03, 's5_mn_wd03_150e', '140/150'),
]
probe_rows = []
for lab, lr, warm, dp, wd, n, status in probes:
    delta = val(n) - base150
    probe_rows.append(
        [
            lab,
            lr,
            f'{val(n):.3f}',
            f'{delta:+.3f}',
            status,
        ]
    )
table(
    'probes',
    'lrrrl',
    r'150-epoch probe & LR ($10^{-4}$) & Best (\%) & $\Delta$ vs selected & Status',
    probe_rows,
)

stage7_control = val('s5_mn_dp07_150e')
table(
    'stage7_diagnostics',
    'llrr',
    r'Variant & Setting / change from control & Best (\%) & $\Delta$ (pp)',
    [
        ['Scratch control', 'drop-path 0.07', f'{stage7_control:.3f}', '0.000'],
        ['DeiT initialization', 'initialize from DeiT-S/8 300e',
         f'{val("s7_mn_deitinit_150e"):.3f}',
         f'{val("s7_mn_deitinit_150e") - stage7_control:+.3f}'],
        ['Progressive latent', r'192 additional latent merges',
         f'{val("s7_mn_proglatent_150e"):.3f}',
         f'{val("s7_mn_proglatent_150e") - stage7_control:+.3f}'],
    ],
)

res_rows = [
    ['224', '300 scratch', f'{val("s3_deit_p8_300e"):.3f}', f'{val("s3_mn_r3_lr75_300e"):.3f}', f'{val("s3_mn_r3_lr75_300e") - val("s3_deit_p8_300e"):+.3f}'],
    ['384', '300+30', f'{val("s3_deit_p8_ft384_30e"):.3f}', f'{val("s3_mn_r3_ft384_30e"):.3f}', f'{val("s3_mn_r3_ft384_30e") - val("s3_deit_p8_ft384_30e"):+.3f}'],
    ['512', '300+30+15', f'{val("s4_deit_ft512_15e"):.3f}', f'{val("s4_mn_ft512_15e"):.3f}', f'{val("s4_mn_ft512_15e") - val("s4_deit_ft512_15e"):+.3f}'],
]
table(
    'resolution',
    'llrrr',
    r'Pixels & Schedule & DeiT-S/8 (\%) & MergeNet (\%) & Gap (pp)',
    res_rows,
)

# Recompute the sweep from latest successful full-validation records, never max score.
records = [json.loads(l) for l in (DATA / 'testtime_records.jsonl').read_text().splitlines()]
latest = {}
for r in sorted(records, key=lambda r: r['ts']):
    z = r['result']
    assert z['n_images'] == 50000 and z['is_final']
    assert not z['load'].get('missing') and not z['load'].get('skipped_shape')
    latest[r['key']] = r
sweep = []
for r in latest.values():
    c = r['cell']
    z = r['result']
    sweep.append(
        {
            'key': r['key'],
            'resolution': c['resolution'],
            'ft384': r['key'].endswith('|ft384'),
            'model': c['model'],
            'method': c['method'],
            'patch_tokens': z['achieved_tokens'],
            'top1': z['top1'],
        }
    )
sweep = pd.DataFrame(sweep)
sweep.to_csv(DATA / 'sweep_latest.csv', index=False)
matched = pd.read_csv(DATA / 'testtime_accuracy_curves.csv')
for _, r in matched.iterrows():
    for meth, col in [('mergenet', 'mergenet_top1'), ('tome', 'tome_top1'), ('pitome', 'pitome_top1')]:
        s = sweep[
            (sweep.resolution == r.resolution)
            & (sweep.ft384 == (r.checkpoint_set == 'ft384'))
            & (sweep.method == meth)
            & (sweep.patch_tokens == r.final_patch_tokens)
        ]
        assert len(s) == 1 and abs(float(s.iloc[0].top1) - r[col]) < 1e-8, (r.to_dict(), meth, s.to_dict())
assert len(matched) == 20 and (matched.tome_minus_mergenet_pp > 0).all()
table(
    'sweep_main',
    'lrrrrr',
    r'Checkpoint / pixels & Patches & MergeNet & ToMe & PiToMe & $\Delta$ ToMe',
    [
        [
            ('224 trained / 224' if r.resolution == 224 else '384 tuned / 384'),
            int(r.final_patch_tokens),
            f'{r.mergenet_top1:.3f}',
            f'{r.tome_top1:.3f}',
            f'{r.pitome_top1:.3f}',
            f'{r.tome_minus_mergenet_pp:+.3f}',
        ]
        for _, r in matched[(matched.resolution == 224) | (matched.checkpoint_set == 'ft384')].iterrows()
    ],
)

probe = json.loads((DATA / 'inference_merge_probe.json').read_text())
table(
    'inference_merge',
    'lrrr',
    r'Evaluation of 300-epoch MN checkpoint & Tokens & Top-1 (\%) & vs native',
    [
        ['Native method', 392, f'{probe["mergenet_392"]["top1"]:.3f}', '0.000'],
        ['method=none', 392, f'{probe["none_392"]["top1"]:.3f}', f'{probe["none_392"]["top1"] - probe["mergenet_392"]["top1"]:+.3f}'],
        ['Native method, 523 carriers', 523, f'{probe["mergenet_523"]["top1"]:.3f}', f'{probe["mergenet_523"]["top1"] - probe["mergenet_392"]["top1"]:+.3f}'],
        ['Native method, 196 carriers', 196, f'{probe["mergenet_196"]["top1"]:.3f}', f'{probe["mergenet_196"]["top1"] - probe["mergenet_392"]["top1"]:+.3f}'],
    ],
)

eff = pd.read_csv(DATA / 'publishable_efficiency.csv')
e = eff[eff.source.str.endswith('bench_clean_2026-09-05.json')]
erows = []
for n, lab in [('deit_p16', 'DeiT-S/16'), ('deit_p8', 'DeiT-S/8'), ('mergenet', 'MergeNet, $R=3$')]:
    t = e[(e.model == n) & (e['mode'] == 'train_fwd_bwd')].iloc[0]
    i = e[(e.model == n) & (e['mode'] == 'inference')].iloc[0]
    erows.append(
        [
            lab,
            f'{i.parameters_all / 1e6:.2f}',
            f'{t.median_seconds * 1000:.2f}',
            f'{i.median_seconds * 1000:.2f}',
            f'{i.iqr_seconds * 1000:.3f}',
            f'{t.peak_allocated_gib:.3f}',
            f'{i.peak_allocated_gib:.3f}',
        ]
    )
table(
    'efficiency224',
    'lrrrrrr',
    r'Model & Params (M) & Train (ms) & Infer (ms) & IQR (ms) & Train GiB & Infer GiB',
    erows,
)
lat = pd.read_csv(DATA / 'matched_latency_caveated.csv')
t = lat[lat.method == 'tome'].reset_index(drop=True)
m = lat[lat.method == 'mergenet'].reset_index(drop=True)
table(
    'latency384',
    'rrrrrr',
    r'ToMe patches & MN patches & ToMe ms & MN ms & ToMe GiB & MN GiB',
    [
        [
            int(a.actual_patch_tokens),
            int(b.actual_patch_tokens),
            f'{a.median_seconds * 1000:.2f}',
            f'{b.median_seconds * 1000:.2f}',
            f'{a.peak_reserved_gib:.3f}',
            f'{b.peak_reserved_gib:.3f}',
        ]
        for (_, a), (_, b) in zip(t.iterrows(), m.iterrows())
    ],
)

cifar = pd.read_csv(DATA / 'cifar_seeds/endpoints.csv')
means = pd.read_csv(DATA / 'cifar_seeds/group_means.csv')
contrasts = pd.read_csv(DATA / 'cifar_seeds/contrasts.csv').set_index('contrast')
g = means[means.selector == 'historical'].set_index('geometry')
cifar_rows = []
for geometry, label, contrast in [
    ('global', 'Global', 'historical:r3-global'),
    ('flat8', 'Flat window 8', 'historical:r3-flat8'),
    ('degree', 'Degree-matched', 'historical:r3-degree'),
    ('r3', 'Spatial $R=3$', None),
]:
    acc = f'{g.loc[geometry, "mean_epoch199"]:.2f}$\\pm${g.loc[geometry, "std_epoch199"]:.2f}'
    delta = (f'{contrasts.loc[contrast, "mean_delta_pp"]:+.2f}'
             f'$\\pm${contrasts.loc[contrast, "std_delta_pp"]:.2f}'
             if contrast else r'---')
    cifar_rows.append([label, acc, delta])
table(
    'cifar_seeds',
    'lrr',
    r'Routing & Top-1 (\%) & $R=3$ minus control (pp)',
    cifar_rows,
)
table(
    'cifar_contrasts',
    'lrrr',
    r'Contrast & Mean $\Delta$ (pp) & Sample std & Seeds 42/43/44',
    [
        [
            lab,
            f'{contrasts.loc[key, "mean_delta_pp"]:+.2f}',
            f'{contrasts.loc[key, "std_delta_pp"]:.2f}',
            '{:+.2f}/{:+.2f}/{:+.2f}'.format(
                contrasts.loc[key, 'seed42_delta'],
                contrasts.loc[key, 'seed43_delta'],
                contrasts.loc[key, 'seed44_delta'],
            ),
        ]
        for key, lab in [
            ('historical:r3-global', r'$R=3$ $-$ global'),
            ('historical:r3-flat8', r'$R=3$ $-$ flat'),
            ('historical:r3-degree', r'$R=3$ $-$ degree'),
        ]
    ],
)

table(
    'deitb',
    'lrrr',
    r'Epoch & Train loss & Eval loss & EMA top-1 (\%)',
    [
        ['150', '2.968', '0.889', f'{curves["s6_deitB_p16_300e"].loc[150, "eval_top1"]:.3f}'],
        ['179 (peak)', '2.808', '0.895', f'{val("s6_deitB_p16_300e"):.3f}'],
        ['299 (final)', f'{curves["s6_deitB_p16_300e"].iloc[-1].train_loss:.3f}', f'{curves["s6_deitB_p16_300e"].iloc[-1].eval_loss:.3f}', f'{val("s6_deitB_p16_300e", "final_top1"):.3f}'],
    ],
)

# Learning curves: 300-epoch, 384 fine-tune, 512 fine-tune.
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
for name, label, col in [
    ('s3_deit_p8_300e', 'DeiT-S/8', C['dense']),
    ('s3_mn_r3_lr75_300e', 'MergeNet $R=3$', C['mn']),
]:
    d = curves[name]
    axes[0].plot(d.epoch, d.eval_top1, label=label, color=col, lw=1.6)
axes[0].set(title='224 px, 300 epochs', xlabel='Epoch', ylabel='EMA top-1 (%)', ylim=(50, 83), xlim=(40, 299))
for name, label, col in [
    ('s3_deit_p8_ft384_30e', 'DeiT-S/8', C['dense']),
    ('s3_mn_r3_ft384_30e', 'MergeNet $R=3$', C['mn']),
    ('s3_mn_r5_ft384_30e', 'MergeNet $R=5.143$', '#B69A52'),
]:
    d = curves[name]
    axes[1].plot(d.epoch, d.eval_top1, label=label, color=col, lw=1.6)
axes[1].set(title='384 px fine-tune', xlabel='Epoch', ylim=(81.3, 83.15), xlim=(0, 29))
for name, label, col in [
    ('s4_deit_ft512_15e', 'DeiT-S/8', C['dense']),
    ('s4_mn_ft512_15e', 'MergeNet $R=6.857$', C['mn']),
]:
    d = curves[name]
    axes[2].plot(d.epoch, d.eval_top1, label=label, color=col, lw=1.6)
axes[2].set(title='512 px fine-tune', xlabel='Epoch', ylim=(82.3, 83.2), xlim=(0, 14))
for ax in axes:
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / 'learning_curves.pdf')
plt.close(fig)

# Resolution ladder.
fig, ax = plt.subplots(figsize=(4.4, 2.7))
xs = np.array([0, 1, 2])
dense = [val('s3_deit_p8_300e'), val('s3_deit_p8_ft384_30e'), val('s4_deit_ft512_15e')]
mn = [val('s3_mn_r3_lr75_300e'), val('s3_mn_r3_ft384_30e'), val('s4_mn_ft512_15e')]
ax.plot(xs, dense, 'o-', color=C['dense'], label='DeiT-S/8', lw=1.8, ms=6)
ax.plot(xs, mn, 'o-', color=C['mn'], label='MergeNet', lw=1.8, ms=6)
for x, a, b in zip(xs, dense, mn):
    ax.annotate(f'{b-a:+.2f} pp', (x, (a + b) / 2), textcoords='offset points', xytext=(10, 0), fontsize=8, color='#444')
ax.set(xticks=xs, xticklabels=['224 px\n300e', '384 px\n+30e', '512 px\n+15e'], ylabel='Best EMA top-1 (%)', ylim=(81.0, 83.4))
ax.grid(alpha=0.17, axis='y')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / 'resolution_ladder.pdf')
plt.close(fig)

# Hyperparameter screen.
fig, ax = plt.subplots(figsize=(6.3, 2.8))
labels_probe = [
    r'Warmup 20',
    r'$\lambda$ curriculum',
    r'LR $10^{-3}$',
    r'Drop-path 0.07',
    r'WD 0.03 (140/150)',
]
vals_probe = [
    val('s4_mn_warm20_150e') - base150,
    val('s4_mn_lamcurr_150e') - base150,
    val('s5_mn_lr1e3_150e') - base150,
    val('s5_mn_dp07_150e') - base150,
    val('s5_mn_wd03_150e') - base150,
]
colors = [C['global'] if abs(v) < 0.40 else C['mn'] for v in vals_probe]
y = np.arange(len(labels_probe))
ax.axvspan(-0.40, 0.40, color='#dddddd', alpha=0.7, zorder=0)
ax.axvline(0, color='#444', lw=0.8, zorder=1)
ax.barh(y, vals_probe, color=colors, height=0.62, zorder=2)
ax.text(0.0, 4.45, r'campaign decision band: $\pm$0.40 pp', ha='center', va='bottom', fontsize=8, color='#555')
ax.set_yticks(y)
ax.set_yticklabels(labels_probe)
ax.set_xlabel(r'Best EMA top-1 minus selected 150-epoch $R=3$ (pp)')
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.55, 4.9)
ax.grid(alpha=0.17, axis='x')
fig.tight_layout()
fig.savefig(FIG / 'tuning_screen.pdf')
plt.close(fig)

# Accuracy-only curves. No latency axis: harness configs differ.
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
for ax, res, ft in zip(axes, [224, 384], [False, True]):
    for meth, lab, col in [
        ('mergenet', 'MergeNet', C['mn']),
        ('tome', 'DeiT-S/8 + ToMe', C['tome']),
        ('pitome', 'DeiT-S/8 + PiToMe', C['pitome']),
    ]:
        d = sweep[(sweep.resolution == res) & (sweep.ft384 == ft) & (sweep.method == meth)].sort_values('patch_tokens')
        ax.plot(d.patch_tokens, d.top1, 'o-', label=lab, color=col, ms=4)
    ax.set(
        xlabel='Final patch tokens',
        ylabel='Top-1 (%)',
        title=('224 px, trained at 224' if not ft else '384 px, fine-tuned'),
    )
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'accuracy_sweep.pdf')
plt.close(fig)

# Architecture diagram.  The two-level layout follows slide 3 of 模型.pptx,
# adapted to the executed image-classification path and verified token counts.
fig, ax = plt.subplots(figsize=(6.5, 3.35))
ax.set(xlim=(0, 10), ylim=(0, 5.2))
ax.axis('off')
ax.add_patch(FancyBboxPatch((0.02, 2.78), 7.20, 2.18, boxstyle='round,pad=0.05',
                            facecolor='#EEF4F8', edgecolor='#B9CFDE', lw=0.9))
ax.add_patch(FancyBboxPatch((0.02, 0.28), 9.83, 1.97, boxstyle='round,pad=0.05',
                            facecolor='#FFF5ED', edgecolor='#E7C4AD', lw=0.9))
ax.text(0.22, 4.72, 'FULL-GRID DIFFERENTIABLE ROUTING', color=C['dense'],
        fontsize=7.5, fontweight='bold', va='center')
ax.text(0.22, 2.08, 'PHYSICALLY COMPRESSED GLOBAL REASONING', color=C['mn'],
        fontsize=7.5, fontweight='bold', va='center')

boxes = [
    (0.25, 3.35, 1.70, 'Patch-8 stem\n784 + CLS', C['dense']),
    (2.30, 3.35, 1.98, 'Local encoder ×6\n28×28 grid', C['dense']),
    (4.65, 3.35, 2.46, 'Spatial mass routing ×6\nradius $R=3$', '#675A9B'),
    (7.58, 0.88, 2.08, 'Exact-budget gather\n784 → 392', '#B9502F'),
    (5.15, 0.88, 1.98, 'Recovery cross-attn\n392 Q · 784 K/V', C['mn']),
    (2.60, 0.88, 2.08, 'Latent encoder ×6\n392 + CLS · log $m$', C['mn']),
    (0.25, 0.88, 1.88, 'CLS head\nImageNet', '#5B7450'),
]
for x, y, w, lab, col in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, 0.94, boxstyle='round,pad=0.055',
                                facecolor=col, edgecolor=col, lw=1.0))
    ax.text(x + w / 2, y + 0.47, lab, ha='center', va='center',
            color='white', fontsize=7.0, linespacing=1.15)


def arrow(a, b, col='#38434C', style='-'):
    ax.annotate('', xy=b, xytext=a,
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.25, linestyle=style,
                                shrinkA=0, shrinkB=0))


arrow((1.99, 3.82), (2.26, 3.82))
arrow((4.32, 3.82), (4.61, 3.82))
arrow((7.15, 3.82), (8.62, 1.86))
arrow((7.54, 1.35), (7.17, 1.35))
arrow((5.11, 1.35), (4.72, 1.35))
arrow((2.56, 1.35), (2.17, 1.35))
arrow((3.29, 3.31), (6.10, 1.86), C['dense'], '--')
ax.text(4.55, 2.50, 'full-grid features\nfor recovery K/V', ha='center', va='center',
        fontsize=7.2, color=C['dense'])
ax.text(8.58, 2.64, 'first shorter tensor', ha='center', va='center',
        fontsize=7.2, color='#9D3D23', fontweight='bold')
ax.text(5.62, 3.12, 'soft transport keeps all slots', ha='center', va='center',
        fontsize=7.0, color='#51477C')
ax.text(5.0, 0.48,
        'Spatial support restricts admissible routes; similarities, masses, and carrier locations remain learned.',
        ha='center', va='center', fontsize=7.3, color='#4B4B4B')
fig.savefig(FIG / 'architecture.pdf')
plt.close(fig)

traj = pd.read_csv(DATA / 'token_trajectories.csv')
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
for model, col in [('ToMe', C['tome']), ('MergeNet', C['mn'])]:
    d = traj[
        (traj.resolution == 224)
        & (traj.checkpoint_set == 'e300_224')
        & (traj.final_patch_tokens == 392)
        & (traj.model == model)
    ]
    if len(d):
        axes[0].step(d.layer, d.attention_input_patches, where='mid', color=col, label=model, lw=1.7)
axes[0].set(xlabel='Transformer block', ylabel='Attention-input patches', xticks=[1, 3, 6, 9, 12], title='Layerwise token trajectories')
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.17)
a = json.loads((DATA / 'analytic_cost.json').read_text())['analytic_MAC_budget']
dm = a['deit_s8']
mm = a['mergenet']
cats = [('Projection / MLP', C['dense']), ('Attention matmuls', C['tome']), ('Routing / recovery', C['mn']), ('Stem / head', '#999999')]
dense_mac = [sum(dm[k] for k in ['attn_qkv', 'attn_proj', 'mlp']), dm['attn_matmul'], 0, dm['patch_embed'] + dm['head']]
merge_mac = [
    sum(v for k, v in mm.items() if ('attn_qkv' in k or 'attn_proj' in k or 'mlp' in k)),
    sum(v for k, v in mm.items() if ('matmul' in k and not k.startswith('xattn'))),
    sum(v for k, v in mm.items() if k.startswith(('dtem', 'xattn'))),
    mm['patch_embed'] + mm['head'],
]
assert abs(sum(merge_mac) - a['mergenet_total_MACs']) < 1
base = np.zeros(2)
for i, (name, col) in enumerate(cats):
    v = np.array([dense_mac[i], merge_mac[i]]) / 1e9
    axes[1].bar(['DeiT-S/8', 'MergeNet'], v, bottom=base, label=name, color=col, width=0.55)
    base += v
axes[1].set(ylabel='Estimated GMAC / image', title='224 px arithmetic estimate', ylim=(0, 29))
axes[1].legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout()
fig.savefig(FIG / 'cost_and_tokens.pdf')
plt.close(fig)

status_map = {'COMPLETE': 'Complete', 'PARTIAL_SNAPSHOT': 'Partial', 'NO_COMPLETED_EPOCH': 'No epoch'}
allrows = []
for n, p in protocol.items():
    allrows.append(
        [
            r'\texttt{' + n.replace('_', r'\_') + '}',
            str(p['epochs_done']) + '/' + str(p['epochs_target']),
            status_map[p['status']],
            (f'{val(n):.3f}' if n in run.index else '--'),
        ]
    )
table('run_inventory', 'lrlr', 'Run ID & Epochs & Status & Best top-1', allrows)

cifar_r3_global = float(contrasts.loc['historical:r3-global', 'mean_delta_pp'])
checks = {
    'summary_rows': int(sum(len(v) for v in curves.values())),
    'runs_with_metrics': int(len(curves)),
    'complete_runs': int(sum(p['status'] == 'COMPLETE' for p in protocol.values())),
    'valid_compression_triplets': int(len(matched)),
    'tome_accuracy_wins': int((matched.tome_minus_mergenet_pp > 0).sum()),
    'cifar_archived_jobs': int(len(cifar)),
    'cifar_reported_geometry_jobs': int((cifar.selector == 'historical').sum()),
    'gaps_pp': {
        'r3_minus_global': val('s2_mn_r3_150e') - val('s2_mn_global_150e'),
        'r3_minus_flat': val('s2_mn_r3_150e') - val('s2_mn_flat_w8_150e'),
        'mn_minus_dense_224': val('s3_mn_r3_lr75_300e') - val('s3_deit_p8_300e'),
        'mn_minus_dense_384': val('s3_mn_r3_ft384_30e') - val('s3_deit_p8_ft384_30e'),
        'mn_minus_dense_512': val('s4_mn_ft512_15e') - val('s4_deit_ft512_15e'),
        'drop_path_delta': val('s5_mn_dp07_150e') - base150,
        'warmup_delta': val('s4_mn_warm20_150e') - base150,
        'curriculum_delta': val('s4_mn_lamcurr_150e') - base150,
        'deit_initialization_delta_vs_dp07': val('s7_mn_deitinit_150e') - stage7_control,
        'progressive_latent_delta_vs_dp07': val('s7_mn_proglatent_150e') - stage7_control,
        'cifar_r3_minus_global_mean': cifar_r3_global,
        'cifar_r3_minus_degree_mean': float(contrasts.loc['historical:r3-degree', 'mean_delta_pp']),
        'inference_none_minus_native': probe['none_392']['top1'] - probe['mergenet_392']['top1'],
    },
    'warning': 'No accuracy-latency Pareto curve is generated: proportional-attention settings differ.',
}
(ROOT / 'research/numerical_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
