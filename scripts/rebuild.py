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


main = [
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
    'main',
    'llrrrr',
    r'Model & Pixels & Epochs & Best (\%) & Final (\%) & Best epoch',
    [
        [a, b, c, f'{val(n):.3f}', f'{val(n, "final_top1"):.3f}', int(run.loc[n, 'best_epoch'])]
        for a, b, c, n in main
    ],
)

geom = [
    ('Global', '$5.0$', 's2_mn_global_150e'),
    ('Flat window 8', '$5.0$', 's2_mn_flat_w8_150e'),
    ('Spatial $R=2$', '$5.0$', 's2_mn_r2_150e'),
    ('Spatial $R=3$', '$5.0$', 's2_mn_r3_150e'),
    ('Spatial $R=3$', '$7.5$', 's2_mn_r3_lr75_150e'),
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
order = ['global', 'flat8', 'degree', 'r3']
labels = {
    'global': 'Global',
    'flat8': 'Flat 8',
    'degree': 'Degree\nmatched',
    'r3': r'$R=3$',
}
cifar_rows = []
for selector, slab in [('historical', 'Historical selector'), ('rowwise', 'Rowwise selector')]:
    g = means[means.selector == selector].set_index('geometry')
    cifar_rows.append(
        [
            slab,
            f'{g.loc["global", "mean_epoch199"]:.2f}$\\pm${g.loc["global", "std_epoch199"]:.2f}',
            f'{g.loc["flat8", "mean_epoch199"]:.2f}$\\pm${g.loc["flat8", "std_epoch199"]:.2f}',
            f'{g.loc["degree", "mean_epoch199"]:.2f}$\\pm${g.loc["degree", "std_epoch199"]:.2f}',
            f'{g.loc["r3", "mean_epoch199"]:.2f}$\\pm${g.loc["r3", "std_epoch199"]:.2f}',
        ]
    )
table(
    'cifar_seeds',
    'lrrrr',
    r'Selector & Global & Flat 8 & Degree-matched & $R=3$',
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
            ('historical:r3-global', r'Historical $R=3$ $-$ global'),
            ('historical:r3-flat8', r'Historical $R=3$ $-$ flat'),
            ('historical:r3-degree', r'Historical $R=3$ $-$ degree'),
            ('rowwise:r3-global', r'Rowwise $R=3$ $-$ global'),
            ('rowwise:r3-flat8', r'Rowwise $R=3$ $-$ flat'),
            ('rowwise:r3-degree', r'Rowwise $R=3$ $-$ degree'),
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

# CIFAR three-seed geometry: endpoints plus mean training trajectories.
geom_colors = {
    'global': C['global'],
    'flat8': C['flat'],
    'degree': C['degree'],
    'r3': C['mn'],
}
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
x = np.arange(len(order))
for col_i, selector, title in [
    (0, 'historical', 'Historical selector'),
    (1, 'rowwise', 'Per-sample selector'),
]:
    ax = axes[0, col_i]
    g = means[means.selector == selector].set_index('geometry').loc[order]
    ax.bar(
        x,
        g.mean_epoch199,
        yerr=g.std_epoch199,
        color=[geom_colors[k] for k in order],
        capsize=3,
        width=0.72,
        error_kw={'lw': 0.9},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.17, axis='y')
    ax.set_ylim(67.8, 72.2)
    if col_i == 0:
        ax.set_ylabel('Epoch-199 EMA top-1 (%)')
    axc = axes[1, col_i]
    for geom in order:
        series = []
        for seed in (42, 43, 44):
            d = pd.read_csv(DATA / 'cifar_seeds/summaries' / f'{selector}_{geom}_s{seed}' / 'summary.csv')
            assert d.epoch.tolist() == list(range(len(d))), f'{selector}_{geom}_s{seed}'
            series.append(d.eval_top1.astype(float).to_numpy())
        arr = np.vstack(series)
        mu, sd = arr.mean(0), arr.std(0, ddof=1)
        ep = np.arange(len(mu))
        axc.plot(ep, mu, color=geom_colors[geom], lw=1.5, label=labels[geom])
        axc.fill_between(ep, mu - sd, mu + sd, color=geom_colors[geom], alpha=0.18, lw=0)
    axc.set(xlabel='Epoch', xlim=(80, 199), ylim=(52.0, 72.2))
    axc.grid(alpha=0.17)
    if col_i == 0:
        axc.set_ylabel('EMA top-1 mean ± 1 s.d. (%)')
        axc.legend(frameon=False, fontsize=7, loc='lower right')
fig.tight_layout()
fig.savefig(FIG / 'cifar_seeds.pdf')
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
        title=('224 px, trained at 224' if not ft else '384 px, fine-tuned (MN $R=5.143$)'),
    )
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'accuracy_sweep.pdf')
plt.close(fig)

# Architecture diagram sized for the manuscript's actual column width.
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.set(xlim=(0, 6.5), ylim=(0, 3.0))
ax.axis('off')
boxes = [
    (0.1, 2.05, 1.5, 'Patch-8 stem\n785 tokens', C['dense']),
    (2.02, 2.05, 1.9, '6 local blocks\n785 tokens each', C['dense']),
    (4.38, 2.05, 2.0, '6 soft routing steps\n785 slots retained', C['dense']),
    (4.38, 0.60, 2.0, 'Hard top-k gather\n785 → 393 tokens', C['mn']),
    (2.02, 0.60, 1.9, 'Global recovery\n393 queries, 785 K/V', C['mn']),
    (0.1, 0.60, 1.5, '6 latent blocks\n393 tokens + head', C['mn']),
]
for x, y, w, lab, col in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, 0.68, boxstyle='round,pad=0.035', facecolor=col, edgecolor=col))
    ax.text(x + w / 2, y + 0.34, lab, ha='center', va='center', color='white', fontsize=8.5)


def arrow(a, b, col='#38434C'):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color=col, lw=1.3))


arrow((1.65, 2.39), (1.95, 2.39))
arrow((3.97, 2.39), (4.31, 2.39))
arrow((5.38, 2.00), (5.38, 1.33))
arrow((4.31, 0.94), (3.97, 0.94))
arrow((1.95, 0.94), (1.65, 0.94))
arrow((2.97, 2.00), (2.97, 1.33), C['dense'])
ax.text(2.82, 1.68, 'Final local features\nas recovery K/V', ha='right', va='center', fontsize=8, color=C['dense'])
ax.text(5.22, 1.67, 'Original-grid\nR3 support', ha='right', va='center', fontsize=8, color=C['dense'])
ax.text(3.25, 0.20, 'Full slots (blue) → physical gather → compressed tokens (orange)', ha='center', fontsize=8)
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
    'cifar_complete_jobs': int(len(cifar)),
    'gaps_pp': {
        'r3_minus_global': val('s2_mn_r3_150e') - val('s2_mn_global_150e'),
        'r3_minus_flat': val('s2_mn_r3_150e') - val('s2_mn_flat_w8_150e'),
        'mn_minus_dense_224': val('s3_mn_r3_lr75_300e') - val('s3_deit_p8_300e'),
        'mn_minus_dense_384': val('s3_mn_r3_ft384_30e') - val('s3_deit_p8_ft384_30e'),
        'mn_minus_dense_512': val('s4_mn_ft512_15e') - val('s4_deit_ft512_15e'),
        'drop_path_delta': val('s5_mn_dp07_150e') - base150,
        'warmup_delta': val('s4_mn_warm20_150e') - base150,
        'curriculum_delta': val('s4_mn_lamcurr_150e') - base150,
        'cifar_historical_r3_minus_global_mean': cifar_r3_global,
        'cifar_historical_r3_minus_degree_mean': float(contrasts.loc['historical:r3-degree', 'mean_delta_pp']),
        'inference_none_minus_native': probe['none_392']['top1'] - probe['mergenet_392']['top1'],
    },
    'warning': 'No accuracy-latency Pareto curve is generated: proportional-attention settings differ.',
}
(ROOT / 'research/numerical_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
