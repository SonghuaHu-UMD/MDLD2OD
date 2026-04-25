"""Validation plots for the 2-week test run of 0-OD_from_Individual.

Processes 5/2 → 5/15 (14 days, all weekday/weekend mix, no skipped days).

Outputs (to OD_DIR):
  test_summary.txt           per-day + pooled summary
  test_hourly_profile.png    pooled hourly distribution — 3 streams compared
  test_hourly_wd_we.png      weekday vs weekend hourly avg from od_flow_hourly.pkl
  test_top_destinations.png  pooled top 20 destinations by flow
"""
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

from paths import OD_DIR

# Auto-discover processed dates from per-day pkl files
_pat = re.compile(r'od_(\d{4}-\d{2}-\d{2})\.pkl$')
DATES = sorted({m.group(1) for f in glob.glob(str(OD_DIR / 'od_2025-*.pkl'))
                if (m := _pat.search(f))})
print(f'Discovered {len(DATES)} processed dates: {DATES[0]} → {DATES[-1]}')

# ---- Load per-day
per_day = {d: pd.read_pickle(OD_DIR / f'od_{d}.pkl') for d in DATES}
per_day_v = {d: pd.read_pickle(OD_DIR / f'od_visits_{d}.pkl') for d in DATES}
od_pooled = pd.concat(per_day.values(), ignore_index=True)
od_visits_pooled = pd.concat(per_day_v.values(), ignore_index=True)

# Monthly aggregates
od_all = pd.read_pickle(OD_DIR / 'od_flow_all.pkl')
od_h = pd.read_pickle(OD_DIR / 'od_flow_hourly.pkl')
od_v = pd.read_pickle(OD_DIR / 'od_visits_all.pkl')

# ---- Summary
lines = ['=== 2-week test run summary ===\n']
for d in DATES:
    wk = pd.Timestamp(d).day_name()[:3]
    f = per_day[d]
    v = per_day_v[d]
    n_pairs = f[['start_cbg', 'end_cbg']].drop_duplicates().shape[0]
    ratio = f['flow'].sum() / f['flow_raw'].sum() if f['flow_raw'].sum() else float('nan')
    lines.append(
        f'{d} ({wk}):  pairs={n_pairs:>9,}  '
        f'flow={f["flow"].sum():>11,.0f}  flow_raw={f["flow_raw"].sum():>11,.0f}  '
        f'visits={v["visits"].sum():>11,.0f}  flow/raw={ratio:.3f}'
    )
n_wd = sum(1 for d in DATES if pd.Timestamp(d).dayofweek < 5)
n_we = sum(1 for d in DATES if pd.Timestamp(d).dayofweek >= 5)
lines.append('')
lines.append(f'Pooled ({len(DATES)} days = {n_wd} weekday + {n_we} weekend):')
lines.append(f'  flow     = {od_pooled["flow"].sum():>15,.0f}')
lines.append(f'  flow_raw = {od_pooled["flow_raw"].sum():>15,.0f}')
lines.append(f'  visits   = {od_visits_pooled["visits"].sum():>15,.0f}')
lines.append('')
lines.append(f'od_flow_all.pkl     rows={len(od_all):>10,}  cols={list(od_all.columns)}')
lines.append(f'od_flow_hourly.pkl  rows={len(od_h):>10,}  cols={list(od_h.columns)}')
lines.append(f'od_visits_all.pkl   rows={len(od_v):>10,}  cols={list(od_v.columns)}')
summary_text = '\n'.join(lines)
print(summary_text)
(OD_DIR / 'test_summary.txt').write_text(summary_text, encoding='utf-8')

# ---- Plot 1: pooled hourly profile — three streams
hourly = od_pooled.groupby('depart_hour')[['flow', 'flow_raw']].sum().reset_index()
hourly_v = od_visits_pooled.groupby('arrive_hour')['visits'].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(hourly_v['arrive_hour'], hourly_v['visits'],
        label='visits (raw, arrive_hour, destination-only)',
        marker='^', alpha=0.7, linestyle=':', color='#2ca02c')
ax.plot(hourly['depart_hour'], hourly['flow_raw'],
        label='flow_raw (trip-constructed, no severance/imputation, depart_hour)',
        marker='s', alpha=0.7, linestyle='--', color='#1f77b4')
ax.plot(hourly['depart_hour'], hourly['flow'],
        label='flow (full pipeline, depart_hour)',
        marker='o', alpha=0.95, linestyle='-', color='#ff7f0e')
ax.set_xticks(range(0, 24))
ax.set_xlabel('Hour')
ax.set_ylabel(f'Total trips / visits ({len(DATES)}-day pooled)')
ax.set_title(f'Hourly distribution — three streams compared — {DATES[0]} → {DATES[-1]}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OD_DIR / 'test_hourly_profile.png', dpi=120)
plt.close()

# ---- Plot 2: weekday vs weekend hourly avg (from od_flow_hourly.pkl)
agg_h = od_h.groupby(['hour', 'is_weekday'])[['avg_flow', 'avg_flow_raw']].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 4.5))
for is_wd, label, color in [(True, 'weekday', '#1f77b4'), (False, 'weekend', '#d62728')]:
    sub = agg_h[agg_h['is_weekday'] == is_wd]
    if sub.empty:
        continue
    ax.plot(sub['hour'], sub['avg_flow_raw'], color=color, linestyle='--', alpha=0.55,
            marker='s', markersize=4, label=f'{label} avg_flow_raw')
    ax.plot(sub['hour'], sub['avg_flow'], color=color, linestyle='-', alpha=0.95,
            marker='o', markersize=4, label=f'{label} avg_flow')
ax.set_xticks(range(0, 24))
ax.set_xlabel('Hour')
ax.set_ylabel('Avg trips per day (sum across all OD pairs)')
ax.set_title(f'Hourly avg from od_flow_hourly.pkl  ({n_wd} weekday + {n_we} weekend days)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OD_DIR / 'test_hourly_wd_we.png', dpi=120)
plt.close()

# ---- Plot 3: per-day hourly flow profile, colored by day-of-week
fig, ax = plt.subplots(figsize=(12, 5.5))
DOW_COLORS = {'Mon': '#1f77b4', 'Tue': '#ff7f0e', 'Wed': '#2ca02c', 'Thu': '#d62728',
              'Fri': '#9467bd', 'Sat': '#8c564b', 'Sun': '#e377c2'}
DOW_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
seen = set()
for d in DATES:
    h = per_day[d].groupby('depart_hour')['flow'].sum().reset_index()
    wk = pd.Timestamp(d).day_name()[:3]
    label = wk if wk not in seen else None
    seen.add(wk)
    ax.plot(h['depart_hour'], h['flow'], color=DOW_COLORS[wk], alpha=0.6,
            linewidth=1.3, label=label)
ax.set_xticks(range(0, 24))
ax.set_xlabel('Depart hour')
ax.set_ylabel('flow (trips per day)')
ax.set_title(f'Hourly flow profile per day — colored by day-of-week  ({len(DATES)} days)')
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(d) for d in DOW_ORDER if d in labels]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          loc='upper left', fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OD_DIR / 'test_hourly_per_day.png', dpi=120)
plt.close()

# ---- Plot 4: continuous hourly time series in date order, weekend bands
flow_long = []
visits_long = []
for d in DATES:
    h = per_day[d].groupby('depart_hour')['flow'].sum().reset_index()
    h['datetime'] = pd.to_datetime(d) + pd.to_timedelta(h['depart_hour'], unit='h')
    flow_long.append(h)
    v = per_day_v[d].groupby('arrive_hour')['visits'].sum().reset_index()
    v['datetime'] = pd.to_datetime(d) + pd.to_timedelta(v['arrive_hour'], unit='h')
    visits_long.append(v)
flow_long = pd.concat(flow_long, ignore_index=True).sort_values('datetime')
visits_long = pd.concat(visits_long, ignore_index=True).sort_values('datetime')

# Reindex onto a complete hourly grid so skipped days show as gaps
full_idx = pd.date_range(pd.Timestamp(DATES[0]),
                         pd.Timestamp(DATES[-1]) + pd.Timedelta(hours=23), freq='h')
flow_long = flow_long.set_index('datetime').reindex(full_idx)
visits_long = visits_long.set_index('datetime').reindex(full_idx)

fig, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                                  gridspec_kw={'hspace': 0.15})
day_range = pd.date_range(pd.Timestamp(DATES[0]), pd.Timestamp(DATES[-1]), freq='D')
# Weekend bands behind both panels
for ax in (ax_v, ax_f):
    for day in day_range:
        if day.dayofweek >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), alpha=0.12, color='red', zorder=0)

# Figure-level title so it sits well above the day-of-week labels
fig.suptitle(f'Hourly time series — {DATES[0]} → {DATES[-1]}  '
             f'(red bands / labels = Sat/Sun, white gaps = skipped days)',
             fontsize=11, y=0.985)

# Day-of-week labels above the top panel (red = weekend, gray = weekday)
for day in day_range:
    is_we = day.dayofweek >= 5
    ax_v.text(day + pd.Timedelta(hours=12), 1.02, day.day_name()[:3],
              transform=ax_v.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=7,
              color='red' if is_we else 'gray',
              fontweight='bold' if is_we else 'normal')

ax_v.plot(visits_long.index, visits_long['visits'], color='#2ca02c',
          linewidth=0.9, alpha=0.95)
ax_v.set_ylabel('visits per hour\n(raw, arrive_hour)')
ax_v.grid(alpha=0.3)

ax_f.plot(flow_long.index, flow_long['flow'], color='#ff7f0e',
          linewidth=1.0, alpha=0.95)
ax_f.set_ylabel('flow per hour\n(full pipeline, depart_hour)')
ax_f.set_xlabel('Date / hour')
ax_f.grid(alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.95))   # leave headroom for suptitle + DOW labels
plt.savefig(OD_DIR / 'test_hourly_timeseries.png', dpi=120)
plt.close()

# ---- Plot 5: per-state hourly profile (one subplot per state, independent y),
#              weekday only, three streams overlaid (visits / flow_raw / flow)
FIPS_TO_ABBR = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE',
    '11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA',
    '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN',
    '28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM',
    '36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI',
    '45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA',
    '54':'WV','55':'WI','56':'WY',
}
WEEKDAY_DATES = [d for d in DATES if pd.Timestamp(d).dayofweek < 5]
N_WD = len(WEEKDAY_DATES)

od_wd = pd.concat([per_day[d] for d in WEEKDAY_DATES], ignore_index=True)
visits_wd = pd.concat([per_day_v[d] for d in WEEKDAY_DATES], ignore_index=True)
od_wd['state'] = od_wd['end_cbg'].str[:2]
visits_wd['state'] = visits_wd['end_cbg'].str[:2]

# Keep states with non-trivial visits (≥0.1% of total) — ranking by flow alone
# would include states that only appear via end-of-day home imputation (device
# lives in state X but visited POIs elsewhere); those have visits=0 / flow_raw=0.
state_visits = visits_wd.groupby('state')['visits'].sum().sort_values(ascending=False)
top_states = state_visits[state_visits >= state_visits.sum() * 1e-3].index.tolist()

flow_p = (od_wd[od_wd['state'].isin(top_states)]
          .groupby(['state', 'depart_hour'], as_index=False)
          .agg(flow=('flow', 'sum'), flow_raw=('flow_raw', 'sum')))
flow_p['avg_flow'] = flow_p['flow'] / N_WD
flow_p['avg_flow_raw'] = flow_p['flow_raw'] / N_WD
visits_p = (visits_wd[visits_wd['state'].isin(top_states)]
            .groupby(['state', 'arrive_hour'], as_index=False)['visits'].sum())
visits_p['avg_visits'] = visits_p['visits'] / N_WD

import math
n_states = len(top_states)
ncols = min(n_states, 4)
nrows = math.ceil(n_states / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
for ax in axes.flat[n_states:]:   # hide unused panels
    ax.set_visible(False)
for i, state in enumerate(top_states):
    ax = axes[i // ncols, i % ncols]
    fp = flow_p[flow_p['state'] == state]
    vp = visits_p[visits_p['state'] == state]
    ax.plot(vp['arrive_hour'], vp['avg_visits'],
            color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.85, label='visits')
    ax.plot(fp['depart_hour'], fp['avg_flow_raw'],
            color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.85, label='flow_raw')
    ax.plot(fp['depart_hour'], fp['avg_flow'],
            color='#ff7f0e', linewidth=1.5, alpha=0.95, label='flow')
    ax.set_title(f'{FIPS_TO_ABBR.get(state, state)}  ({state})', fontsize=10)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='upper left')
fig.suptitle(f'Hourly profile by destination state — {n_states} states with data, '
             f'weekday avg ({N_WD} weekdays, independent y-axes)',
             fontsize=12, y=0.995)
fig.text(0.5, 0.005, 'Hour', ha='center', fontsize=10)
fig.text(0.005, 0.5, 'Avg trips / visits per weekday', va='center', rotation=90, fontsize=10)
plt.tight_layout(rect=(0.012, 0.02, 1, 0.97))
plt.savefig(OD_DIR / 'test_state_profiles_wd.png', dpi=120)
plt.close()

# ---- Plot 6: pooled top destinations
top_dest = od_pooled.groupby('end_cbg')['flow'].sum().nlargest(20).reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(top_dest)), top_dest['flow'])
ax.set_xticks(range(len(top_dest)))
ax.set_xticklabels(top_dest['end_cbg'], rotation=70, fontsize=7)
ax.set_xlabel('Destination CBG')
ax.set_ylabel(f'Total trips ({len(DATES)}-day sum)')
ax.set_title(f'Top 20 destinations by flow — {DATES[0]} → {DATES[-1]}')
plt.tight_layout()
plt.savefig(OD_DIR / 'test_top_destinations.png', dpi=120)
plt.close()

print(f'\nPlots saved to: {OD_DIR}')
