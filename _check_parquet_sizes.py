"""Health check on Veraset parquet daily volume across all three folders.
Counts files + total size per day, saves CSV + comparison plot to OD_DIR.

Use to spot incomplete days / weekend dropoffs before running 0-OD.
"""
import pandas as pd
import matplotlib.pyplot as plt

from paths import VISITS_DIR, HOME_VISITS_DIR, WORK_VISITS_DIR, OD_DIR

DATE_START = '2025-04-30'
DATE_END   = '2025-06-01'

DIRS = {
    'visits':      VISITS_DIR,
    'home-visits': HOME_VISITS_DIR,
    'work-visits': WORK_VISITS_DIR,
}
COLORS = {'visits': '#1f77b4', 'home-visits': '#ff7f0e', 'work-visits': '#2ca02c'}

dates = pd.date_range(DATE_START, DATE_END).strftime('%Y-%m-%d').tolist()

rows = []
for d in dates:
    row = {'date': d, 'dow': pd.Timestamp(d).day_name()[:3]}
    for label, dr in DIRS.items():
        files = list(dr.glob(f'{d}*.snappy.parquet'))
        row[f'{label}_n'] = len(files)
        row[f'{label}_gb'] = sum(f.stat().st_size for f in files) / (1024 ** 3)
    rows.append(row)

df = pd.DataFrame(rows)
df['total_gb'] = df[[f'{k}_gb' for k in DIRS]].sum(axis=1)
df['total_n']  = df[[f'{k}_n'  for k in DIRS]].sum(axis=1)
df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek >= 5

csv_path = OD_DIR / 'parquet_sizes.csv'
df.to_csv(csv_path, index=False)
print(f'Saved CSV: {csv_path}')
print()
print(df[['date', 'dow', 'visits_n', 'visits_gb', 'home-visits_gb', 'work-visits_gb',
         'total_gb', 'total_n']].to_string(index=False, formatters={
    'visits_gb': '{:>7.2f}'.format, 'home-visits_gb': '{:>7.2f}'.format,
    'work-visits_gb': '{:>7.2f}'.format, 'total_gb': '{:>7.2f}'.format}))

# 2-panel plot: stacked size + file count, weekend bands
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
x = range(len(df))

# --- size panel
bottom = pd.Series(0.0, index=df.index)
for label in DIRS:
    ax1.bar(x, df[f'{label}_gb'], bottom=bottom, color=COLORS[label], label=label)
    bottom = bottom + df[f'{label}_gb']
for i, is_we in enumerate(df['is_weekend']):
    if is_we:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red', zorder=0)
ax1.set_ylabel('Total size (GB)')
ax1.set_title(f'Veraset parquet daily volume — {DATE_START} to {DATE_END}  '
              f'(red bands = Sat/Sun)')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# --- file count panel
bottom = pd.Series(0.0, index=df.index)
for label in DIRS:
    ax2.bar(x, df[f'{label}_n'], bottom=bottom, color=COLORS[label], label=label)
    bottom = bottom + df[f'{label}_n']
for i, is_we in enumerate(df['is_weekend']):
    if is_we:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red', zorder=0)
ax2.set_ylabel('File count')
ax2.set_title('File count per folder per day (constant ⇒ files complete; dips ⇒ missing files)')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{d.split('-', 1)[1]}\n({wd})" for d, wd in zip(df['date'], df['dow'])],
                    rotation=70, fontsize=7)
ax2.grid(axis='y', alpha=0.3)
ax2.legend(loc='upper right')

plt.tight_layout()
plot_path = OD_DIR / 'parquet_sizes.png'
plt.savefig(plot_path, dpi=120)
plt.close()
print(f'\nPlot saved: {plot_path}')
