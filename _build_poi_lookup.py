"""Build a US-only (PLACEKEY -> LAT/LON/NAME) lookup from the 130 partitioned POI files.

Run once, save pickle, reuse for trajectory plotting etc.
"""
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from paths import SHP_DIR

POI_DIR = Path(r'F:\Data\Dewey\SafeGraph\global-places-poi-geometry')
COLS = ['PLACEKEY', 'LATITUDE', 'LONGITUDE', 'LOCATION_NAME', 'ISO_COUNTRY_CODE', 'TOP_CATEGORY']
OUT = SHP_DIR / 'poi_lookup.pkl'

files = sorted(glob.glob(str(POI_DIR / '*.snappy.parquet')))
print(f'POI files: {len(files)}')

frames = []
for f in tqdm(files, desc='POI'):
    try:
        sub = pd.read_parquet(f, columns=COLS)
    except Exception as e:
        tqdm.write(f'[skip] {f}: {e}')
        continue
    sub = sub[sub['ISO_COUNTRY_CODE'] == 'US']
    sub = sub.dropna(subset=['PLACEKEY', 'LATITUDE', 'LONGITUDE'])
    if sub.empty:
        continue
    sub = sub.drop(columns=['ISO_COUNTRY_CODE'])
    frames.append(sub)

print('\nconcatenating...', flush=True)
lookup = pd.concat(frames, ignore_index=True)
print(f'before dedup: {len(lookup):,}')
lookup = lookup.drop_duplicates(subset='PLACEKEY', keep='first').reset_index(drop=True)
print(f'after  dedup: {len(lookup):,}')

# Rename to lowercase for convenience when joining to visits.placekey
lookup.columns = ['placekey', 'latitude', 'longitude', 'location_name', 'top_category']

lookup.to_pickle(OUT)
print(f'\nsaved: {OUT}  ({lookup.memory_usage(deep=True).sum()/1e6:.1f} MB)')
print(lookup.head())
