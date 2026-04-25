"""One-shot: extract block-level centroid lookup from nhgis0028_shape.zip (52 state zips).

Saves: SHP_DIR / 'block_centroids.pkl'  →  DataFrame['block_id', 'lat', 'lon']
"""
import io
import os
import tempfile
import zipfile

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from paths import SHP_DIR

OUTER_ZIP = r'F:\Data\ACS\shape\nhgis0028_shape.zip'
OUT = SHP_DIR / 'block_centroids.pkl'

frames = []
with zipfile.ZipFile(OUTER_ZIP) as z1:
    inner_zips = [n for n in z1.namelist() if n.endswith('.zip')]
    print(f'found {len(inner_zips)} state zips')
    for inner_name in tqdm(inner_zips, desc='states'):
        with z1.open(inner_name) as raw:
            buf = raw.read()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'inner.zip')
            with open(p, 'wb') as f:
                f.write(buf)
            try:
                gdf = gpd.read_file(f'zip://{p}')
            except Exception as e:
                tqdm.write(f'[skip] {inner_name}: {e}')
                continue
        frames.append(gdf[['GEOID20', 'INTPTLAT20', 'INTPTLON20']].copy())
        del gdf

print('concatenating...', flush=True)
blocks = pd.concat(frames, ignore_index=True)
print(f'raw rows: {len(blocks):,}')

blocks['block_id'] = blocks['GEOID20'].astype(str).str.zfill(15)
blocks['lat'] = pd.to_numeric(blocks['INTPTLAT20'], errors='coerce').astype('float32')
blocks['lon'] = pd.to_numeric(blocks['INTPTLON20'], errors='coerce').astype('float32')
blocks = blocks[['block_id', 'lat', 'lon']].dropna().drop_duplicates(subset='block_id')
print(f'final: {len(blocks):,}')
blocks.to_pickle(OUT)
print(f'saved → {OUT}  ({blocks.memory_usage(deep=True).sum()/1e6:.1f} MB)')
