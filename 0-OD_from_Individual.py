import pandas as pd
import numpy as np
import glob
import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from paths import VISITS_DIR, HOME_VISITS_DIR, WORK_VISITS_DIR, SHP_DIR, OD_DIR

PARQUET_THREADS = 8   # parallel file readers in read_and_prefilter_date; I/O-bound

all_dates = pd.date_range(start='2025-05-13', end='2025-05-17', freq='D').strftime('%Y-%m-%d').tolist()

# Upstream Veraset data outage on these days (total size <6 GB vs typical 20+GB).
# They're kept in all_dates (so adjacent days still get cache spillover) but the
# loop skips processing them. The monthly avg denominator uses od_month['date']
# so these don't appear in the numerator or denominator.
SKIP_DATES = {'2025-05-17', '2025-05-18', '2025-05-19', '2025-05-20'}

# home-visits / work-visits rows carry placekey=null and tag themselves via
# safegraph_place_id='home'/'work', which we treat as the POI id for those rows.
VISIT_FOLDERS = [VISITS_DIR, HOME_VISITS_DIR, WORK_VISITS_DIR]
VISIT_COLS = ['caid', 'local_timestamp', 'naics_code', 'minimum_dwell',
              'safegraph_place_id', 'placekey', 'census_block_group']
VISIT_RENAME = ['caid', 'end_timestamp', 'end_naics_code', 'minimum_dwell',
                '_sgid', 'end_place_id', 'end_cbg']

PING_COLLAPSE_MIN = 30   # consecutive same-POI pings within this gap are one visit
MAX_SPEED_KMH = 120      # gap < dist/MAX_SPEED ⇒ physically impossible (Rule C sever)

# Distance-banded speed assumption. Used for both per-row dwell back-derivation and
# home imputation. dist >= TRAVEL_MAX_DIST_KM returns NaN — home imputation reads
# this as "tourist, skip"; the dwell derivation falls back to MAX_SPEED_KMH.
TRAVEL_SPEED_PROFILE = [
    (2.0,    5),    # walk
    (100.0, 35),    # car
]
TRAVEL_MAX_DIST_KM = 100

# Dwell ceiling (minutes) by POI category. final_dwell above this is interpreted as
# evidence of a missed intermediate visit → outgoing trip severed.
DWELL_CEILING_MIN = {
    'home':    14 * 60,
    'work':    12 * 60,
    'school':  10 * 60,   # NAICS prefix '61'
    'others':   4 * 60,
}


def _travel_hours(dist_km):
    """Per-trip synthetic travel-time (hours) from TRAVEL_SPEED_PROFILE.
    Returns NaN when dist >= TRAVEL_MAX_DIST_KM (caller decides what to do)."""
    dist_km = np.asarray(dist_km, dtype='float64')
    hours = np.full_like(dist_km, np.nan)
    prev_upper = 0.0
    for upper, speed in TRAVEL_SPEED_PROFILE:
        mask = (dist_km >= prev_upper) & (dist_km < upper)
        hours[mask] = dist_km[mask] / speed
        prev_upper = upper
    return hours


def _ceiling_min(place_id_series, naics_series):
    """Vectorized: per-row dwell ceiling in minutes by POI category.
    home/work via place_id; school via NAICS prefix '61'; everything else 'others'."""
    out = np.full(len(place_id_series), DWELL_CEILING_MIN['others'], dtype='float64')
    is_school = naics_series.astype(str).str.startswith('61').fillna(False).values
    is_work = (place_id_series == 'work').values
    is_home = (place_id_series == 'home').values
    out[is_school] = DWELL_CEILING_MIN['school']
    out[is_work]   = DWELL_CEILING_MIN['work']
    out[is_home]   = DWELL_CEILING_MIN['home']
    return out


def _prefilter_one_file(f, need_tracts_int):
    """Read one parquet and apply the tract + NaN filters. Returns None on error
    or empty result. Isolated for ThreadPoolExecutor."""
    try:
        sub = pd.read_parquet(f, columns=VISIT_COLS)
    except Exception as e:
        print(f'[skip] {f}: {e}')
        return None
    sub.columns = VISIT_RENAME
    # home/work rows have placekey=null, fall back to safegraph_place_id
    sub['end_place_id'] = sub['end_place_id'].fillna(sub['_sgid'])
    sub = sub.drop(columns=['_sgid'])
    sub = sub[sub['minimum_dwell'] >= 0]
    if sub.empty:
        return None
    # `census_block_group` is actually a 15-digit Census BLOCK despite the
    # misleading name. Some rows are literal 'NA' strings → coerce/dropna.
    sub['end_cbg'] = pd.to_numeric(sub['end_cbg'], errors='coerce')
    sub = sub.dropna(subset=['end_cbg'])
    if sub.empty:
        return None
    sub['end_cbg'] = sub['end_cbg'].astype('int64')
    mask = (sub['end_cbg'] // 10000).isin(need_tracts_int)
    if not mask.any():
        return None
    sub = sub.loc[mask].copy()
    # Keep both: end_block (15-digit) for block-centroid lookup on home/work,
    # end_cbg (12-digit) for OD aggregation.
    sub['end_block'] = sub['end_cbg'].astype(str).str.zfill(15)
    sub['end_cbg'] = (sub['end_cbg'] // 1000).astype(str).str.zfill(12)
    # Some Veraset rows have corrupt local_timestamp (huge negative ints out of
    # nanosecond bounds). errors='coerce' → NaT, then drop them.
    sub['end_timestamp'] = pd.to_datetime(sub['end_timestamp'], unit='s', errors='coerce')
    sub = sub.dropna(subset=['end_timestamp'])
    if sub.empty:
        return None
    return sub


def read_and_prefilter_date(date_str, need_tracts_int):
    """Load one date's visits across all three folders in parallel via ThreadPool
    (parquet read is I/O-bound), applying tract + NaN filters per file so the
    3-day concat in the main loop stays small and cacheable."""
    all_files = []
    for folder in VISIT_FOLDERS:
        all_files.extend(glob.glob(str(folder / f'{date_str}*.snappy.parquet')))
    if not all_files:
        cols = [c for c in VISIT_RENAME if c != '_sgid'] + ['end_block']
        return pd.DataFrame(columns=cols)
    frames = []
    with ThreadPoolExecutor(max_workers=PARQUET_THREADS) as pool:
        for sub in pool.map(lambda f: _prefilter_one_file(f, need_tracts_int), all_files):
            if sub is not None:
                frames.append(sub)
    if not frames:
        cols = [c for c in VISIT_RENAME if c != '_sgid'] + ['end_block']
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def build_caid_cbg_lookup(folder, cbg_col, batch_size=200):
    """Run-once builder of caid → home CBG. Streams + dedups in batches because a
    naive concat of all ~8700 files would need ~80 GB RAM."""
    files = sorted(glob.glob(str(folder / '*.snappy.parquet')))
    accumulator = None
    batch = []
    bad = []
    for i, f in enumerate(tqdm(files)):
        try:
            sub = pd.read_parquet(f, columns=['caid', 'census_block_group'])
        except Exception as e:
            bad.append((f, str(e)))
            tqdm.write(f'[skip] {f}: {e}')
            continue
        sub = sub.dropna(subset='census_block_group').drop_duplicates(subset='caid')
        batch.append(sub)
        if (i + 1) % batch_size == 0 or i == len(files) - 1:
            pieces = batch if accumulator is None else [accumulator] + batch
            accumulator = pd.concat(pieces, ignore_index=True).drop_duplicates(subset='caid')
            batch = []
    if bad:
        print(f'Skipped {len(bad)} unreadable file(s): {[f for f, _ in bad]}')
    accumulator.columns = ['caid', cbg_col]
    return accumulator


# Run once to build hw.pkl:
# build_caid_cbg_lookup(HOME_VISITS_DIR, 'home_cbg').to_pickle(SHP_DIR / 'hw.pkl')

# hw.pkl stores raw 15-digit Blocks. Keep home_block (15-digit) for centroid lookup
# on home/work rows, home_cbg (12-digit) for OD aggregation.
hw = pd.read_pickle(SHP_DIR / 'hw.pkl')
hw['home_cbg'] = pd.to_numeric(hw['home_cbg'], errors='coerce')
hw = hw.dropna(subset=['home_cbg'])
_block_int = hw['home_cbg'].astype('int64')
hw['home_block'] = _block_int.astype(str).str.zfill(15)
hw['home_cbg'] = (_block_int // 1000).astype(str).str.zfill(12)
del _block_int
hw_block_map = hw.set_index('caid')['home_block']  # cheap lookup during imputation

print('loading POI lat/lon lookup...', flush=True)
_poi = pd.read_pickle(SHP_DIR / 'poi_lookup.pkl')[['placekey', 'latitude', 'longitude']].copy()
_poi['latitude'] = _poi['latitude'].astype('float32')
_poi['longitude'] = _poi['longitude'].astype('float32')
poi_lat = _poi.set_index('placekey')['latitude']
poi_lon = _poi.set_index('placekey')['longitude']
del _poi

# Block centroid (15-digit) — more precise than CBG centroid; used for home/work rows.
print('loading Block centroid lookup...', flush=True)
_blk = pd.read_pickle(SHP_DIR / 'block_centroids.pkl')
block_lat = _blk.set_index('block_id')['lat']
block_lon = _blk.set_index('block_id')['lon']
del _blk


def _coords(df_src, pid_col, block_col):
    """Return (lat, lon) Series for each row.
    placekey (real POI) → POI lat/lon; 'home'/'work' → Block centroid.
    """
    lat = df_src[pid_col].map(poi_lat)
    lon = df_src[pid_col].map(poi_lon)
    is_cat = df_src[pid_col].isin(['home', 'work'])
    if is_cat.any():
        lat = lat.where(~is_cat, df_src[block_col].map(block_lat))
        lon = lon.where(~is_cat, df_src[block_col].map(block_lon))
    return lat.astype('float32'), lon.astype('float32')


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres; element-wise over numpy arrays."""
    r = 6371.0
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    lon1 = np.radians(lon1); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _build_trips_in_place(d, apply_severance):
    """Build trip-level fields on d (visits sorted by caid + end_timestamp).

    Each output row represents the END of a trip whose origin is the previous
    visit of the same caid. After shifting to populate start_*, depart_time is
    derived from a dwell back-derivation that combines visit timing with the
    SafeGraph minimum_dwell as a lower bound:

        gap         = end_timestamp - end_timestamp_prev
        est_travel  = dist(prev_POI → cur_POI) / TRAVEL_SPEED_PROFILE
        derived     = gap - est_travel
        final_dwell = clamp(max(derived, minimum_dwell_prev), 0, gap)
        depart_time = end_timestamp_prev + final_dwell

    apply_severance=True (main pipeline): also apply Rule C (gap < dist/MAX_SPEED
    → impossible, propagated to next row + bad row's own end_* nulled) and the
    per-NAICS dwell ceiling on final_dwell. Severed rows have start_* nulled.
    apply_severance=False (flow_raw baseline): just construct the trip with the
    dwell-derived depart_time, keep all chains.
    """
    d['start_place_id']   = d.groupby('caid')['end_place_id'].shift(1)
    d['start_cbg']        = d.groupby('caid')['end_cbg'].shift(1)
    d['start_block']      = d.groupby('caid')['end_block'].shift(1)
    d['start_naics_code'] = d.groupby('caid')['end_naics_code'].shift(1)
    d['_end_prev']        = d.groupby('caid')['end_timestamp'].shift(1)
    d['_min_dwell_prev']  = d.groupby('caid')['minimum_dwell'].shift(1).fillna(0)

    slat, slon = _coords(d, 'start_place_id', 'start_block')
    elat, elon = _coords(d, 'end_place_id', 'end_block')
    dist_km = _haversine_km(slat.values, slon.values, elat.values, elon.values)

    gap_arr = (d['end_timestamp'] - d['_end_prev']).dt.total_seconds().values / 3600.0

    # For dist beyond TRAVEL_SPEED_PROFILE fall back to MAX_SPEED so final_dwell
    # collapses toward gap - dist/120 (matches Rule C threshold).
    est_travel = _travel_hours(dist_km)
    est_travel = np.where(np.isnan(est_travel), dist_km / MAX_SPEED_KMH, est_travel)
    derived_hr = gap_arr - est_travel

    min_dwell_hr = d['_min_dwell_prev'].values / 60.0
    final_hr = np.maximum(derived_hr, min_dwell_hr)
    final_hr = np.clip(final_hr, 0.0, gap_arr)
    d['start_timestamp'] = d['_end_prev'] + pd.to_timedelta(final_hr, unit='h')

    severed = pd.Series(False, index=d.index)
    if apply_severance:
        rule_c = pd.Series(gap_arr < dist_km / MAX_SPEED_KMH, index=d.index).fillna(False)
        # Propagate Rule C: bad row's own end_* is untrustworthy too, so the next
        # row (whose start_* came from shifting this end_*) is also severed.
        next_after_c = rule_c.groupby(d['caid']).shift(1, fill_value=False).astype(bool)
        d.loc[rule_c, ['end_cbg', 'end_block']] = None
        severed = severed | rule_c | next_after_c

        ceiling = _ceiling_min(d['start_place_id'], d['start_naics_code'])
        over_ceiling = pd.Series((final_hr * 60.0) > ceiling, index=d.index).fillna(False)
        severed = severed | over_ceiling

    d.loc[severed, ['start_place_id', 'start_cbg', 'start_block', 'start_naics_code']] = None
    d.loc[severed, 'start_timestamp'] = pd.NaT
    d['gap_too_long'] = severed
    d.drop(columns=['_end_prev', '_min_dwell_prev'], inplace=True)
    return d


need_cbgs = pd.read_pickle(SHP_DIR / 'all_ctrct.pkl')
need_cbgs = need_cbgs[0].values.tolist()
need_tracts_int = {int(t) for t in need_cbgs}

# Rolling 3-day cache. Consecutive iterations overlap in 2 of 3 dates, so without
# caching each day's parquets would be read+prefiltered 3 times.
visit_cache = {}

for kk in tqdm(range(1, len(all_dates) - 1)):
    if all_dates[kk] in SKIP_DATES:
        continue
    need_dates = [all_dates[kk - 1], all_dates[kk], all_dates[kk + 1]]
    for d in need_dates:
        if d not in visit_cache:
            visit_cache[d] = read_and_prefilter_date(d, need_tracts_int)
    for d in list(visit_cache):
        if d not in need_dates:
            del visit_cache[d]

    df = pd.concat([visit_cache[d] for d in need_dates], ignore_index=True)

    # ±4h around the target day to cover trips spanning midnight.
    f_dt = datetime.datetime.strptime(all_dates[kk], '%Y-%m-%d')
    date_lower = f_dt - datetime.timedelta(hours=4)
    date_upper = f_dt + datetime.timedelta(hours=24 + 4)
    df = df[(df['end_timestamp'] < date_upper) & (df['end_timestamp'] > date_lower)]

    # flow_raw baseline: same dwell-derived trip construction as the main pipeline
    # but no dedup, no severance, no imputation — isolates the impact of those steps
    # from the trip-construction method itself.
    _raw = df.sort_values(['caid', 'end_timestamp']).reset_index(drop=True)
    _build_trips_in_place(_raw, apply_severance=False)
    _raw = _raw[_raw['start_timestamp'].dt.date == f_dt.date()]
    _raw = _raw.dropna(subset=['start_cbg', 'end_cbg'])
    _raw['depart_hour'] = _raw['start_timestamp'].dt.hour.astype('int8')
    od_raw = _raw.groupby(['start_cbg', 'end_cbg', 'depart_hour']).agg(
        flow_raw=('caid', 'size'),
    ).reset_index()
    del _raw

    # visits stream: pure observed-visit count, no trip construction, no inference.
    # Bucket by (destination CBG, end_timestamp.hour) on the target date — the most-
    # untouched arrival-side signal. Different schema than od (destination-only,
    # arrive_hour rather than depart_hour), so saved to a separate per-day pickle.
    _visits = df[df['end_timestamp'].dt.date == f_dt.date()]
    _visits = _visits.dropna(subset=['end_cbg'])
    od_visits = (_visits.assign(arrive_hour=_visits['end_timestamp'].dt.hour.astype('int8'))
                        .groupby(['end_cbg', 'arrive_hour'], as_index=False).size()
                        .rename(columns={'size': 'visits'}))
    od_visits['date'] = f_dt
    od_visits.to_pickle(OD_DIR / f'od_visits_{all_dates[kk]}.pkl')
    del _visits, od_visits

    df = df.drop_duplicates(subset=['caid', 'end_timestamp', 'end_place_id'])
    df = df.sort_values(by=['caid', 'end_timestamp']).reset_index(drop=True)

    # Collapse same-POI pings within PING_COLLAPSE_MIN into one visit. Keep the first
    # timestamp; aggregate minimum_dwell to the run's full extent so the dwell floor
    # used by _build_trips_in_place reflects the entire in-POI window.
    _run = (
        (df['caid'].shift(1) != df['caid']) |
        (df['end_place_id'].shift(1) != df['end_place_id']) |
        ((df['end_timestamp'] - df['end_timestamp'].shift(1)) >= pd.Timedelta(minutes=PING_COLLAPSE_MIN))
    ).cumsum()
    df['_end_ended'] = df['end_timestamp'] + pd.to_timedelta(df['minimum_dwell'], unit='m')
    df = df.groupby(_run, sort=False).agg(
        caid=('caid', 'first'),
        end_timestamp=('end_timestamp', 'first'),
        end_naics_code=('end_naics_code', 'first'),
        end_place_id=('end_place_id', 'first'),
        end_cbg=('end_cbg', 'first'),
        end_block=('end_block', 'first'),
        _end_ended=('_end_ended', 'max'),
    ).reset_index(drop=True)
    df['minimum_dwell'] = (df['_end_ended'] - df['end_timestamp']).dt.total_seconds() / 60
    df = df.drop(columns='_end_ended').sort_values(['caid', 'end_timestamp']).reset_index(drop=True)

    _build_trips_in_place(df, apply_severance=True)

    # First-of-caid rows have start_timestamp=NaT — give them a placeholder so the
    # dep_date filter below can pick them up. The placeholder never reaches OD output:
    # start_cbg stays NaN unless home imputation overwrites start_timestamp with a
    # synthetic value. Severed rows (gap_too_long=True) are left as NaT.
    need_fill = df['start_timestamp'].isnull() & (~df['gap_too_long'])
    df.loc[need_fill, 'start_timestamp'] = df.loc[need_fill, 'end_timestamp']
    df.drop(columns='gap_too_long', inplace=True)

    df['is_imputed'] = False
    date_d = f_dt.date()

    # Imputation gate: only caids with ≥ 2 visits on the target date are eligible.
    # A single-ping caid has no movement evidence; imputing 2 home trips for it is
    # pure noise and dilutes the signal from active devices (>50% of caids are N=1).
    todays_counts = df[df['end_timestamp'].dt.date == date_d].groupby('caid').size()
    active_caids = set(todays_counts[todays_counts >= 2].index)

    df['dep_date'] = False
    df.loc[(df['start_timestamp'] > (f_dt + datetime.timedelta(hours=3))) & (
            df['start_timestamp'] < (f_dt + datetime.timedelta(hours=24))) & (
            df['caid'].isin(active_caids)), 'dep_date'] = True
    # Only impute on the earliest dep_date row per caid, and only if it wasn't already
    # starting from home — otherwise we'd overwrite valid observed origins.
    first_dep = df[df['dep_date']].groupby('caid').head(1)
    first_indices_all = first_dep[first_dep['start_place_id'] != 'home'].index

    # home → first-POI synthetic travel time. caids with home > TRAVEL_MAX_DIST_KM
    # away are treated as tourists and skipped (then dropped via NaN start_cbg).
    _home_block = df.loc[first_indices_all, 'caid'].map(hw_block_map)
    _home_lat = _home_block.map(block_lat).values
    _home_lon = _home_block.map(block_lon).values
    _poi_lat, _poi_lon = _coords(df.loc[first_indices_all], 'end_place_id', 'end_block')
    _home_to_poi = _haversine_km(_home_lat, _home_lon, _poi_lat.values, _poi_lon.values)
    _impute_h = _travel_hours(_home_to_poi)
    _valid = ~np.isnan(_impute_h)
    first_indices = first_indices_all[_valid]
    _impute_h = _impute_h[_valid]

    df.loc[first_indices, 'start_place_id'] = 'home'
    # Clamp to day boundary so an early-morning arrival doesn't push start into yesterday.
    day_start = pd.Timestamp(date_d)
    synthetic_start = (df.loc[first_indices, 'end_timestamp'].values
                       - pd.to_timedelta(_impute_h, unit='h'))
    df.loc[first_indices, 'start_timestamp'] = pd.Series(synthetic_start, index=first_indices).clip(lower=day_start)
    df.loc[first_indices, 'is_imputed'] = True

    df['dep_date'] = False
    df.loc[(df['start_timestamp'] > (f_dt + datetime.timedelta(hours=3))) & (
            df['start_timestamp'] < (f_dt + datetime.timedelta(hours=24 + 3))) & (
            df['caid'].isin(active_caids)), 'dep_date'] = True
    last_trips = df[df['dep_date']].groupby('caid').last().reset_index()
    need_home_end = last_trips[last_trips['end_place_id'] != 'home'].copy()

    # last-POI → home synthetic travel time; same tourist filter as the first-POI side.
    _home_block_l = need_home_end['caid'].map(hw_block_map)
    _home_lat_l = _home_block_l.map(block_lat).values
    _home_lon_l = _home_block_l.map(block_lon).values
    _poi_lat_l, _poi_lon_l = _coords(need_home_end, 'end_place_id', 'end_block')
    _poi_to_home = _haversine_km(_poi_lat_l.values, _poi_lon_l.values, _home_lat_l, _home_lon_l)
    _impute_h_l = _travel_hours(_poi_to_home)
    _valid_l = ~np.isnan(_impute_h_l)
    need_home_end = need_home_end[_valid_l].reset_index(drop=True)
    _impute_h_l = _impute_h_l[_valid_l]

    end_corrections = pd.DataFrame({
        'caid': need_home_end['caid'].values,
        'end_place_id': 'home',
        'end_timestamp': (need_home_end['end_timestamp'].values
                          + pd.to_timedelta(_impute_h_l, unit='h')),
        'start_place_id': need_home_end['end_place_id'].values,
        'start_cbg': need_home_end['end_cbg'].values,
        'start_block': need_home_end['end_block'].values,
        'start_timestamp': need_home_end['end_timestamp'].values,
        'is_imputed': True,
    })

    df_fixed = pd.concat([df, end_corrections], ignore_index=True)
    df_fixed = df_fixed.sort_values(['caid', 'end_timestamp']).reset_index(drop=True)
    df_fixed = df_fixed[df_fixed['start_timestamp'].dt.date == date_d]

    # Backfill home rows' cbg/block from the canonical hw lookup (one home per caid).
    df_fixed = df_fixed.merge(hw, on='caid', how='left')
    df_fixed.loc[df_fixed['end_place_id'] == 'home', 'end_cbg'] = df_fixed['home_cbg']
    df_fixed.loc[df_fixed['end_place_id'] == 'home', 'end_block'] = df_fixed['home_block']
    df_fixed.loc[df_fixed['start_place_id'] == 'home', 'start_cbg'] = df_fixed['home_cbg']
    df_fixed.loc[df_fixed['start_place_id'] == 'home', 'start_block'] = df_fixed['home_block']

    # OD groupby with depart_hour for downstream weekday/weekend hourly avg.
    # flow = full pipeline; flow_raw = baseline (computed at top of loop).
    df_fixed = df_fixed.dropna(subset=['start_cbg', 'end_cbg'])
    df_fixed['depart_hour'] = df_fixed['start_timestamp'].dt.hour.astype('int8')
    od = df_fixed.groupby(['start_cbg', 'end_cbg', 'depart_hour']).agg(
        flow=('caid', 'size'),
    ).reset_index()
    od = od.merge(od_raw, on=['start_cbg', 'end_cbg', 'depart_hour'], how='outer')
    od[['flow', 'flow_raw']] = od[['flow', 'flow_raw']].fillna(0)
    od['date'] = f_dt
    od.to_pickle(OD_DIR / f'od_{all_dates[kk]}.pkl')
    del df, df_fixed, od_raw, od

# Build file lists explicitly from the loop's actually-processed date range so
# prior-run leftovers can't pollute the aggregation on a re-run with a new range.
#   od_flow_all.pkl     — one row per OD pair: monthly sums (1-main reads this)
#   od_flow_hourly.pkl  — one row per (OD pair, depart_hour, is_weekday)
#   od_hourly.pkl       — destination-level hourly trip count (back-compat)
#   od_visits_all.pkl   — destination-level pure visit count by arrive_hour and
#                         is_weekday (raw observation baseline, no inference)
processed_dates = all_dates[1:-1]
all_files = [OD_DIR / f'od_{d}.pkl' for d in processed_dates]
all_files = [f for f in all_files if f.exists()]
od_month = pd.concat([pd.read_pickle(f) for f in all_files], ignore_index=True)

od_flow = od_month.groupby(['start_cbg', 'end_cbg']).agg(
    monthly_total=('flow', 'sum'),
    monthly_total_raw=('flow_raw', 'sum'),
).reset_index()
od_flow = od_flow.rename(columns={'start_cbg': 'origin', 'end_cbg': 'destination'})
od_flow.to_pickle(OD_DIR / 'od_flow_all.pkl')

# Hourly avg by weekday vs weekend. Denominator = days that actually produced
# OD output (i.e., days present in od_month['date']) — robust to missing/skipped
# days; days with no data are excluded from both numerator and denominator.
valid_dates = pd.to_datetime(sorted(od_month['date'].unique()))
n_weekday = int((valid_dates.dayofweek < 5).sum())
n_weekend = int((valid_dates.dayofweek >= 5).sum())
od_month['is_weekday'] = od_month['date'].dt.dayofweek < 5
od_hourly_avg = od_month.groupby(
    ['start_cbg', 'end_cbg', 'depart_hour', 'is_weekday'], as_index=False
).agg(_sum_flow=('flow', 'sum'), _sum_flow_raw=('flow_raw', 'sum'))
denom = np.where(od_hourly_avg['is_weekday'], n_weekday, n_weekend)
od_hourly_avg['avg_flow'] = od_hourly_avg['_sum_flow'] / denom
od_hourly_avg['avg_flow_raw'] = od_hourly_avg['_sum_flow_raw'] / denom
od_hourly_avg = od_hourly_avg.drop(columns=['_sum_flow', '_sum_flow_raw'])
od_hourly_avg = od_hourly_avg.rename(columns={
    'start_cbg': 'origin', 'end_cbg': 'destination', 'depart_hour': 'hour',
})
od_hourly_avg.to_pickle(OD_DIR / 'od_flow_hourly.pkl')

# Destination-level hourly trip count (depart_hour basis, full pipeline `flow`).
# Output schema kept as (destination, hourly_flow, Datetime) for back-compat
# with 1-main's hourly_ratio.csv consumer.
od_hourly = od_month.groupby(['end_cbg', 'depart_hour', 'date'], as_index=False)['flow'].sum()
od_hourly['Datetime'] = od_hourly['date'] + pd.to_timedelta(od_hourly['depart_hour'], unit='h')
od_hourly = od_hourly.drop(columns=['depart_hour', 'date']).rename(
    columns={'end_cbg': 'destination', 'flow': 'hourly_flow'})
od_hourly.to_pickle(OD_DIR / 'od_hourly.pkl')

# Pure visit baseline (no inference). Destination + arrive_hour, weekday/weekend
# split. Use the same calendar denominator as od_flow_hourly so they're comparable.
visit_files = [OD_DIR / f'od_visits_{d}.pkl' for d in processed_dates]
visit_files = [f for f in visit_files if f.exists()]
od_visits_month = pd.concat([pd.read_pickle(f) for f in visit_files], ignore_index=True)
od_visits_month['is_weekday'] = od_visits_month['date'].dt.dayofweek < 5
od_visits_avg = od_visits_month.groupby(['end_cbg', 'arrive_hour', 'is_weekday'],
                                        as_index=False)['visits'].sum()
denom_v = np.where(od_visits_avg['is_weekday'], n_weekday, n_weekend)
od_visits_avg['avg_visits'] = od_visits_avg['visits'] / denom_v
od_visits_avg = od_visits_avg.rename(columns={'end_cbg': 'destination', 'arrive_hour': 'hour'})
od_visits_avg.to_pickle(OD_DIR / 'od_visits_all.pkl')

devices = hw.groupby('home_cbg')['caid'].count().reset_index()
devices.columns = ['census_block_group', 'number_devices_residing']
devices.to_pickle(OD_DIR / 'device_count.pkl')
