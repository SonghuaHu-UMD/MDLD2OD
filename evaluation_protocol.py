"""Route-group holdout, calibration-only weights, and measured traffic errors."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = 'route_holdout_v1'


def assign_holdout(observations, fraction=.2, seed=42):
    result = observations.copy()
    route = result.Route_ID.astype('string').str.strip()
    if route.isna().any() or route.eq('').any():
        raise ValueError('A verified Route_ID is required for group holdout; missing routes cannot be treated as independent links')
    groups = sorted(route.unique(), key=lambda v: hashlib.sha256(f'{seed}:{v}'.encode()).hexdigest())
    if len(groups) < 2 or not 0 < fraction < 1:
        raise ValueError('At least two route groups and a proper holdout fraction are required')
    count = min(len(groups)-1, max(1, round(len(groups) * fraction)))
    result['observation_group'] = route
    result['evaluation_split'] = np.where(route.isin(groups[:count]), 'holdout', 'calibration')
    result['evaluation_protocol'] = PROTOCOL
    return result


def calibration_weights(frame, p_ratio):
    subset = frame[(frame.evaluation_split == 'calibration') & np.isfinite(frame.AADT) & (frame.AADT > 0)].copy()
    if subset.empty or not np.isfinite(p_ratio) or p_ratio <= 0:
        raise ValueError('No valid calibration observations')
    prediction = subset.volume.to_numpy(float)
    length = subset.distance_km.to_numpy(float)
    if not np.isfinite(prediction).all() or not np.isfinite(length).all() or (length <= 0).any() or (prediction < 0).any():
        raise ValueError('Invalid calibration prediction or length')
    truth = subset.AADT.to_numpy(float) * p_ratio
    volume = prediction.sum()
    vmt = (prediction * length).sum()
    if volume <= 0 or vmt <= 0:
        raise ValueError('No predicted calibration traffic; ratio is undefined')
    return float((truth * length).sum() / vmt), float(truth.sum() / volume), float(abs(truth - prediction).mean())


def traffic_metrics(frame, column):
    observed = frame.AADT_hour.to_numpy(float)
    valid = np.isfinite(observed) & (observed >= 0)
    predicted = frame[column].to_numpy(float)
    if np.any(valid & ~np.isfinite(predicted)):
        raise ValueError('Missing predictions at measured links')
    truth, pred = observed[valid], predicted[valid]
    length = frame.length.to_numpy(float)[valid]
    if not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError('Invalid measured-link lengths')
    positive = truth > 0
    error = abs(pred - truth)
    obs_vmt = (truth * length).sum()
    return dict(total_links=len(frame), observed_links=int(valid.sum()), mape_links=int(positive.sum()),
        zero_observation_links=int((truth == 0).sum()), excluded_missing_observations=int((~valid).sum()),
        tt_volume_bias=float((pred.sum()-truth.sum()) / truth.sum()) if truth.sum() else np.nan,
        tt_vmt_bias=float(((pred-truth) * length).sum() / obs_vmt) if obs_vmt else np.nan,
        l_avg_volume_mae=float(error.mean()) if len(error) else np.nan,
        l_avg_volume_mape=float((error[positive] / truth[positive]).mean()) if positive.any() else np.nan,
        l_volume_corr=float(np.corrcoef(truth, pred)[0, 1]) if len(truth)>1 and np.std(truth)>0 and np.std(pred)>0 else np.nan)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_run_manifest(directory, catalog):
    directory = Path(directory)
    metadata = dict(protocol=PROTOCOL, catalog_sha256=digest(catalog),
                    performance_sha256=digest(directory / 'link_performance_s0_25nb.csv'),
                    demand_sha256=digest(directory / 'demand.csv'))
    sensor = directory / 'sensor_data.csv'
    metadata['sensor_sha256'] = digest(sensor) if sensor.exists() else None
    (directory / 'evaluation_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def validate_run(directory, catalog):
    directory = Path(directory)
    path = directory / 'evaluation_manifest.json'
    if not path.exists():
        raise ValueError('Legacy simulation lacks route-holdout provenance; regenerate sensors and rerun DTALite')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if metadata.get('protocol') != PROTOCOL or metadata.get('catalog_sha256') != digest(catalog):
        raise ValueError('Simulation and holdout catalog differ; rerun the simulation')
    for key, file in [('performance', 'link_performance_s0_25nb.csv'), ('demand', 'demand.csv')]:
        if metadata.get(key + '_sha256') != digest(directory / file):
            raise ValueError('Simulation output provenance mismatch')
    sensor = directory / 'sensor_data.csv'
    if metadata.get('sensor_sha256') != (digest(sensor) if sensor.exists() else None):
        raise ValueError('Calibration sensors have changed')
    return metadata
