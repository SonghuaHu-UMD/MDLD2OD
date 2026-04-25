import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import contextily as ctx
from shapely import wkt
import mapclassify
import warnings
import numpy as np
import random
from shapely.geometry import LineString
from functools import reduce

from paths import SHP_DIR, RAW_DATA_DIR, SIMULATION_DIR, RESULTS_DIR, RESULTS_ALL_DIR

warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
pd.options.mode.chained_assignment = None
random.seed(42)
np.random.seed(42)
plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

un_st = ['02', '15', '60', '66', '69', '72', '78']
# Read CBSA Info
smart_loc = pd.read_pickle(SHP_DIR / 'SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(un_st)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')

# Select Top 100 CBSA for simulation
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

# Read socio-demo
ctract_socio = pd.read_csv(SHP_DIR / 'CTract_2022.csv', index_col=0)
ctract_socio['BGFIPS'] = ctract_socio['BGFIPS'].astype(str).apply(lambda x: x.zfill(11))


# replace the legend labels
def replace_maxmin_legend(ax):
    legend_texts = ax.get_legend().get_texts()
    n = len(legend_texts)
    for i, text_obj in enumerate(legend_texts):
        label = text_obj.get_text().strip()
        try:
            lower, upper = [int(x.strip()) for x in label.split(',')]
            if i == 0:
                new_label = f"< {upper}"
            elif i == n - 1:
                new_label = f"> {lower}"
            else:
                new_label = f"{lower}–{upper}"
            text_obj.set_text(new_label)
            text_obj.set_fontsize(12)
        except Exception as e:
            print(f"Could not parse label: {label}, error: {e}")


def plot_natural_breaks(gdf, col, out_path, k=4):
    """Plot a GeoDataFrame column with natural-breaks styling, zeros in gray."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    binning = mapclassify.NaturalBreaks(gdf[col], k=k)
    gdf = gdf.copy()
    gdf['cut_jenks'] = (binning.yb + 1) * 0.5
    gdf[gdf[col] == 0].plot(ax=ax, alpha=0.3, lw=0.25, color='gray')
    gdfr = gdf[gdf[col] > 0].reset_index(drop=True)
    gdfr.plot(column=col, cmap='RdYlGn_r', scheme="natural_breaks", k=k, lw=gdfr['cut_jenks'],
              ax=ax, alpha=0.4, legend=True,
              legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    replace_maxmin_legend(ax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def read_perf(src_dir, rename):
    """Read link_performance_s0_25nb.csv, keep/rename columns."""
    return pd.read_csv(src_dir / 'link_performance_s0_25nb.csv',
                       usecols=['link_id'] + list(rename)).rename(columns=rename)


all_bound = []
all_metric_od = []
all_metric_v = []
# Loop for each CBSA
for emsa in tqdm(range(0, 100)):
    wt_s = []
    msa_name = msa_pop.loc[emsa, 'CBSA_Name']
    msa_id = msa_pop.loc[emsa, 'CBSA']

    result_dir = RESULTS_DIR / msa_name
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = SIMULATION_DIR / 'Raw_OD' / msa_name
    weight_dir = SIMULATION_DIR / 'Weighted_OD' / msa_name
    odme_dir = SIMULATION_DIR / 'ODME' / msa_name
    final_dir = SIMULATION_DIR / 'Final' / msa_name

    # Read all MSA shp
    boundary = pd.read_csv(RAW_DATA_DIR / msa_name / 'boundary.csv', index_col=0)
    boundary['mas_name'] = msa_name
    boundary['mas_id'] = msa_id
    boundary['pop'] = msa_pop.loc[emsa, 'CBSA_POP']
    boundary['rank'] = emsa
    all_bound.append(boundary)

    link = pd.read_csv(final_dir / 'link.csv', index_col=0)
    link['mas_name'] = msa_name
    link['mas_id'] = msa_id
    link['rank'] = emsa
    link['geometry'] = link['geometry'].apply(wkt.loads)
    link = gpd.GeoDataFrame(link, geometry='geometry', crs="EPSG:4326")
    link = link.to_crs(epsg=5070)

    ### 1. Figure 1: Plot simulation road network for each CBSA ###
    fig, ax = plt.subplots(figsize=(4.5, 4))
    link.plot(column='free_speed', cmap='RdYlGn', scheme="natural_breaks", k=3, lw=link['free_speed'] / 60,
              ax=ax, alpha=0.3, legend=True, legend_kwds={"fmt": "{:.0f}", 'ncol': 1})
    ctx.add_basemap(ax, crs=link.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_dir / 'links.pdf')
    plt.close()

    ### 2. Figure 2: Comparison across different OD Tables ###
    od_raw = pd.read_csv(raw_dir / 'demand.csv', names=['o_zone_id', 'd_zone_id', 'od_raw'], header=0)
    od_weight = pd.read_csv(weight_dir / 'demand.csv', names=['o_zone_id', 'd_zone_id', 'od_weight'], header=0)
    od_final = pd.read_csv(final_dir / 'demand.csv', names=['o_zone_id', 'd_zone_id', 'od_final'], header=0)
    od_flows = reduce(lambda left, right: pd.merge(left, right, on=['o_zone_id', 'd_zone_id'], how='outer'),
                      [od_raw, od_weight, od_final])
    od_flows = od_flows.fillna(0)

    # Get OD lat lng
    zones = pd.read_csv(final_dir / 'zone_id.csv')
    zones['BGFIPS'] = zones['BGFIPS'].astype(str).apply(lambda x: x.zfill(11))
    zones['geometry'] = zones['geometry'].apply(wkt.loads)
    zones = gpd.GeoDataFrame(zones, geometry='geometry', crs="EPSG:4326")
    zones['lng'] = zones.geometry.centroid.x
    zones['lat'] = zones.geometry.centroid.y
    zones = zones.to_crs(epsg=3857)
    zones['area'] = zones.geometry.area / 1e6  # km2
    zones = zones.to_crs(epsg=5070)

    # Merge with OD flows
    t_geo_xt = zones[['zone_id', 'lng', 'lat', 'area']]
    t_geo_xt.columns = ['d_zone_id', 'd_lng', 'd_lat', 'd_area']
    od_flows = od_flows.merge(t_geo_xt, on='d_zone_id')
    t_geo_xt.columns = ['o_zone_id', 'o_lng', 'o_lat', 'o_area']
    od_flows = od_flows.merge(t_geo_xt, on='o_zone_id')
    od_flows['geometry'] = od_flows.apply(
        lambda row: LineString([(row['o_lng'], row['o_lat']), (row['d_lng'], row['d_lat'])]), axis=1)
    od_flows = gpd.GeoDataFrame(od_flows, geometry='geometry', crs="EPSG:4326")
    od_flows = od_flows.to_crs(epsg=5070)

    # Plot OD flow (density)
    cct = 0
    for od_kk in ['od_raw', 'od_weight', 'od_final', ]:
        od_flows[od_kk + '_d'] = 2 * od_flows[od_kk] / (od_flows['o_area'] + od_flows['d_area'])
        fig, ax = plt.subplots(figsize=(5, 4), nrows=1, ncols=1)
        zones.boundary.plot(ax=ax, color='gray', lw=0.1, alpha=0.1)
        demand0 = od_flows[od_flows['o_zone_id'] != od_flows['d_zone_id']]  # .sample(frac=0.1, replace=False)
        demand0 = demand0.sort_values(by=od_kk + '_d', ascending=False).head(int(len(od_flows) * 0.1)).reset_index(
            drop=True)
        binning = mapclassify.NaturalBreaks(demand0[od_kk + '_d'], k=20)
        demand0['cut_jenks_w'] = (binning.yb + 1) * 0.2
        # demand0 = demand0.sample(frac=1, random_state=42).reset_index(drop=True)
        demand0 = demand0.sort_values(by=od_kk + '_d', ascending=True).reset_index(drop=True)
        demand0.plot(ax=ax, column=od_kk + '_d', scheme="natural_breaks", k=20, linewidth=demand0['cut_jenks_w'],
                     alpha=0.3, cmap='coolwarm')
        ctx.add_basemap(ax, crs=od_flows.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
        # ax.set_title(title_name[cct])
        ax.axis('off')
        cct += 1
        plt.tight_layout()
        plt.savefig(result_dir / f'od_flows_{od_kk}.pdf')
        plt.close()

    # Plot scatter comparison: three od, od pair comparison
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.regplot(data=od_flows, x='od_raw', y='od_weight', ax=ax, color='#00A08799', scatter_kws={'alpha': 0.2, 's': 10},
                label='Weighted')
    sns.regplot(data=od_flows, x='od_raw', y='od_final', ax=ax, color='#E64B3599', scatter_kws={'alpha': 0.2, 's': 10},
                label='Adjusted')
    plt.xlabel('OD (Original)')
    plt.ylabel('OD (Revised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / f'od_comparison_{od_kk}.png', dpi=500)
    plt.close()

    # Plot PA changes
    prodt = od_flows.groupby('o_zone_id')[['od_raw', 'od_weight', 'od_final']].sum().reset_index()
    prodt['od_raw_ratio'] = prodt['od_final'] / prodt['od_raw']
    prodt['od_weight_ratio'] = prodt['od_final'] / prodt['od_weight']
    zones_p = zones.copy()
    zones_p = zones_p.merge(prodt, left_on='zone_id', right_on='o_zone_id')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    zones_p.plot(column='od_weight_ratio', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=6,
                 legend_kwds={'labelcolor': 'k', "fmt": "{:.0f}", 'ncol': 2, 'title': '', 'loc': 'lower center',
                              'frameon': False, 'facecolor': 'k', 'edgecolor': 'k', 'framealpha': 0.5}, linewidth=0,
                 edgecolor='white', alpha=0.5)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    replace_maxmin_legend(ax)
    ctx.add_basemap(ax, crs=zones_p.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.tight_layout()
    plt.savefig(result_dir / f'PA_change_{od_kk}.pdf')
    plt.close()

    # Merge with socio-demo
    zones_p = zones_p.merge(ctract_socio, on='BGFIPS')
    zones_p['Population_Density'] = zones_p['Total_Population'] / zones_p['area']
    # zones_p.corr(numeric_only=True).to_csv(r'corr.csv')
    corrs = zones_p[['Population_Density', 'od_weight_ratio']].corr().values[0][1]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.regplot(data=zones_p, x='Population_Density', y='od_weight_ratio', ax=ax, color='orange',
                scatter_kws={'alpha': 0.2, 's': 10}, label=r'$\rho$=%s' % round(corrs, 3))
    plt.xlabel('Population Density')
    plt.ylabel('OD Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / f'od_with_popdensity_{od_kk}.png', dpi=500)
    plt.close()

    ### 3. Plot traffic volume: Raw; Weighted; ODME; AADT ###
    link_raw = read_perf(raw_dir, {'volume': 'volume_raw'})
    link_weight = read_perf(weight_dir, {'volume': 'volume_weight'})
    link_final = read_perf(odme_dir, {'ODME_volume_before': 'volume_before',
                                      'ODME_volume_after': 'volume_final'})
    link_flows = reduce(lambda left, right: pd.merge(left, right, on=['link_id'], how='outer'),
                        [link, link_raw, link_weight, link_final])
    valid_linkss = pd.read_csv(RAW_DATA_DIR / msa_name / 'valid_linkss.csv', index_col=0)
    link_flows = link_flows.merge(valid_linkss, on='link_id', how='left')
    link_flows.to_pickle(result_dir / 'link_flows.pkl')

    for plt_name in ['volume_raw', 'volume_weight', 'volume_final']:
        plot_natural_breaks(link_flows, plt_name, result_dir / f'link_volume_{plt_name}.pdf')

    # Plot AADT
    aadt = pd.read_pickle(RAW_DATA_DIR / msa_name / 'all_aadt.pkl').to_crs(epsg=5070)
    plot_natural_breaks(aadt, 'AADT_hour', result_dir / 'link_volume_aadt.pdf')

    # Plot ODME
    fig, ax = plt.subplots(figsize=(4.5, 4))
    aadt_nn = link_flows[~link_flows['AADT_hour'].isnull()]
    for y_col, color, prefix in [('volume_before', '#00A08799', 'Before'),
                                 ('volume_final', '#E64B3599', 'After')]:
        rho = round(aadt_nn[[y_col, 'AADT_hour']].corr().values[1][0], 2)
        sns.regplot(data=aadt_nn, x='AADT_hour', y=y_col, ax=ax, color=color,
                    label=f'{prefix}: ' + r'$\rho=$' + str(rho),
                    scatter_kws={'alpha': 0.5, 's': 10})
    ax.plot([0, max(aadt_nn['volume_final'])], [0, max(aadt_nn['volume_final'])], '--', lw=2, color='k')
    plt.xlabel('Ground truth')
    plt.ylabel('Assignment volume')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(result_dir / 'volume_before_after.png', dpi=500)
    plt.close()

    # Summary all features: OD difference and Volume accuracy
    for each_v in ['volume_raw', 'volume_weight', 'volume_final']:
        aadt_nn['mae'] = abs(aadt_nn[each_v] - aadt_nn['AADT_hour'])
        aadt_nn['mape'] = abs(aadt_nn[each_v] - aadt_nn['AADT_hour']) / aadt_nn['AADT_hour']
        tt_vmt_mape = ((link_flows[each_v] * link_flows['length']).sum() - (
                aadt['AADT_hour'] * aadt['aadt_length']).sum()) / ((aadt['AADT_hour'] * aadt['aadt_length']).sum())
        tt_volume_mape = (aadt_nn[each_v].sum() - aadt_nn['AADT_hour'].sum()) / aadt_nn['AADT_hour'].sum()
        l_avg_volume_mape = aadt_nn['mape'].mean()
        l_avg_volume_mae = aadt_nn['mae'].mean()
        l_volume_corr = aadt_nn[[each_v, 'AADT_hour']].corr().values[1][0]
        all_metric_v.append(
            [emsa, msa_name, msa_id, tt_vmt_mape, tt_volume_mape, l_avg_volume_mape, l_avg_volume_mae, l_volume_corr])

    tt_od_fr = (od_flows['od_final'].sum() - od_flows['od_raw'].sum()) / od_flows['od_raw'].sum()
    tt_od_fw = (od_flows['od_final'].sum() - od_flows['od_weight'].sum()) / od_flows['od_weight'].sum()
    tt_od_wr = (od_flows['od_weight'].sum() - od_flows['od_raw'].sum()) / od_flows['od_raw'].sum()
    od_flows['od_final_raw'] = abs(od_flows['od_final'] - od_flows['od_raw']) / od_flows['od_raw']
    od_flows['od_weight_raw'] = abs(od_flows['od_weight'] - od_flows['od_raw']) / od_flows['od_raw']
    od_flows['od_final_weight'] = abs(od_flows['od_final'] - od_flows['od_weight']) / od_flows['od_weight']
    all_metric_od.append([emsa, msa_name, msa_id, tt_od_fr, tt_od_fw, tt_od_wr, od_flows['od_final_raw'].mean(),
                          od_flows['od_weight_raw'].mean(), od_flows['od_final_weight'].mean()])
    od_flows.to_pickle(result_dir / 'od_flows.pkl')

all_metric_od = pd.DataFrame(all_metric_od)
all_metric_od.columns = ['emsa', 'msa_name', 'msa_id', 'tt_od_fr', 'tt_od_fw', 'tt_od_wr', 'od_final_raw',
                         'od_weight_raw', 'od_final_weight']
all_metric_od.to_csv(RESULTS_ALL_DIR / 'all_metric_od.csv')
all_metric_v = pd.DataFrame(all_metric_v)
all_metric_v.columns = ['emsa', 'msa_name', 'msa_id', 'tt_vmt_mape', 'tt_volume_mape', 'l_avg_volume_mape',
                        'l_avg_volume_mae', 'l_volume_corr']
all_metric_v['case'] = ['volume_raw', 'volume_weight', 'volume_final'] * 100
all_metric_v.to_csv(RESULTS_ALL_DIR / 'all_metric_v.csv')

# # Plot all areas
all_bound = pd.concat(all_bound, axis=0).reset_index(drop=True)
all_bound['geometry'] = all_bound['geometry'].apply(wkt.loads)
all_bound = gpd.GeoDataFrame(all_bound, geometry='geometry', crs="EPSG:4326")
all_bound = all_bound.to_crs(epsg=5070)
# all_bound.to_csv(SHP_DIR / 'all_bound.csv')
fig, ax = plt.subplots(figsize=(9, 6))
all_bound.plot(column='pop', ax=ax, alpha=0.7, cmap='coolwarm', scheme="natural_breaks", k=5)
ctx.add_basemap(ax, crs=all_bound.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
plt.tight_layout()
plt.axis('off')
plt.savefig(RESULTS_ALL_DIR / 'all_bound.pdf')

# merge with msa features
all_metric_v_final = all_metric_v[all_metric_v['case'] == 'volume_final'].merge(
    msa_pop, left_on='msa_id', right_on='CBSA')
all_metric_od_final = all_metric_od.merge(msa_pop, left_on='msa_id', right_on='CBSA')
for data, ycol in [(all_metric_v_final, 'tt_volume_mape'),
                   (all_metric_od_final, 'od_final_weight')]:
    fig, ax = plt.subplots(figsize=(9, 6))
    rho = round(data[['CBSA_POP', ycol]].corr().values[1][0], 2)
    sns.regplot(data=data, x='CBSA_POP', y=ycol, ax=ax, color='#00A08799',
                scatter_kws={'alpha': 0.5, 's': 50}, label=r'$\rho=$' + str(rho))
    plt.legend(loc='upper left')
