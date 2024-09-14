import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import glob
import contextily as ctx
import matplotlib as mpl
import glob
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess
import mapclassify
import yaml

plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
data_url = r'G:\Data\Dewey\Advan\Neighborhood Patterns - US\\'
url_r = r'D:\MDLD_OD\Roadosm\\'
all_files = glob.glob(url_r + '*.pbf')


def plot_od(demand0, boundary_plot, plot_name, o_x, o_y, d_x, d_y):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    boundary_plot.boundary.plot(ax=ax, color='gray', lw=0.2)
    for kk in range(0, len(demand0)):
        ax.annotate('', xy=(demand0.loc[kk, o_x], demand0.loc[kk, o_y]),
                    xytext=(demand0.loc[kk, d_x], demand0.loc[kk, d_y]),
                    arrowprops={'arrowstyle': '->', 'lw': 5 * demand0.loc[kk, plot_name] / max(demand0[plot_name]),
                                'color': 'royalblue', 'alpha': 0.5, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    plt.xlim(demand0[o_x].min(), demand0[o_x].max())
    plt.ylim(demand0[o_y].min(), demand0[o_y].max())
    ctx.add_basemap(ax, crs=boundary_plot.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.tight_layout()


def save_settings_yml(filename, assignment_settings, mode_types, demand_periods, demand_files, subarea, link_types,
                      departure_time_profiles):
    settings = {'assignment': assignment_settings, 'mode_types': mode_types, 'demand_periods': demand_periods,
                'demand_files': demand_files, 'subarea': subarea, 'link_types': link_types,
                'departure_time_profile': departure_time_profiles}
    with open(filename, 'w') as file:
        yaml.dump(settings, file)


# Read MSA
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\Google_Review\Parking\tl_2019_us_cbsa\tl_2019_us_cbsa.shp')
# Read CBG
un_st = ['02', '15', '60', '66', '69', '72', '78']
CBG_geo = gpd.GeoDataFrame.from_file(
    r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_blck_grp_2019_84.shp')
CBG_geo['BGFIPS'] = CBG_geo['GEOID']
CBG_geo = CBG_geo[~CBG_geo['GISJOIN'].str[1:3].isin(un_st)].reset_index(drop=True)
# Read State
state = pd.read_csv(r'D:\MDLD_OD\Others\us-state-ansi-fips.csv')
state['st'] = state[' st'].astype(str).apply(lambda x: x.zfill(2))
state['stusps'] = state[' stusps'] + '.'
state['stusps'] = state['stusps'].str.replace(' ', '')

# Read CBG-MSA
smart_loc = pd.read_pickle(r'F:\Research_Old\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(un_st)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')
# Select Top 50 MSA
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

# Generate OD
# all_files = glob.glob(data_url + '*DATE_RANGE_START-2019-05-01.csv.gz')
all_files = [r'D:\MDLD_OD\Others\19100_20220408.csv', ]
emsa = 3
msa_name = msa_pop.loc[emsa, 'CBSA_Name']
print("------------------- Start processing MSA: %s -----------------" % msa_name)
need_cbg = list(smart_loc.loc[smart_loc['CBSA_Name'] == msa_name, 'BGFIPS'])
od_flows = pd.DataFrame()
for file in tqdm(all_files):
    # Read OD flow: monthly
    ng_pattern = pd.read_csv(file, index_col=0)
    replace_s = pd.Series(state.st.values, index=state.stusps).to_dict()
    replace_s['US.'] = ''
    replace_s['\.'] = ''
    ng_pattern['start_block_group_id'] = ng_pattern['start_block_group_id'].replace(replace_s, regex=True)
    ng_pattern['end_block_group_id'] = ng_pattern['end_block_group_id'].replace(replace_s, regex=True)
    # ng_pattern = ng_pattern[ng_pattern['provider_id'] == 190199]
    ng_pattern = ng_pattern[ng_pattern['hour'] == 9]
    od_flow = ng_pattern[(ng_pattern['start_block_group_id'].isin(need_cbg)) & (
        ng_pattern['end_block_group_id'].isin(need_cbg))].reset_index(drop=True)
    od_flow = od_flow[['end_block_group_id', 'start_block_group_id', 'total_trips']]
    od_flows = pd.concat([od_flows, od_flow])

od_flows.columns = ['destination', 'origin', 'monthly_total']
od_flows[['destination', 'origin', 'monthly_total']].to_csv(r'D:\MDLD_OD\Test\MDLDod\data\%s_OD.csv' % msa_name)

# Link OD data to road network
e_cbsa = msa_pop.loc[emsa, 'CBSA']
e_name = msa_pop.loc[msa_pop['CBSA'] == e_cbsa, 'CBSA_Name'].values[0]
print('Start processing %s--------------' % e_name)
msa_need = MSA_geo[MSA_geo['CBSAFP'] == e_cbsa]
msa_need = msa_need.to_crs('EPSG:4326')

node = pd.read_csv(url_r + e_cbsa + '.pbf_node.csv')
link = pd.read_csv(url_r + e_cbsa + '.pbf_link.csv')
link = link[link['link_type_name'].isin(
    ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'connector'])].reset_index(drop=True)

# All link's node should be found in node.csv
link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
node_node = set(node['node_id'])
# print('Pct of nodes in links: %s' % (len(link_node & node_node) / len(link_node)))
node = node[node['node_id'].isin(link_node)].reset_index(drop=True)

# To geopandas
node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord), crs="EPSG:4326")
link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

# Read od we need
od_raw = pd.read_csv('D:\MDLD_OD\Test\MDLDod\data\\%s_OD.csv' % e_name, index_col=0)
od_raw['destination'] = od_raw['destination'].astype(str).apply(lambda x: x.zfill(12))
od_raw['origin'] = od_raw['origin'].astype(str).apply(lambda x: x.zfill(12))
# od_raw['monthly_total'] = (od_raw['monthly_total'] / 31) * 0.1
# od_raw = od_raw[od_raw['monthly_total'] > 0.1].reset_index(drop=True)
cbg_list = set(od_raw['destination']).union(set(od_raw['origin']))
print('Number of zones: %s' % len(cbg_list))

# Change zone id from CBFIPS to int
zone_ids = pd.DataFrame({'destination': list(cbg_list), 'd_zone_id': range(0, len(cbg_list))})
od_raw = od_raw.merge(zone_ids, on='destination')
zone_ids.columns = ['origin', 'o_zone_id']
od_raw = od_raw.merge(zone_ids, on='origin')
od_raw = od_raw.drop(['destination', 'origin'], axis=1)
od_raw.columns = ['volume', 'd_zone_id', 'o_zone_id']

# Node and CBG join: assign zone id (CBG) to node; connect to link with the highest class
cbg_need = CBG_geo[CBG_geo['BGFIPS'].isin(cbg_list)].reset_index(drop=True)
cbg_need = cbg_need.to_crs('EPSG:4326')
SInBG = gpd.sjoin(node, cbg_need, how='inner', predicate='within').reset_index(drop=True)
SInBG_index = SInBG[['node_id', 'BGFIPS']]
node_speed = link.groupby('from_node_id')[['free_speed', 'capacity']].mean().reset_index()
node_speed.columns = ['node_id', 'node_speed', 'node_cap']
SInBG_index = SInBG_index.merge(node_speed, on='node_id', how='left')
idx = SInBG_index.groupby(['BGFIPS'])['node_speed'].transform('max') == SInBG_index['node_speed']
SInBG_index1 = SInBG_index[idx].reset_index(drop=True)
idx = SInBG_index1.groupby(['BGFIPS'])['node_cap'].transform('max') == SInBG_index1['node_cap']
SInBG_index2 = SInBG_index1[idx]
SInBG_indexf = SInBG_index2.groupby('BGFIPS').sample(n=1)[['node_id', 'BGFIPS']]
node = node.merge(SInBG_indexf, on='node_id', how='left')
node = node.drop('zone_id', axis=1)
zone_ids.columns = ['BGFIPS', 'zone_id']
node = node.merge(zone_ids, on='BGFIPS', how='left')
node = node.drop('BGFIPS', axis=1)

# Generate setting for DTALite
assignment_settings = {'number_of_iterations': 20, 'route_output': 0, 'simulation_output': 0,
                       'number_of_cpu_processors': 6, 'length_unit': 'meter', 'speed_unit': 'kmh',
                       'UE_convergence_percentage': 0.001, 'odme_activate': 0}
mode_types = [{'mode_type': 'auto', 'vot': 10, 'person_occupancy': 1, 'pce': 1}]
demand_periods = [{'period': 'AM', 'time_period': '0700_0800'}]
demand_files = [{'file_sequence_no': 1, 'file_name': 'demand.csv', 'demand_period': 'am', 'mode_type': 'auto',
                 'format_type': 'column', 'scale_factor': 1, 'departure_time_profile_no': 1}]
subarea = [{'activate': 0, 'subarea_geometry': 'POLYGON ((-180 -90, 180 -90, 180 90, -180 90,-180 -90))'}]
departure_time_profiles = [
    {'departure_time_profile_no': 1, 'time_period': '0700_0800', 'T0420': 0.005002, 'T0425': 0.005020,
     'T0430': 0.005002, 'T0435': 0.005207, 'T0440': 0.005207, 'T0445': 0.005207, 'T0450': 0.005677,
     'T0455': 0.005677, 'T0460': 0.005677, 'T0465': 0.005994, 'T0470': 0.005994, 'T0475': 0.005994,
     'T0480': 0.006018}]
link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()
link_type['traffic_flow_model'] = ['kw', 'spatial_queue', 'spatial_queue', 'point_queue', 'point_queue']
link_type.columns = ['link_type', 'link_type_name', 'free_speed_auto', 'capacity_auto', 'traffic_flow_model']
link_types = link_type.to_dict(orient='records')

# Output
Path(r"D:\MDLD_OD\Test\Simulation\%s" % e_cbsa).mkdir(parents=True, exist_ok=True)
# shutil.copy2(r'D:\MDLD_OD\DTALite_230915.exe', r"D:\MDLD_OD\Simulation\%s" % e_cbsa)
shutil.copy2(r'D:\MDLD_OD\DTALite_0602_2024.exe', r"D:\MDLD_OD\Test\Simulation\%s" % e_cbsa)
save_settings_yml(r"D:\MDLD_OD\Test\Simulation\%s\settings.yml" % e_cbsa, assignment_settings, mode_types,
                  demand_periods, demand_files, subarea, link_types, departure_time_profiles)
od_raw[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"D:\MDLD_OD\Test\Simulation\%s\demand.csv" % e_cbsa, index=False)
node.to_csv(r"D:\MDLD_OD\Test\Simulation\%s\node.csv" % e_cbsa, index=False)
link.to_csv(r"D:\MDLD_OD\Test\Simulation\%s\link.csv" % e_cbsa, index=False)

# Run assignment
os.chdir(r"D:\MDLD_OD\Test\Simulation\%s" % e_cbsa)
subprocess.call([r"D:\MDLD_OD\Test\Simulation\%s\DTALite_0602_2024.exe" % e_cbsa])

# Plot link performance
assign_all = pd.read_csv(r'D:\MDLD_OD\Test\Simulation\%s\link_performance.csv' % e_cbsa)
assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)
binning = mapclassify.NaturalBreaks(assign_all['vehicle_volume'], k=5)  # NaturalBreaks
assign_all['cut_jenks'] = (binning.yb + 1) * 0.5
aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'vehicle_volume', 'speed_kmph']],
                  on=['from_node_id', 'to_node_id'], how='left')

fig, ax = plt.subplots(figsize=(9, 7))
aadt.plot(column='vehicle_volume', cmap='RdYlGn_r', scheme="natural_breaks", k=5, lw=aadt['cut_jenks'], ax=ax,
          alpha=0.6, legend=True, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
# plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
plt.tight_layout()
plt.axis('off')
plt.savefig(r'D:\MDLD_OD\Test\Simulation\%s\assigned_traffic.pdf' % e_cbsa)
plt.close()
