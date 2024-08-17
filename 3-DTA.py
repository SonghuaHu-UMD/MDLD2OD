import glob
import pandas as pd
import numpy as np
import matplotlib
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import subprocess
import mapclassify
import contextily as ctx

url_r = r'D:\MDLD_OD\Roadosm\\'
all_files = glob.glob(url_r + '*.pbf')
out_files = os.listdir(url_r)

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
# Read CBG-MSA
smart_loc = pd.read_pickle(r'F:\Research_Old\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(un_st)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')
# Select Top 50 MSA
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

# Link OD data to road network
for ef in all_files:
    # ef=all_files[-1]
    e_cbsa = ef.split('\\')[-1].split('.')[0]
    e_name = msa_pop.loc[msa_pop['CBSA'] == e_cbsa, 'CBSA_Name'].values[0]
    print('Start processing %s--------------' % e_name)
    msa_need = MSA_geo[MSA_geo['CBSAFP'] == e_cbsa]
    msa_need = msa_need.to_crs('EPSG:4326')

    node = pd.read_csv(url_r + e_cbsa + '.pbf_node.csv')
    link = pd.read_csv(url_r + e_cbsa + '.pbf_link.csv')
    link = link[link['link_type_name'].isin(
        ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'connector'])].reset_index(drop=True)

    # check link and node: all link's node should be found in node.csv
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    node_node = set(node['node_id'])
    print('Pct of nodes in links: %s' % (len(link_node & node_node) / len(link_node)))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)
    # check link type
    link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()

    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord), crs="EPSG:4326")
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    # read od we need
    od_raw = pd.read_csv('D:\MDLD_OD\MDLDod\data\\%s_OD.csv' % e_name, index_col=0)
    od_raw['destination'] = od_raw['destination'].astype(str).apply(lambda x: x.zfill(12))
    od_raw['origin'] = od_raw['origin'].astype(str).apply(lambda x: x.zfill(12))
    # od_raw['monthly_total'] = od_raw['monthly_total'] / (31 * 4)
    # od_raw = od_raw[od_raw['monthly_total'] > 0.1].reset_index(drop=True)
    cbg_list = set(od_raw['destination']).union(set(od_raw['origin']))
    print('Number of zones: %s' % len(cbg_list))

    # change zone id
    zone_ids = pd.DataFrame({'destination': list(cbg_list), 'd_zone_id': range(0, len(cbg_list))})
    od_raw = od_raw.merge(zone_ids, on='destination')
    zone_ids.columns = ['origin', 'o_zone_id']
    od_raw = od_raw.merge(zone_ids, on='origin')
    od_raw = od_raw.drop(['destination', 'origin'], axis=1)

    cbg_need = CBG_geo[CBG_geo['BGFIPS'].isin(cbg_list)].reset_index(drop=True)
    cbg_need = cbg_need.to_crs('EPSG:4326')

    # node and CBG join: assign zone id (CBG) to node; connect to link with the highest class
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

    # fig, ax = plt.subplots(figsize=(9, 7))
    # link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    # msa_need.boundary.plot(ax=ax, color='royalblue')
    # node[~node['zone_id'].isnull()].plot(ax=ax, markersize=10, color='red', alpha=1)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # output
    Path(r"D:\MDLD_OD\Simulation\%s" % e_cbsa).mkdir(parents=True, exist_ok=True)
    shutil.copy2(r'D:\MDLD_OD\DTALite_230915.exe', r"D:\MDLD_OD\Simulation\%s" % e_cbsa)

    # Generate demand period
    t_p = 'am'
    demand_period = pd.DataFrame(
        {'first_column': [0], "demand_period_id": 1, "demand_period": t_p, "notes": 'weekday',
         "time_period": '0600_0900', "peak_time": '0830'})
    demand_period.to_csv(r"D:\MDLD_OD\Simulation\%s\demand_period.csv" % e_cbsa, index=False)
    demand_file_list = pd.DataFrame(
        {'first_column': [0], "file_sequence_no": 1, "scenario_index_vector": 0, "file_name": "demand.csv",
         "demand_period": t_p, "mode_type": 'auto', "format_type": "column", "scale_factor": 1,
         "departure_time_profile_no": 1})
    demand_file_list.to_csv(r"D:\MDLD_OD\Simulation\%s\demand_file_list.csv" % e_cbsa, index=False)

    # demand = od_raw[od_raw['monthly_total'] > 1].reset_index(drop=True)
    od_raw.columns = ['volume', 'd_zone_id', 'o_zone_id']
    od_raw[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"D:\MDLD_OD\Simulation\%s\demand.csv" % e_cbsa, index=False)

    node.to_csv(r"D:\MDLD_OD\Simulation\%s\node.csv" % e_cbsa, index=False)
    link.to_csv(r"D:\MDLD_OD\Simulation\%s\link.csv" % e_cbsa, index=False)

    # Run assignment
    os.chdir(r"D:\MDLD_OD\Simulation\%s" % e_cbsa)
    subprocess.call([r"D:\MDLD_OD\Simulation\%s\DTALite_230915.exe" % e_cbsa])

    # Plot link performance
    assign_all = pd.read_csv(r'D:\MDLD_OD\Simulation\%s\link_performance_s0_25nb.csv' % e_cbsa)
    assign_all['volume'] = assign_all['volume'].fillna(0)
    binning = mapclassify.NaturalBreaks(assign_all['volume'], k=5)  # NaturalBreaks
    assign_all['cut_jenks'] = (binning.yb + 1)
    aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'volume']],
                      on=['from_node_id', 'to_node_id'], how='left')

    fig, ax = plt.subplots(figsize=(9, 7))
    aadt.plot(column='volume', cmap='RdYlGn_r', scheme="natural_breaks", k=5, lw=aadt['cut_jenks'], ax=ax,
              alpha=0.6, legend=True, legend_kwds={"fmt": "{:.0f}", 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'D:\MDLD_OD\Simulation\%s\assigned_volume.pdf' % e_cbsa)
    # plt.close()
