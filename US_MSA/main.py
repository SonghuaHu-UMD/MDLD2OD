import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import glob
import contextily as ctx
import osmnx as ox
from pathlib import Path
import osm2gmns as og
import shutil
import subprocess
import mapclassify
import os
import yaml


def save_settings_yml(filename, assignment_settings, mode_types, demand_periods, demand_files, subarea, link_types,
                      departure_time_profiles):
    settings = {'assignment': assignment_settings, 'mode_types': mode_types, 'demand_periods': demand_periods,
                'demand_files': demand_files, 'subarea': subarea, 'link_types': link_types,
                'departure_time_profile': departure_time_profiles}
    with open(filename, 'w') as file:
        yaml.dump(settings, file)


# set default link features
default_lanes_dict = {'motorway': 4, 'trunk': 3, 'primary': 3, 'secondary': 2, 'tertiary': 2, 'residential': 1,
                      'unclassified': 1, 'connector': 2}
default_speed_dict = {'motorway': 120, 'trunk': 100, 'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 30,
                      'unclassified': 30, 'connector': 120}
default_capacity_dict = {'motorway': 2300, 'trunk': 2200, 'primary': 1800, 'secondary': 1600, 'tertiary': 1200,
                         'residential': 1000, 'unclassified': 800, 'connector': 9999}
defaults_all = pd.DataFrame([default_lanes_dict, default_speed_dict, default_capacity_dict]).T
defaults_all = defaults_all.reset_index()
defaults_all.columns = ['link_type_name', 'lanes_default', 'speed_default', 'capacity_default']

############# 1. Read raw shapefile and regional data #############
# Read MSA
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\MDLD_OD\MDLDod\shp\tl_2019_us_cbsa.shp')
# Read CT
CT_geo = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\poly_ct_84.pkl')
CT_geo['centroid_lon'] = CT_geo.centroid.x
CT_geo['centroid_lat'] = CT_geo.centroid.y
# Read CBG
un_st = ['02', '15', '60', '66', '69', '72', '78']
CBG_geo = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\poly_tract_84.pkl')
CBG_geo['BGFIPS'] = CBG_geo['GEOID']
CBG_geo = CBG_geo[~CBG_geo['GISJOIN'].str[1:3].isin(un_st)].reset_index(drop=True)
# Read CBG+MSA
smart_loc = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(un_st)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')
# Select Top 100 CBSA
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)
# Read device count
devices = pd.read_csv(r'D:\MDLD_OD\MDLDod\shp\device_2019_05.csv')
devices = devices[['census_block_group', 'number_devices_residing']]
devices.columns = ['BGFIPS', 'devices']
devices['BGFIPS'] = devices['BGFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
devices = devices.merge(smart_loc[['BGFIPS', 'TotPop']], on='BGFIPS')
devices['devices_ratio'] = devices['devices'] / devices['TotPop']  # devices.corr()
# plt.plot(devices['devices'], devices['TotPop'],'o')
device_ratio = devices[['BGFIPS', 'devices_ratio']]

############# 2. Prepare simulation data and run for each CBSA #############
url_r = r'D:\MDLD_OD\MDLDod\simulation'
all_od_files = glob.glob(r'G:\Data\Dewey\Advan\Neighborhood Patterns - US\\*DATE_RANGE_START-2019-05-01.csv.gz')
# Loop for each CBSA
for emsa in range(0, 100):
    msa_name = msa_pop.loc[emsa, 'CBSA_Name']
    print("------------------- Start processing MSA: %s -----------------" % msa_name)

    # Get MSA center by population density
    smart_loc_m = smart_loc.loc[smart_loc['CBSA_Name'] == msa_name, ['BGFIPS', 'TotPop', 'Ac_Land']]
    smart_loc_m['CTFIPS'] = smart_loc['BGFIPS'].str[0:5]
    smart_loc_m = smart_loc_m.groupby(['CTFIPS']).sum().reset_index()
    smart_loc_m['PopDes'] = smart_loc_m['TotPop'] / smart_loc_m['Ac_Land']
    smart_loc_m = smart_loc_m.sort_values(by='PopDes', ascending=False).reset_index(drop=True)
    CT_geo_m = CT_geo.loc[CT_geo['GEOID'] == smart_loc_m.head(1)['CTFIPS'].item(), :]
    msa_center = (CT_geo_m['centroid_lat'].item(), CT_geo_m['centroid_lon'].item())

    # # Get network for the whole MSA
    # msa_e_geo = MSA_geo.loc[MSA_geo['NAME'] == msa_name, 'geometry'].item()
    # # msa_e_geo.plot()
    # print('-------------- %s: Downloading Network--------------' % msa_name)
    # G = ox.graph.graph_from_polygon(msa_e_geo, network_type="drive")
    # print('No of edges: %s' % G.number_of_edges())
    # ox.io.save_graph_xml(G, filepath=r'D:\MDLD_OD\MDLDod\network\%s_whole.osm' % msa_name)

    # Get network from MSA center
    print('-------------- %s: Downloading Network--------------' % msa_name)
    G = ox.graph.graph_from_point(msa_center, dist=48.2803 * 1000, dist_type='network', network_type="drive")  # 30 mile
    print('No of edges: %s' % G.number_of_edges())
    edge_gpd = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    node_gpd = ox.convert.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    ox.io.save_graph_xml(G, filepath=r'D:\MDLD_OD\MDLDod\network\%s.osm' % msa_name)

    # Convert the simulation network using osm2gmns
    print('-------------- %s: Converting Network--------------' % msa_name)
    net = og.getNetFromFile(r'D:\MDLD_OD\MDLDod\network\%s.osm' % msa_name)
    Path(r"%s\%s" % (url_r, msa_name)).mkdir(parents=True, exist_ok=True)
    og.outputNetToCSV(net, output_folder=r"%s\%s" % (url_r, msa_name))

    # Get OD data
    SInBG = gpd.sjoin(node_gpd, CBG_geo, how='inner', predicate='within').reset_index(drop=True)
    SInBG_index = SInBG[['osmid', 'BGFIPS']]
    need_cbg = set(SInBG_index['BGFIPS'])
    print(len(need_cbg))
    od_flows = []
    for file in tqdm(all_od_files):
        # Read OD flow
        ng_pattern = pd.read_csv(file)
        ng_pattern = ng_pattern.dropna(subset=['AREA']).reset_index(drop=True)
        ng_pattern = ng_pattern[~ng_pattern['AREA'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        ng_pattern['AREA'] = ng_pattern['AREA'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        ng_pattern = ng_pattern[(ng_pattern['AREA'].str[0:11].isin(need_cbg))].reset_index(drop=True)

        # Get monthly OD flow
        ng_pattern['DEVICE_HOME_AREAS'] = ng_pattern['DEVICE_HOME_AREAS'].apply(ast.literal_eval).reset_index(drop=True)
        d_explode = pd.DataFrame([*ng_pattern['DEVICE_HOME_AREAS']], ng_pattern.index).stack() \
            .rename_axis([None, 'Origin']).reset_index(1, name='Flow')
        od_flow = ng_pattern[['AREA']].join(d_explode)
        od_flow = od_flow.dropna(subset=['Origin']).reset_index(drop=True)
        od_flow = od_flow[~od_flow['Origin'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        od_flow['Origin'] = od_flow['Origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        od_flow = od_flow[(od_flow['Origin'].str[0:11].isin(need_cbg))].reset_index(drop=True)
        od_flows.append(od_flow)

    od_flows = pd.concat(od_flows, ignore_index=True)
    od_flows.columns = ['destination', 'origin', 'monthly_total']
    od_flows = od_flows.groupby(['destination', 'origin']).sum().reset_index()
    od_flows.to_csv(r'D:\MDLD_OD\MDLDod\od\%s_OD.csv' % msa_name)

    # Plot PA
    # od_flows = pd.read_csv(r'D:\MDLD_OD\MDLDod\od\%s_OD.csv' % msa_name, index_col=0)
    # od_flows['origin'] = od_flows['origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
    # od_flows['destination'] = od_flows['destination'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
    pa_flows = od_flows.copy()
    pa_flows['destination'] = pa_flows['destination'].str[0:11]
    pa_flows = pa_flows.groupby(['destination'])['monthly_total'].sum().reset_index()
    pa_flows.columns = ['BGFIPS', 'attraction']
    msa_t_geo = CBG_geo.merge(pa_flows, on='BGFIPS')
    msa_t_geo = msa_t_geo.to_crs('EPSG:3857')
    msa_t_geo['area'] = msa_t_geo.area
    msa_t_geo['attraction_density'] = msa_t_geo['attraction'] / (msa_t_geo['area'] * 0.000247105)  # to acre
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    msa_t_geo.plot(column='attraction_density', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=9,
                   legend_kwds={'labelcolor': 'k', "fmt": "{:.2f}", 'ncol': 3, 'title': '',
                                'loc': 'lower center', 'frameon': False, 'facecolor': 'k', 'edgecolor': 'k',
                                'framealpha': 0.5}, linewidth=0, edgecolor='white', alpha=0.5)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ctx.add_basemap(ax, crs=msa_t_geo.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.title(msa_name)
    plt.tight_layout()
    plt.savefig(r'%s\%s\PA.pdf' % (url_r, msa_name))
    plt.close()

    # Weighting
    od_flows.columns = ['destination', 'origin', 'Flow']
    device_ratio.columns = ['destination', 'destination_ratio']
    od_flows = od_flows.merge(device_ratio, on='destination')
    device_ratio.columns = ['origin', 'origin_ratio']
    od_flows = od_flows.merge(device_ratio, on='origin')
    od_flows['Flow_w'] = od_flows['Flow'] / ((od_flows['origin_ratio'] + od_flows['destination_ratio']) / 2)
    od_flows['Flow_w'] = (od_flows['Flow_w'] / 30) * 0.6
    # CBG TO TRACT
    od_flows['origin'] = od_flows['origin'].str[0:11]
    od_flows['destination'] = od_flows['destination'].str[0:11]
    od_flows = od_flows.groupby(['destination', 'origin']).sum().reset_index()
    # Final ODs
    od_flows = od_flows[od_flows['Flow_w'] > 0.1].reset_index(drop=True)
    cbg_list = set(od_flows['destination']).union(set(od_flows['origin']))
    print('Number of zones: %s' % len(cbg_list))
    od_flows = od_flows[['destination', 'origin', 'Flow_w']]

    # Merge with DTA
    node = pd.read_csv(r'%s\%s\node.csv' % (url_r, msa_name))
    link = pd.read_csv(r'%s\%s\link.csv' % (url_r, msa_name), on_bad_lines='skip')
    link.rename({'facility_type': 'link_type_name'}, axis=1, inplace=True)
    print(link['link_type_name'].value_counts())

    # Reassign speed, capacity, and lanes
    link = link.merge(defaults_all, on='link_type_name')
    link['lanes'] = link['lanes_default']
    link['free_speed'] = link['speed_default']
    link['capacity'] = link['capacity_default']
    link['capacity'] = link['capacity'] * link['lanes']
    link = link.drop(['lanes_default', 'speed_default', 'capacity_default'], axis=1)

    # All link's node should be found in node.csv
    # link = link[link['link_type_name'].isin(['motorway', 'trunk', 'primary', 'secondary'])].reset_index(drop=True)
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    node_node = set(node['node_id'])
    # print('Pct of nodes in links: %s' % (len(link_node & node_node) / len(link_node)))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)

    # To geopandas
    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord), crs="EPSG:4326")
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    # Change zone id from CBFIPS to int
    zone_ids = pd.DataFrame({'destination': list(cbg_list), 'd_zone_id': range(0, len(cbg_list))})
    od_flows = od_flows.merge(zone_ids, on='destination')
    zone_ids.columns = ['origin', 'o_zone_id']
    od_flows = od_flows.merge(zone_ids, on='origin')
    od_flows = od_flows.drop(['destination', 'origin'], axis=1)
    od_flows.columns = ['volume', 'd_zone_id', 'o_zone_id']
    zone_ids.columns = ['BGFIPS', 'zone_id']

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

    ## Plot nodes and links
    fig, ax = plt.subplots(figsize=(9, 7))
    link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    node[~node['zone_id'].isnull()].plot(ax=ax, markersize=10, color='red', alpha=1)
    plt.title(msa_name)
    # ctx.add_basemap(ax, crs=G.graph['crs'], source=ctx.providers.CartoDB.Positron)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(r"%s\%s\node_link.png" % (url_r, msa_name), dpi=500)
    plt.close()

    # # Generate setting for DTALite
    assignment_settings = {'number_of_iterations': 20, 'route_output': 0, 'simulation_output': 0,
                           'number_of_cpu_processors': 8, 'length_unit': 'meter', 'speed_unit': 'kmh',
                           'UE_convergence_percentage': 0.001, 'odme_activate': 0}
    mode_types = [{'mode_type': 'auto', 'vot': 10, 'person_occupancy': 1, 'pce': 1}]
    demand_periods = [{'period': 'AM', 'time_period': '0700_0800'}]
    demand_files = [{'file_sequence_no': 1, 'file_name': 'demand.csv', 'demand_period': 'am', 'mode_type': 'auto',
                     'format_type': 'column', 'scale_factor': 1, 'departure_time_profile_no': 1}]
    subarea = [{'activate': 0,
                'subarea_geometry': 'POLYGON ((-73.933165 40.888486,-74.012753 40.754034,-74.027655 40.694427,-74.014108 40.686299,-73.995820 40.705942,-73.975499 40.707974,-73.956872 40.750308,-73.928762 40.793320,-73.865431 40.803818,-73.933165 40.888825,-73.933165 40.888486,))'}]
    departure_time_profiles = [
        {'departure_time_profile_no': 1, 'time_period': '0700_0800', 'T0420': 0.005002, 'T0425': 0.005020,
         'T0430': 0.005002, 'T0435': 0.005207, 'T0440': 0.005207, 'T0445': 0.005207, 'T0450': 0.005677,
         'T0455': 0.005677, 'T0460': 0.005677, 'T0465': 0.005994, 'T0470': 0.005994, 'T0475': 0.005994,
         'T0480': 0.006018}]
    link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()
    link_type['traffic_flow_model'] = ['kw', 'spatial_queue', 'spatial_queue'] + ['point_queue'] * (len(link_type) - 3)
    link_type.columns = ['link_type', 'link_type_name', 'free_speed_auto', 'capacity_auto', 'traffic_flow_model']
    link_types = link_type.to_dict(orient='records')

    # Output
    shutil.copy2(r'D:\MDLD_OD\DTALite_0602_2024.exe', r"%s\%s" % (url_r, msa_name))
    save_settings_yml(r"%s\%s\settings.yml" % (url_r, msa_name), assignment_settings, mode_types,
                      demand_periods, demand_files, subarea, link_types, departure_time_profiles)
    od_flows[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"%s\%s\demand.csv" % (url_r, msa_name),
                                                          index=False)
    node.to_csv(r"%s\%s\node.csv" % (url_r, msa_name), index=False)
    link.to_csv(r"%s\%s\link.csv" % (url_r, msa_name), index=False)
    zone_ids.to_csv(r"%s\%s\zone_id.csv" % (url_r, msa_name), index=False)

    # # Run assignment
    os.chdir(r"%s\%s" % (url_r, msa_name))
    subprocess.call([r"%s\%s\DTALite_0602_2024.exe" % (url_r, msa_name)])

    # Plot link performance
    assign_all = pd.read_csv(r'%s\%s\link_performance.csv' % (url_r, msa_name))
    assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)
    binning = mapclassify.NaturalBreaks(assign_all['vehicle_volume'], k=5)  # NaturalBreaks
    assign_all['cut_jenks'] = (binning.yb + 1) * 0.5
    aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'vehicle_volume', 'speed_kmph']],
                      on=['from_node_id', 'to_node_id'], how='left')
    fig, ax = plt.subplots(figsize=(9, 7))
    aadt[aadt['vehicle_volume'] == 0].plot(ax=ax, alpha=0.3, lw=0.25, color='gray')
    aadtr = aadt[aadt['vehicle_volume'] > 0].reset_index(drop=True)
    aadtr.plot(column='vehicle_volume', cmap='RdYlGn_r', scheme="natural_breaks", k=5, lw=aadtr['cut_jenks'], ax=ax,
               alpha=0.4, legend=True, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.title(msa_name)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'%s\%s\assigned_traffic.pdf' % (url_r, msa_name))
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.set_palette('coolwarm', 7)
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.barplot(aadt, x='link_type_name', y='vehicle_volume')
    plt.ylabel('Volume')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(r'%s\%s\volume_by_roadtype.png' % (url_r, msa_name), dpi=500)
    plt.close()
