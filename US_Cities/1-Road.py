import glob
import matplotlib.pyplot as plt
import osm2gmns as og
import pandas as pd
import geopandas as gpd
import os
import seaborn as sns
import ast
import contextily as ctx
from tqdm import tqdm
import osmnx as ox
from pathlib import Path
import shutil
import subprocess
import mapclassify
import yaml


def save_settings_yml(filename, assignment_settings, mode_types, demand_periods, demand_files, subarea, link_types,
                      departure_time_profiles):
    settings = {'assignment': assignment_settings, 'mode_types': mode_types, 'demand_periods': demand_periods,
                'demand_files': demand_files, 'subarea': subarea, 'link_types': link_types,
                'departure_time_profile': departure_time_profiles}
    with open(filename, 'w') as file:
        yaml.dump(settings, file)


# url_r = r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\\'
# # Read shapefile
# shp_layer = gpd.read_file(url_r + r'nhgis0011_shape\US_place_2023.shp')
# shp_layer = shp_layer.to_crs("EPSG:4326")
# # Read population
# total_pop = pd.read_csv(url_r + r'nhgis0010_csv\nhgis0018_ds267_20235_place.csv')
# total_pop = total_pop[['GISJOIN', 'ASN1E001']]
# # Merge
# shp_layer = shp_layer.merge(total_pop, on='GISJOIN')
# # Extract centroids for each city
# shp_layer['geo_lat'] = shp_layer.geometry.centroid.y  # Latitude
# shp_layer['geo_lon'] = shp_layer.geometry.centroid.x  # Longitude
# shp_layer = shp_layer.sort_values(by=['Country', 'Population'])
# shp_layer.to_file(r'D:\MDLD_OD\resilience\cities_84.shp')

# Read Cities
shp_layer = gpd.read_file(r'D:\MDLD_OD\resilience\cities_84.shp')
shp_layer = shp_layer.sort_values(by=['ASN1E001'], ascending=False).reset_index(drop=True)
# Read CBG
cbg_layer = pd.read_pickle(r'D:\Hurricane_Helene\Results\poly_cbg_84.pkl')
cbg_layer['BGFIPS'] = cbg_layer['GEOID'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
# Get # of devices
devices = pd.read_csv(
    r'G:\Data\SafeGraph\Neighbourhood Patterns\neighborhood_home_panel_summary\y=2019\m=5\part-00000-tid-4483054447075143375-997cc666-3832-460a-8831-3e198824ad6e-23887-1-c000.csv')
devices = devices[['census_block_group', 'number_devices_residing']]
devices.columns = ['BGFIPS', 'devices']
devices['BGFIPS'] = devices['BGFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
cbg_pop = pd.read_csv(r'F:\Research_Old\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
cbg_pop = cbg_pop[['BGFIPS', 'Total_Population']]
cbg_pop['BGFIPS'] = cbg_pop['BGFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
devices = devices.merge(cbg_pop, on='BGFIPS')
devices['devices_ratio'] = devices['devices'] / devices['Total_Population']  # devices.corr()
# plt.plot(devices['devices'], devices['Total_Population'],'o')
devices = devices[['BGFIPS', 'devices_ratio']]

# all file indexes
all_cities = list(range(0, 10)) + list(range(100, 110)) + list(range(200, 210)) + list(range(300, 310)) + list(
    range(400, 410))
all_ods = glob.glob('G:\Data\Dewey\Advan\\Neighborhood Patterns - US\\*DATE_RANGE_START-2019-05-01.csv.gz')
url_r = r'D:\MDLD_OD\resilience\road_network'
for kk in all_cities[0:1]:
    # Get network by distance
    city_center = (shp_layer.loc[kk, 'geo_lat'], shp_layer.loc[kk, 'geo_lon'])
    city_name = shp_layer.loc[kk, 'NAMELSAD']
    Path(r"%s\%s" % (url_r, city_name)).mkdir(parents=True, exist_ok=True)
    print('-------------- %s: Downloading --------------' % city_name)
    G = ox.graph.graph_from_point(city_center, dist=50 * 1000, dist_type='network', network_type="drive")  # meter
    print('No of edges: %s' % G.number_of_edges())
    edge_gpd = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    node_gpd = ox.convert.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    # G_gpd['highway'].value_counts()
    ef = r"%s\%s\%s.osm" % (url_r, city_name, city_name)
    ox.io.save_graph_xml(G, filepath=ef)

    # # Plot network
    print('-------------- %s: Plotting --------------' % city_name)
    fig, ax = ox.plot.plot_graph(G, node_size=0, show=False, close=False, edge_linewidth=0.6, edge_color='blue',
                                 edge_alpha=0.2)
    ctx.add_basemap(ax, crs=G.graph['crs'], source=ctx.providers.CartoDB.Positron)
    plt.title(city_name)
    plt.tight_layout()
    plt.savefig(r"%s\%s\%s.png" % (url_r, city_name, city_name), dpi=500)
    plt.close()

    # Convert the simulation network
    print('-------------- %s: Converting --------------' % city_name)
    net = og.getNetFromFile(
        ef, network_types=('auto',), default_lanes=True, default_speed=True, default_capacity=True,
        link_types=['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified',
                    'connector'])
    og.outputNetToCSV(net, output_folder=r"%s\%s" % (url_r, city_name), prefix=city_name + '_')

    # Get OD data
    SInBG = gpd.sjoin(node_gpd, cbg_layer, how='inner', predicate='within').reset_index(drop=True)
    SInBG_index = SInBG[['osmid', 'BGFIPS']]
    need_cbg = set(SInBG_index['BGFIPS'])
    od_flows = pd.DataFrame()
    for file in tqdm(all_ods):
        # Read OD flow
        ng_pattern = pd.read_csv(file)
        ng_pattern = ng_pattern.dropna(subset=['AREA']).reset_index(drop=True)
        ng_pattern = ng_pattern[~ng_pattern['AREA'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        ng_pattern['AREA'] = ng_pattern['AREA'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        ng_pattern = ng_pattern[(ng_pattern['AREA'].isin(need_cbg))].reset_index(drop=True)

        # Get monthly OD flow
        ng_pattern['DEVICE_HOME_AREAS'] = ng_pattern['DEVICE_HOME_AREAS'].apply(ast.literal_eval).reset_index(drop=True)
        d_explode = pd.DataFrame([*ng_pattern['DEVICE_HOME_AREAS']], ng_pattern.index).stack() \
            .rename_axis([None, 'Origin']).reset_index(1, name='Flow')
        od_flow = ng_pattern[['AREA']].join(d_explode)
        od_flow = od_flow.dropna(subset=['Origin']).reset_index(drop=True)
        od_flow = od_flow[~od_flow['Origin'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        od_flow['Origin'] = od_flow['Origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        od_flow = od_flow[(od_flow['Origin'].isin(need_cbg))].reset_index(drop=True)
        od_flows = pd.concat([od_flows, od_flow])

    # Weighting
    od_flows.columns = ['destination', 'origin', 'Flow']
    devices.columns = ['destination', 'destination_ratio']
    od_flows = od_flows.merge(devices, on='destination')
    devices.columns = ['origin', 'origin_ratio']
    od_flows = od_flows.merge(devices, on='origin')
    od_flows['Flow_w'] = od_flows['Flow'] / ((od_flows['origin_ratio'] + od_flows['destination_ratio']) / 2)
    od_flows['Flow_w'] = (od_flows['Flow_w'] / 30) * 0.1
    od_flows = od_flows[od_flows['Flow_w'] > 0.1].reset_index(drop=True)
    cbg_list = set(od_flows['destination']).union(set(od_flows['origin']))
    print('Number of zones: %s' % len(cbg_list))
    od_flows = od_flows[['destination', 'origin', 'Flow_w']]

    # Merge with DTA
    node = pd.read_csv(r'%s\%s\%s_node.csv' % (url_r, city_name, city_name))
    link = pd.read_csv(r'%s\%s\%s_link.csv' % (url_r, city_name, city_name))
    # All link's node should be found in node.csv
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

    # Node and CBG join: assign zone id (CBG) to node; connect to link with the highest class
    cbg_need = cbg_layer[cbg_layer['BGFIPS'].isin(cbg_list)].reset_index(drop=True)
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
    # fig, ax = plt.subplots(figsize=(9, 7))
    # link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    # node[~node['zone_id'].isnull()].plot(ax=ax, markersize=10, color='red', alpha=1)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # # Generate setting for DTALite
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
    link_type['traffic_flow_model'] = ['kw', 'spatial_queue', 'spatial_queue', 'point_queue', 'point_queue',
                                       'point_queue', 'point_queue']
    link_type.columns = ['link_type', 'link_type_name', 'free_speed_auto', 'capacity_auto', 'traffic_flow_model']
    link_types = link_type.to_dict(orient='records')

    # Output
    shutil.copy2(r'D:\MDLD_OD\DTALite_0602_2024.exe', r"%s\%s" % (url_r, city_name))
    save_settings_yml(r"%s\%s\settings.yml" % (url_r, city_name), assignment_settings, mode_types,
                      demand_periods, demand_files, subarea, link_types, departure_time_profiles)
    od_flows[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"%s\%s\demand.csv" % (url_r, city_name), index=False)
    node.to_csv(r"%s\%s\node.csv" % (url_r, city_name), index=False)
    link.to_csv(r"%s\%s\link.csv" % (url_r, city_name), index=False)
    #
    # # Run assignment
    os.chdir(r"%s\%s" % (url_r, city_name))
    subprocess.call([r"%s\%s\DTALite_0602_2024.exe" % (url_r, city_name)])

    # Plot link performance
    assign_all = pd.read_csv(r'%s\%s\link_performance.csv' % (url_r, city_name))
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
    plt.title(city_name)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'%s\%s\assigned_traffic.png' % (url_r, city_name), dpi=500)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.set_palette('coolwarm', 7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.barplot(aadt, x='link_type_name', y='vehicle_volume')
    plt.ylabel('Volume')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(r'%s\%s\volume_by_roadtype.png' % (url_r, city_name), dpi=500)
    plt.close()
