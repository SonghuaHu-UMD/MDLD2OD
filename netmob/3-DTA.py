import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import subprocess
import mapclassify
import contextily as ctx
import yaml
import h3
from collections import Counter
import scipy.spatial
import math
import pygeohash as pgh
import igraph
from functools import reduce

from IPython.core.pylabtools import figsize

plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
url_r = r'D:\MDLD_OD\Netmob\\network\\'
all_files = glob.glob(url_r + '*.pbf')
city_list = pd.read_excel(r'D:\MDLD_OD\Netmob\City_list.xlsx')
city_wt = pd.read_excel(r'D:\MDLD_OD\Netmob\results\city_wt.xlsx')
car_wt = pd.read_excel(r'D:\MDLD_OD\Netmob\car_weight.xlsx')

# From osm2gmns
default_lanes_dict = {'motorway': 4, 'trunk': 3, 'primary': 3, 'secondary': 2, 'tertiary': 2, 'residential': 1,
                      'living_street': 1, 'service': 1, 'cycleway': 1, 'footway': 1, 'track': 1, 'unclassified': 1,
                      'connector': 2}
default_speed_dict = {'motorway': 120, 'trunk': 100, 'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 30,
                      'living_street': 30, 'service': 30, 'cycleway': 5, 'footway': 5, 'track': 30, 'unclassified': 30,
                      'connector': 120}
default_capacity_dict = {'motorway': 2300, 'trunk': 2200, 'primary': 1800, 'secondary': 1600, 'tertiary': 1200,
                         'residential': 1000, 'living_street': 1000, 'service': 800, 'cycleway': 800, 'footway': 800,
                         'track': 800, 'unclassified': 800, 'connector': 9999}
defaults_all = pd.DataFrame([default_lanes_dict, default_speed_dict, default_capacity_dict]).T
defaults_all = defaults_all.reset_index()
defaults_all.columns = ['link_type_name', 'lanes_default', 'speed_default', 'capacity_default']

# Get hourly ratio
for kk in ['co', 'id', 'mx', 'in']:
    OD_3h = pd.read_csv(r'D:\MDLD_OD\Netmob\OD\3h\GH5\od_3h_gh5_%s_2019.csv' % kk)
    OD_3h['start_geohash5_cor'] = OD_3h['start_geohash5'].apply(pgh.decode)
    OD_3h['end_geohash5_cor'] = OD_3h['end_geohash5'].apply(pgh.decode)
    OD_3h['local_time_dt'] = pd.to_datetime(OD_3h['local_time'].str[0:-11])
    # - pd.to_timedelta(OD_3h['local_time'].str[-11:])
    OD_3h['hour'] = OD_3h['local_time_dt'].dt.hour
    OD_3h['dayofweek'] = OD_3h['local_time_dt'].dt.dayofweek
    # hour_ratio.plot(marker='o', color='blue')
    # hour_ratio = (hour_ratio / hour_ratio.groupby('dayofweek').sum()).reset_index()

    # Add n blank rows
    hour_ratio = OD_3h.groupby(['dayofweek', 'hour'])['trip_count'].mean().reset_index()
    n = 3 * 60 / 5
    new_index = pd.RangeIndex(len(hour_ratio) * (n + 1))
    new_df = pd.DataFrame(index=new_index, columns=hour_ratio.columns, dtype='float')
    ids = np.arange(len(hour_ratio)) * (n + 1)
    new_df.loc[ids] = hour_ratio.values
    new_df.loc[len(new_df) + 1] = new_df.loc[0]

    new_df['trip_count'] = new_df['trip_count'].interpolate(method='polynomial', order=3)
    new_df['dayofweek'] = new_df['dayofweek'].fillna(method='ffill').fillna(method='bfill')
    # new_df['trip_count'].plot(marker='o', color='royalblue', alpha=0.5, markersize=2)
    new_df['trip_total'] = new_df.groupby('dayofweek')['trip_count'].transform('sum')
    new_df['trip_count'] = (new_df['trip_count'] / new_df['trip_total'])
    new_df = new_df[new_df['hour'].isnull()]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    new_df['trip_count'].plot(marker='o', color='royalblue', alpha=0.3, markersize=1, ax=ax)
    plt.ylabel('5-minute trip rate')
    plt.tight_layout()
    plt.savefig(r'D:\MDLD_OD\Netmob\results\trip_ratio_%s.pdf' % kk)
    plt.close()


# defaults_all.to_csv(r'D:\MDLD_OD\Netmob\results\link_defaults.csv')

def save_settings_yml(filename, assignment_settings, mode_types, demand_periods, demand_files, subarea, link_types,
                      departure_time_profiles):
    settings = {'assignment': assignment_settings, 'mode_types': mode_types, 'demand_periods': demand_periods,
                'demand_files': demand_files, 'subarea': subarea, 'link_types': link_types,
                'departure_time_profile': departure_time_profiles}
    with open(filename, 'w') as file:
        yaml.dump(settings, file)


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


# Link OD data to road network
# all_files = ['D:\\MDLD_OD\\Netmob\\\\network\\Medan_planet_97.769_2.686_bd62ebe3.osm.pbf', ]
for ef in all_files[20:]:
    # ef=all_files[10]
    e_name = ef.split('\\')[-1].split('_')[0]
    e_ct = city_list.loc[city_list['Name'] == e_name, 'Country_code'].values[0]
    car_wtr = car_wt.loc[car_wt['name'] == e_name, 'ct_weight'].values[0]
    Path(r"D:\MDLD_OD\Netmob\Simulation\%s" % e_name).mkdir(parents=True, exist_ok=True)
    print('-------------- Start processing %s --------------' % e_name)

    node = pd.read_csv(url_r + ef.split('\\')[-1][0:-4] + '.pbf_node.csv')
    link = pd.read_csv(url_r + ef.split('\\')[-1][0:-4] + '.pbf_link.csv')
    link = link[link['link_type_name'].isin(
        ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'connector'])].reset_index(drop=True)

    # Reassign speed, capacity, and lanes
    link = link.merge(defaults_all, on='link_type_name')
    link['lanes'] = link['lanes_default']
    link['free_speed'] = link['speed_default']
    link['capacity'] = link['capacity_default']
    link['capacity'] = link['capacity'] * link['lanes']
    link = link.drop(['lanes_default', 'speed_default', 'capacity_default'], axis=1)
    # link.groupby('link_type_name')['free_speed'].mean()

    # All link's node should be found in node.csv
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    # node_node = set(node['node_id'])
    # print('Pct of nodes in links: %s' % (len(link_node & node_node) / len(link_node)))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)

    link_main = link[link['link_type_name'].isin(['motorway', 'trunk', 'primary', 'secondary'])].reset_index(drop=True)
    link_node_main = set(list(set(link_main['from_node_id'])) + list(set(link_main['to_node_id'])))

    # To geopandas
    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord),
                            crs="EPSG:4326").reset_index()
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    # poly_bound = node.geometry.bounds
    # utm_code = convert_wgs_to_utm(poly_bound['minx'][0], poly_bound['miny'][0])

    # Read od we need
    od_raw = pd.read_csv(r'D:\MDLD_OD\Netmob\OD\weekly\H37\od_week_h37_%s_2019.csv' % e_ct)
    od_raw = od_raw[od_raw['week_number'] != 53]
    od_raw = od_raw.groupby(['start_h3_7', 'end_h3_7']).mean().reset_index()
    od_raw['trip_count'] = (od_raw['trip_count'] / 7) * 0.1
    # od_raw.groupby('week_number')['trip_count'].sum().plot()

    # Do the weighting
    avg_weight = pd.read_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_new_%s.pkl' % e_ct)
    avg_weight = avg_weight[['geohash_5', 'weights']]
    # daily_pop_weight = 4345714 / 394269.96875  # daily adjust
    # avg_weight['weights'] = avg_weight['weights'] * daily_pop_weight

    od_raw['start_h3_7_cor'] = od_raw['start_h3_7'].apply(h3.h3_to_geo)
    od_raw['end_h3_7_cor'] = od_raw['end_h3_7'].apply(h3.h3_to_geo)

    od_raw_corr = pd.DataFrame(od_raw['start_h3_7_cor'].tolist(), index=od_raw.index)
    od_raw_corr.columns = ['start_lat', 'start_lng']
    od_raw_corr['start_geohash_5'] = od_raw_corr.apply(lambda x: pgh.encode(x.start_lat, x.start_lng, 5), axis=1)
    od_raw = od_raw.join(od_raw_corr)
    avg_weight.columns = ['start_geohash_5', 'start_weights']
    od_raw = od_raw.merge(avg_weight, on='start_geohash_5', how='left')

    od_raw_corr = pd.DataFrame(od_raw['end_h3_7_cor'].tolist(), index=od_raw.index)
    od_raw_corr.columns = ['end_lat', 'end_lng']
    od_raw_corr['end_geohash_5'] = od_raw_corr.apply(lambda x: pgh.encode(x.end_lat, x.end_lng, 5), axis=1)
    od_raw = od_raw.join(od_raw_corr)
    avg_weight.columns = ['end_geohash_5', 'end_weights']
    od_raw = od_raw.merge(avg_weight, on='end_geohash_5', how='left')
    na_rt = od_raw['start_weights'].isnull().sum() / len(od_raw)
    print(od_raw['start_weights'].isnull().sum() / len(od_raw))
    od_raw['start_weights'] = od_raw['start_weights'].fillna(od_raw['start_weights'].mean())
    od_raw['end_weights'] = od_raw['end_weights'].fillna(od_raw['end_weights'].mean())
    od_raw['trip_count_raw'] = od_raw['trip_count']
    od_raw['trip_count'] = od_raw['trip_count'] * ((od_raw['start_weights'] + od_raw['end_weights']) / 2)

    # All H3_7 points with OD flows
    H3_7_points = pd.DataFrame(np.vstack([od_raw[['start_h3_7']].values, od_raw[['end_h3_7']].values]))
    H3_7_points.columns = ['h3_7']
    H3_7_points = H3_7_points.drop_duplicates().reset_index(drop=True)
    H3_7_points['h3_7_corr'] = H3_7_points['h3_7'].apply(h3.h3_to_geo)
    H3_7_points[['h3_7_lat', 'h3_7_lng']] = pd.DataFrame(H3_7_points['h3_7_corr'].tolist(), index=H3_7_points.index)
    H3_7_points_gpd = gpd.GeoDataFrame(
        H3_7_points, geometry=gpd.points_from_xy(H3_7_points.h3_7_lng, H3_7_points.h3_7_lat), crs="EPSG:4326")
    H3_7_points_gpd = H3_7_points_gpd.to_crs(epsg=3857)
    H3_7_points_gpd['lon_utm'] = H3_7_points_gpd.get_coordinates()['x']
    H3_7_points_gpd['lat_utm'] = H3_7_points_gpd.get_coordinates()['y']
    H3_7_points_gpd_raw = H3_7_points_gpd.copy()

    node_3857 = node.to_crs(epsg=3857)
    node_3857['lon_utm'] = node_3857.get_coordinates()['x']
    node_3857['lat_utm'] = node_3857.get_coordinates()['y']
    node_3857_main = node_3857[node_3857['node_id'].isin(link_node_main)].reset_index(drop=True)

    # Querying for the k-nearest neighbors: at least secondary roads
    ref_points = node_3857_main[['lon_utm', 'lat_utm']].values
    points = np.dstack([H3_7_points_gpd['lon_utm'], H3_7_points_gpd['lat_utm']])[0]
    dist_ckd1, indexes_ckd1 = scipy.spatial.cKDTree(ref_points).query(points)
    # H3_7_points_gpd['index'] = indexes_ckd1
    H3_7_points_gpd['index'] = list(node_3857_main.loc[indexes_ckd1, 'index'])  # get index of node
    H3_7_points_gpd['distance'] = dist_ckd1
    # H3_7_points_gpd = H3_7_points_gpd.merge(node_3857[['index', 'node_id']], on='index')
    H3_7_points_gpd = H3_7_points_gpd[H3_7_points_gpd['distance'] < 8000].reset_index(drop=True)
    od_raw = od_raw[
        (od_raw['start_h3_7'].isin(H3_7_points_gpd['h3_7'])) & (od_raw['end_h3_7'].isin(H3_7_points_gpd['h3_7']))]

    H3_7_points_gpd_cor = H3_7_points_gpd[['h3_7', 'lat_utm', 'lon_utm']]
    H3_7_points_gpd_cor.columns = ['start_h3_7', 'start_h3_7_lat', 'start_h3_7_lng']
    od_raw = od_raw.merge(H3_7_points_gpd_cor, on='start_h3_7')
    H3_7_points_gpd_cor.columns = ['end_h3_7', 'end_h3_7_lat', 'end_h3_7_lng']
    od_raw = od_raw.merge(H3_7_points_gpd_cor, on='end_h3_7')

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    # # H3_7_points_gpd_raw.plot(ax=ax, color='k', markersize=5)  # all OD points
    # # node_3857.plot(ax=ax, color='red', markersize=1, alpha=0.1)  # all road points
    # H3_7_points_gpd.plot(ax=ax, color='green', markersize=10, alpha=0.5)
    # for kk in range(0, len(od_raw)):
    #     ax.annotate('', xy=(od_raw.loc[kk, 'start_h3_7_lng'], od_raw.loc[kk, 'start_h3_7_lat']),
    #                 xytext=(od_raw.loc[kk, 'end_h3_7_lng'], od_raw.loc[kk, 'end_h3_7_lat']),
    #                 arrowprops={'arrowstyle': '->', 'lw': 10 * od_raw.loc[kk, 'trip_count'] / max(od_raw['trip_count']),
    #                             'color': 'royalblue', 'alpha': 0.3, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    # ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    # ctx.add_basemap(ax, crs=H3_7_points_gpd.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.tight_layout()
    # plt.savefig(r'D:\MDLD_OD\Netmob\Simulation\%s\assigned_od_%s.pdf' % (e_name, e_name))
    # plt.close()

    # OD Network analysis
    OD_Distance = od_raw.drop_duplicates(subset=['start_h3_7', 'end_h3_7'])[['start_h3_7', 'end_h3_7', 'mdn_length_m']]
    OD_Graph1 = igraph.Graph.TupleList([tuple(x) for x in OD_Distance.values], directed=True,
                                       edge_attrs=['mdn_length_m'])
    communities1 = OD_Graph1.community_infomap(edge_weights='mdn_length_m')
    components1 = OD_Graph1.connected_components(mode='weak')

    OD_Trip = od_raw.drop_duplicates(subset=['start_h3_7', 'end_h3_7'])[['start_h3_7', 'end_h3_7', 'trip_count_raw']]
    OD_Graph2 = igraph.Graph.TupleList([tuple(x) for x in OD_Trip.values], directed=True, edge_attrs=['trip_count_raw'])
    communities2 = OD_Graph2.community_infomap(edge_weights='trip_count_raw')
    components2 = OD_Graph2.connected_components(mode='weak')

    dist_nt = [OD_Graph1.diameter(directed=True, weights='mdn_length_m'), OD_Graph1.density(loops=True),
               OD_Graph1.transitivity_undirected(), len(set(communities1.membership)),
               max(components1.sizes()), len(components1.sizes()), max(Counter(communities1.membership).values()),
               OD_Graph2.diameter(directed=True, weights='trip_count_raw'), OD_Graph2.density(loops=True),
               OD_Graph2.transitivity_undirected(), len(set(communities2.membership)),
               max(components2.sizes()), len(components2.sizes()), max(Counter(communities2.membership).values()),
               od_raw[od_raw['start_h3_7'] == od_raw['end_h3_7']]['trip_count_raw'].sum() / od_raw[
                   'trip_count_raw'].sum(), len(od_raw), len(od_raw[od_raw['start_h3_7'] == od_raw['end_h3_7']]),
               (od_raw['trip_count_raw'] * od_raw['m_length_m']).sum() / (od_raw['trip_count_raw'].sum()),
               (od_raw['trip_count_raw'] * od_raw['m_duration_min']).sum() / (od_raw['trip_count_raw'].sum()),
               (od_raw['trip_count_raw'] * od_raw['m_points_no']).sum() / (od_raw['trip_count_raw'].sum()),
               sum(od_raw['trip_count_raw'])]
    od_network = pd.DataFrame([dist_nt])
    od_network.columns = ['diameter_d', 'density_d', 'transitivity_d', 'communities_d', 'max_comp_d', 'num_comp_d',
                          'max_comm_d', 'diameter_v', 'density_v', 'transitivity_v', 'communities_v', 'max_comp_v',
                          'num_comp_v', 'max_comm_v', 'self_loop', 'od_count', 'od_self_count', 'avg_dist',
                          'avg_duration', 'avg_points', 'total_trips']
    od_network = pd.concat([city_wt[city_wt['name'] == e_name].reset_index(drop=True), od_network], axis=1)

    # od_raw = od_raw[od_raw['monthly_total'] > 0.1].reset_index(drop=True)
    cbg_list = set(od_raw['start_h3_7']).union(set(od_raw['end_h3_7']))
    print('Number of zones: %s' % len(cbg_list))
    od_network['PA_count'] = len(cbg_list)
    od_network['na_ratio'] = na_rt

    # Change zone id from CBFIPS to int
    zone_ids = pd.DataFrame({'start_h3_7': list(cbg_list), 'o_zone_id': range(0, len(cbg_list))})
    od_raw = od_raw.merge(zone_ids, on='start_h3_7')
    zone_ids.columns = ['end_h3_7', 'd_zone_id']
    od_raw = od_raw.merge(zone_ids, on='end_h3_7')
    # od_raw = od_raw.drop(['destination', 'origin'], axis=1)
    od_raw = od_raw[['o_zone_id', 'd_zone_id', 'trip_count']]
    od_raw.columns = ['o_zone_id', 'd_zone_id', 'volume']
    od_raw = od_raw[od_raw['o_zone_id'] != od_raw['d_zone_id']].reset_index(drop=True)

    # node with zone id
    node = node.merge(H3_7_points_gpd[['index', 'h3_7']], on='index', how='left')
    node = node.drop('zone_id', axis=1)
    zone_ids.columns = ['h3_7', 'zone_id']
    node = node.merge(zone_ids, on='h3_7', how='left')
    node = node.drop('h3_7', axis=1)

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
    link_type_df = pd.DataFrame(
        {'link_type_name': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified'],
         'traffic_flow_model': ['kw', 'spatial_queue', 'spatial_queue', 'point_queue', 'point_queue', 'point_queue',
                                'point_queue']})
    link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()
    link_type = link_type.merge(link_type_df, on='link_type_name')
    link_type.columns = ['link_type', 'link_type_name', 'free_speed_auto', 'capacity_auto', 'traffic_flow_model']
    link_types = link_type.to_dict(orient='records')

    # Output
    shutil.copy2(r'D:\MDLD_OD\DTALite_0602_2024.exe', r"D:\MDLD_OD\Netmob\Simulation\%s" % e_name)
    save_settings_yml(r"D:\MDLD_OD\Netmob\Simulation\%s\settings.yml" % e_name, assignment_settings, mode_types,
                      demand_periods, demand_files, subarea, link_types, departure_time_profiles)
    od_raw['volume'] = od_raw['volume'] * car_wtr
    od_raw[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"D:\MDLD_OD\Netmob\Simulation\%s\demand.csv" % e_name,
                                                        index=False)
    node.to_csv(r"D:\MDLD_OD\Netmob\Simulation\%s\node.csv" % e_name, index=False)
    link.to_csv(r"D:\MDLD_OD\Netmob\Simulation\%s\link.csv" % e_name, index=False)

    # Run assignment
    os.chdir(r"D:\MDLD_OD\Netmob\Simulation\%s" % e_name)
    subprocess.call([r"D:\MDLD_OD\Netmob\Simulation\%s\DTALite_0602_2024.exe" % e_name])

    # # Plot link performance
    assign_all = pd.read_csv(r'D:\MDLD_OD\Netmob\Simulation\%s\link_performance.csv' % e_name)
    assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)

    # calculate speed
    assign_all = pd.concat([assign_all, link[['link_type_name', 'free_speed', 'capacity']]], axis=1)
    assign_all.loc[assign_all['link_type_name'].isin(['motorway', 'trunk']), 'speed_bpr'] = (
            assign_all.loc[assign_all['link_type_name'].isin(['motorway', 'trunk']), 'free_speed'] / (
            1 + 0.65625 * np.power(
        assign_all.loc[assign_all['link_type_name'].isin(['motorway', 'trunk']), 'VOC'] + 0.5, 4.8)))
    assign_all.loc[assign_all['link_type_name'].isin(['primary', 'secondary']), 'speed_bpr'] = (
            assign_all.loc[assign_all['link_type_name'].isin(['primary', 'secondary']), 'free_speed'] / (
            1 + 1 * np.power(assign_all.loc[assign_all['link_type_name'].isin(['primary', 'secondary']), 'VOC'] + 0.5,
                             4)))
    assign_all.loc[assign_all['link_type_name'].isin(['tertiary']), 'speed_bpr'] = (
            assign_all.loc[assign_all['link_type_name'].isin(['tertiary']), 'free_speed'] / (
            1 + 1.28571 * np.power(assign_all.loc[assign_all['link_type_name'].isin(['tertiary']), 'VOC'] + 0.5, 3)))

    # Plot: volume
    assign_all_cor = assign_all.copy()
    assign_all = assign_all[assign_all['vehicle_volume'] > 0].reset_index(drop=True)
    binning = mapclassify.NaturalBreaks(assign_all['vehicle_volume'], k=5)  # NaturalBreaks
    assign_all['cut_jenks'] = binning.yb + 1
    aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'vehicle_volume', 'speed_kmph']],
                      on=['from_node_id', 'to_node_id'], how='left')
    aadt['cut_jenks'] = aadt['cut_jenks'].fillna(0.5)
    aadt['vehicle_volume'] = aadt['vehicle_volume'].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 7))
    aadt[aadt['vehicle_volume'] == 0].plot(ax=ax, alpha=0.5, lw=0.25, color='gray')
    aadt[aadt['vehicle_volume'] > 0].plot(
        column='cut_jenks', cmap='RdYlGn_r', scheme="user_defined",
        classification_kwds={'bins': list(set(aadt['cut_jenks']))}, lw=0.5, ax=ax, alpha=0.6,
        legend=False, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    # plt.axis('off')
    plt.savefig(r'D:\MDLD_OD\Netmob\Simulation\%s\assigned_traffic_%s.pdf' % (e_name, e_name))
    plt.close()

    # Plot: speed
    binning = mapclassify.NaturalBreaks(assign_all['speed_bpr'], k=5)  # NaturalBreaks
    assign_all['cut_jenks'] = binning.yb + 1
    # aadt = pd.concat([link, assign_all[['cut_jenks', 'vehicle_volume', 'speed_bpr']]], axis=1)
    aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'vehicle_volume', 'speed_kmph']],
                      on=['from_node_id', 'to_node_id'], how='left')
    aadt['cut_jenks'] = aadt['cut_jenks'].fillna(0.5)
    aadt['vehicle_volume'] = aadt['vehicle_volume'].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 7))
    aadt[aadt['vehicle_volume'] == 0].plot(ax=ax, alpha=0.5, lw=0.25, color='gray')
    aadt[aadt['vehicle_volume'] > 0].plot(
        column='cut_jenks', cmap='RdYlGn', scheme="user_defined",
        classification_kwds={'bins': list(set(aadt['cut_jenks']))}, lw=0.5, ax=ax, alpha=0.6,
        legend=False, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    # plt.axis('off')
    plt.savefig(r'D:\MDLD_OD\Netmob\Simulation\%s\assigned_speed_%s.pdf' % (e_name, e_name))
    plt.close()

    # compute all metrics for assignment outcome
    link_length = link.groupby('link_type')['length'].sum().reset_index()
    link_length.columns = ['link_type', 'length']
    assign_all_cor0 = assign_all_cor[assign_all_cor['vehicle_volume'] == 0]
    zero_link_length = assign_all_cor0.groupby('link_type')['distance_km'].sum().reset_index()
    zero_link_length.columns = ['link_type', 'zero_length']
    zero_link_length['zero_length'] = zero_link_length['zero_length'] * 1000  # to m

    link_count = link.groupby('link_type')['link_id'].count().reset_index()
    link_count.columns = ['link_type', 'link_count']
    VOC = assign_all_cor.groupby('link_type')['VOC'].mean().reset_index()
    VOC.columns = ['link_type', 'VOC_mean']
    VOC_max = assign_all_cor.groupby('link_type')['VOC'].max().reset_index()
    VOC_max.columns = ['link_type', 'VOC_max']
    speed = assign_all_cor.groupby('link_type')['speed_bpr'].mean().reset_index()
    speed.columns = ['link_type', 'speed']
    volume = assign_all_cor.groupby('link_type')['vehicle_volume'].mean().reset_index()
    volume.columns = ['link_type', 'volume']
    volume_max = assign_all_cor.groupby('link_type')['vehicle_volume'].max().reset_index()
    volume_max.columns = ['link_type', 'volume_max']

    assign_all_cor1 = assign_all_cor[assign_all_cor['vehicle_volume'] > 0]
    VOC1 = assign_all_cor1.groupby('link_type')['VOC'].mean().reset_index()
    VOC1.columns = ['link_type', 'VOC_mean1']
    speed1 = assign_all_cor1.groupby('link_type')['speed_bpr'].mean().reset_index()
    speed1.columns = ['link_type', 'speed1']
    volume1 = assign_all_cor1.groupby('link_type')['vehicle_volume'].mean().reset_index()
    volume1.columns = ['link_type', 'volume1']

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['link_type'], how='outer'),
                       [link_type, link_length, zero_link_length, link_count, VOC, VOC_max, volume, speed, volume_max,
                        VOC1, speed1, volume1])

    all_metrics = pd.concat([df_merged, od_network.loc[od_network.index.repeat(len(df_merged))].reset_index(drop=True)],
                            axis=1)
    all_metrics.to_excel(r'D:\MDLD_OD\Netmob\Simulation\%s\all_metrics.xlsx' % e_name)
