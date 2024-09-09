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
import scipy.spatial
import math

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
for ef in all_files[17:]:
    # ef=all_files[-12]
    e_name = ef.split('\\')[-1].split('_')[0]
    e_ct = city_list.loc[city_list['Name'] == e_name, 'Country_code'].values[0]
    print('-------------- Start processing %s --------------' % e_name)

    node = pd.read_csv(url_r + ef.split('\\')[-1][0:-4] + '.pbf_node.csv')
    link = pd.read_csv(url_r + ef.split('\\')[-1][0:-4] + '.pbf_link.csv')
    # link = link[link['link_type_name'].isin(
    #     ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'connector'])].reset_index(drop=True)

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
    # od_raw['start_h3_7_cor'] = od_raw['start_h3_7'].apply(h3.h3_to_geo)
    # od_raw['end_h3_7_cor'] = od_raw['end_h3_7'].apply(h3.h3_to_geo)

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

    # Querying for the k-nearest neighbors
    ref_points = node_3857[['lon_utm', 'lat_utm']].values
    points = np.dstack([H3_7_points_gpd['lon_utm'], H3_7_points_gpd['lat_utm']])[0]
    dist_ckd1, indexes_ckd1 = scipy.spatial.cKDTree(ref_points).query(points)
    H3_7_points_gpd['index'] = indexes_ckd1
    H3_7_points_gpd['distance'] = dist_ckd1
    # H3_7_points_gpd = H3_7_points_gpd.merge(node_3857[['index', 'node_id']], on='index')
    H3_7_points_gpd = H3_7_points_gpd[H3_7_points_gpd['distance'] < 5000].reset_index(drop=True)
    od_raw = od_raw[
        (od_raw['start_h3_7'].isin(H3_7_points_gpd['h3_7'])) & (od_raw['end_h3_7'].isin(H3_7_points_gpd['h3_7']))]

    H3_7_points_gpd_cor = H3_7_points_gpd[['h3_7', 'lat_utm', 'lon_utm']]
    H3_7_points_gpd_cor.columns = ['start_h3_7', 'start_h3_7_lat', 'start_h3_7_lng']
    od_raw = od_raw.merge(H3_7_points_gpd_cor, on='start_h3_7')
    H3_7_points_gpd_cor.columns = ['end_h3_7', 'end_h3_7_lat', 'end_h3_7_lng']
    od_raw = od_raw.merge(H3_7_points_gpd_cor, on='end_h3_7')

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    # H3_7_points_gpd_raw.plot(ax=ax, color='k', markersize=5)  # all OD points
    # node_3857.plot(ax=ax, color='red', markersize=1, alpha=0.1)  # all road points
    H3_7_points_gpd.plot(ax=ax, color='green', markersize=10, alpha=0.5)
    for kk in range(0, len(od_raw)):
        ax.annotate('', xy=(od_raw.loc[kk, 'start_h3_7_lng'], od_raw.loc[kk, 'start_h3_7_lat']),
                    xytext=(od_raw.loc[kk, 'end_h3_7_lng'], od_raw.loc[kk, 'end_h3_7_lat']),
                    arrowprops={'arrowstyle': '->', 'lw': 10 * od_raw.loc[kk, 'trip_count'] / max(od_raw['trip_count']),
                                'color': 'royalblue', 'alpha': 0.5, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tight_layout()
    Path(r"D:\MDLD_OD\Netmob\Simulation\%s" % e_name).mkdir(parents=True, exist_ok=True)
    plt.savefig(r'D:\MDLD_OD\Netmob\Simulation\%s\assigned_od.pdf' % e_name)
    plt.close()

    # od_raw = od_raw[od_raw['monthly_total'] > 0.1].reset_index(drop=True)
    cbg_list = set(od_raw['start_h3_7']).union(set(od_raw['end_h3_7']))
    print('Number of zones: %s' % len(cbg_list))

    # Change zone id from CBFIPS to int
    zone_ids = pd.DataFrame({'start_h3_7': list(cbg_list), 'o_zone_id': range(0, len(cbg_list))})
    od_raw = od_raw.merge(zone_ids, on='start_h3_7')
    zone_ids.columns = ['end_h3_7', 'd_zone_id']
    od_raw = od_raw.merge(zone_ids, on='end_h3_7')
    # od_raw = od_raw.drop(['destination', 'origin'], axis=1)
    od_raw = od_raw[['o_zone_id', 'd_zone_id', 'trip_count']]
    od_raw.columns = ['o_zone_id', 'd_zone_id', 'volume']

    # node with zone id
    node = node.merge(H3_7_points_gpd[['index', 'h3_7']], on='index', how='left')
    node = node.drop('zone_id', axis=1)
    zone_ids.columns = ['h3_7', 'zone_id']
    node = node.merge(zone_ids, on='h3_7', how='left')
    node = node.drop('h3_7', axis=1)

    ## Plot nodes and links
    # fig, ax = plt.subplots(figsize=(9, 7))
    # link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    # node[~node['zone_id'].isnull()].plot(ax=ax, markersize=10, color='red', alpha=1)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.close()
    # plt.show()

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
    # shutil.copy2(r'D:\MDLD_OD\DTALite_230915.exe', r"D:\MDLD_OD\Simulation\%s" % e_cbsa)
    shutil.copy2(r'D:\MDLD_OD\DTALite_0602_2024.exe', r"D:\MDLD_OD\Netmob\Simulation\%s" % e_name)
    save_settings_yml(r"D:\MDLD_OD\Netmob\Simulation\%s\settings.yml" % e_name, assignment_settings, mode_types,
                      demand_periods, demand_files, subarea, link_types, departure_time_profiles)
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
    assign_all = assign_all[assign_all['vehicle_volume'] > 0].reset_index(drop=True)
    binning = mapclassify.NaturalBreaks(assign_all['vehicle_volume'], k=5)  # NaturalBreaks
    assign_all['cut_jenks'] = binning.yb + 1
    aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'vehicle_volume', 'speed_kmph']],
                      on=['from_node_id', 'to_node_id'], how='left')
    aadt['cut_jenks'] = aadt['cut_jenks'].fillna(0.5)
    aadt['vehicle_volume'] = aadt['vehicle_volume'].fillna(0)
    # aadt.to_file(r'D:\MDLD_OD\Netmob\aadt_test.shp')

    fig, ax = plt.subplots(figsize=(9, 7))
    aadt[aadt['vehicle_volume'] == 0].plot(ax=ax, alpha=0.1, lw=1, color='gray')
    aadt[aadt['vehicle_volume'] > 0].plot(
        column='cut_jenks', cmap='RdYlGn_r', scheme="user_defined",
        classification_kwds={'bins': list(set(aadt['cut_jenks']))}, lw=aadt['cut_jenks'], ax=ax, alpha=0.6,
        legend=False, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    # plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'D:\MDLD_OD\Netmob\Simulation\%s\assigned_traffic.pdf' % e_name)
    plt.close()
