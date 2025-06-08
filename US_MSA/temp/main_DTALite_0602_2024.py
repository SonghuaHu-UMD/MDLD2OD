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
import numpy as np
import os
import yaml
from shapely.geometry import LineString, Point


def save_settings_yml(filename, assignment_settings, mode_types, demand_periods, demand_files, subarea, link_types,
                      departure_time_profiles):
    settings = {'assignment': assignment_settings, 'mode_types': mode_types, 'demand_periods': demand_periods,
                'demand_files': demand_files, 'subarea': subarea, 'link_types': link_types,
                'departure_time_profile': departure_time_profiles}
    with open(filename, 'w') as file:
        yaml.dump(settings, file)


# Calculate heading (bearing)
def calculate_heading(line):
    start = line.coords[0]
    end = line.coords[-1]
    heading_degrees = np.degrees(np.arctan2(end[0] - start[0], end[1] - start[1]))
    # Normalize heading to range [0, 360]
    if heading_degrees < 0:
        heading_degrees += 360
    return heading_degrees


# set default parameters
default_link_dict = {"motorway": 1, "trunk": 2, "primary": 3, "secondary": 4, "tertiary": 5, "residential": 6,
                     "unclassified": 20, 'connector': 99}
default_lanes_dict = {'motorway': 4, 'trunk': 3, 'primary': 3, 'secondary': 2, 'tertiary': 2, 'residential': 1,
                      'unclassified': 1, 'connector': 2}
default_speed_dict = {'motorway': 120, 'trunk': 100, 'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 30,
                      'unclassified': 30, 'connector': 120}
default_capacity_dict = {'motorway': 2300, 'trunk': 2200, 'primary': 1800, 'secondary': 1600, 'tertiary': 1200,
                         'residential': 1000, 'unclassified': 800, 'connector': 9999}
defaults_all = pd.DataFrame([default_lanes_dict, default_speed_dict, default_capacity_dict, default_link_dict]).T
defaults_all = defaults_all.reset_index()
defaults_all.columns = ['link_type_name', 'lanes_default', 'speed_default', 'capacity_default', 'link_type_default']
fips_to_abbr = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
                '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
                '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
                '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
                '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
                '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
                '54': 'WV', '55': 'WI', '56': 'WY'}
F_System_dict = {1: "Interstate", 2: "Principal Arterial – Other Freeways and Expressways",
                 3: "Principal Arterial – Other", 4: "Minor Arterial", 5: "Major Collector", 6: "Minor Collector",
                 7: "Local"}
un_st = ['02', '15', '60', '66', '69', '72', '78']

# Assignment unit: census track (CTR) or census block group (CBG)
s_unit = 'CTR'

############# 1. Read raw shapefile and regional data #############
# Read CBSA
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\MDLD_OD\MDLDod\shp\tl_2019_us_cbsa.shp')

# Read county
CT_geo = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\poly_ct_84.pkl')
CT_geo['centroid_lon'] = CT_geo.centroid.x
CT_geo['centroid_lat'] = CT_geo.centroid.y

if s_unit == 'CBG':
    # Read census block group
    CBG_geo = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\poly_cbg_84.pkl')
    CBG_geo['BGFIPS'] = CBG_geo['GEOID']
    CBG_geo = CBG_geo[~CBG_geo['GISJOIN'].str[1:3].isin(un_st)].reset_index(drop=True)
else:
    # Read census tract
    CBG_geo = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\poly_tract_84.pkl')
    CBG_geo['BGFIPS'] = CBG_geo['GEOID']
    CBG_geo = CBG_geo[~CBG_geo['GISJOIN'].str[1:3].isin(un_st)].reset_index(drop=True)

# Read CBSA Info
smart_loc = pd.read_pickle(r'D:\MDLD_OD\MDLDod\shp\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(un_st)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')

# Select Top 100 CBSA for simulation
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

# Read device count (CBG-level)
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
all_od_files = glob.glob(r'G:\Dewey\Advan\Neighborhood Patterns - US\\*DATE_RANGE_START-2019-05-01.csv.gz')
# Loop for each CBSA
for emsa in range(0, 100):
    msa_name = msa_pop.loc[emsa, 'CBSA_Name']
    Path(r"%s\%s" % (url_r, msa_name)).mkdir(parents=True, exist_ok=True)
    Path(r"D:\MDLD_OD\MDLDod\raw_data\%s" % msa_name).mkdir(parents=True, exist_ok=True)
    print("------------------- Start processing MSA: %s -----------------" % msa_name)

    ########## 1. Get the center county of CBSA based on highest population density ##########
    smart_loc_m = smart_loc.loc[smart_loc['CBSA_Name'] == msa_name, ['BGFIPS', 'TotPop', 'Ac_Land']]
    smart_loc_m['CTFIPS'] = smart_loc['BGFIPS'].str[0:5]
    smart_loc_m = smart_loc_m.groupby(['CTFIPS']).sum().reset_index()
    smart_loc_m['PopDes'] = smart_loc_m['TotPop'] / smart_loc_m['Ac_Land']
    smart_loc_m = smart_loc_m.sort_values(by='PopDes', ascending=False).reset_index(drop=True)
    CT_geo_m = CT_geo.loc[CT_geo['GEOID'] == smart_loc_m.head(1)['CTFIPS'].item(), :]
    msa_center = (CT_geo_m['centroid_lat'].item(), CT_geo_m['centroid_lon'].item())

    # 2. Get simulation network based on driving distance from CBSA center
    ox.settings.all_oneway = True
    ddist = 30 * 1.60934 * 1000  # 30 miles
    if os.path.exists(r'D:\MDLD_OD\MDLDod\raw_data\%s\osm_network.osm' % msa_name):
        G = ox.graph.graph_from_xml(r'D:\MDLD_OD\MDLDod\raw_data\%s\osm_network.osm' % msa_name)
    else:
        print('-------------- %s: Downloading Network--------------' % msa_name)
        G = ox.graph.graph_from_point(msa_center, dist=ddist, dist_type='network', network_type="drive", simplify=True)
        ox.io.save_graph_xml(G, filepath=r'D:\MDLD_OD\MDLDod\raw_data\%s\osm_network.osm' % msa_name)
    print('No of edges: %s' % G.number_of_edges())
    edge_gpd = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    node_gpd = ox.convert.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    boundary_gpd = gpd.GeoDataFrame(geometry=[edge_gpd.unary_union.convex_hull], crs=edge_gpd.crs)
    boundary_gpd.to_csv(r'D:\MDLD_OD\MDLDod\raw_data\%s\boundary.csv' % msa_name)

    # Get need CBG based on simulation network
    SInBG = gpd.sjoin(node_gpd, CBG_geo, how='inner', predicate='within').reset_index(drop=True)
    need_cbg = set(SInBG['BGFIPS'])
    need_st = set(SInBG['STATEFP'])

    # Convert the simulation network using osm2gmns
    print('-------------- %s: Converting Network--------------' % msa_name)
    net = og.getNetFromFile(r'D:\MDLD_OD\MDLDod\raw_data\%s\osm_network.osm' % msa_name)
    # og.fillLinkAttributesWithDefaultValues(net)
    # og.generateNodeActivityInfo(net)
    og.consolidateComplexIntersections(net, auto_identify=True)
    og.outputNetToCSV(net, output_folder=r"D:\MDLD_OD\MDLDod\raw_data\%s" % msa_name)

    # Add additional features to links
    node = pd.read_csv(r'D:\MDLD_OD\MDLDod\raw_data\%s\node.csv' % msa_name)
    link = pd.read_csv(r'D:\MDLD_OD\MDLDod\raw_data\%s\link.csv' % msa_name, on_bad_lines='skip')
    link.rename({'facility_type': 'link_type_name'}, axis=1, inplace=True)
    # print(link['link_type_name'].value_counts())
    # link.drop_duplicates(subset=['from_node_id', 'to_node_id'])
    # link = link[link['link_type_name'] != 'residential'].reset_index(drop=True)

    # Reassign speed, capacity, and lanes
    link = link.merge(defaults_all, on='link_type_name')
    link['link_type'] = link['link_type_default']
    link['lanes'] = link['lanes_default']
    link['free_speed'] = link['speed_default']
    link['capacity'] = link['capacity_default'] * link['lanes']
    # link['capacity'] = link['capacity'] * link['lanes']
    link = link.drop(['lanes_default', 'speed_default', 'capacity_default', 'link_type_default'], axis=1)

    # All link's node should be found in node.csv
    # link = link[link['link_type_name'].isin(['motorway', 'trunk', 'primary', 'secondary'])].reset_index(drop=True)
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    print('Pct of nodes in links: %s' % (len(link_node & set(node['node_id'])) / len(link_node)))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)

    # To geopandas
    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord), crs="EPSG:4326")
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    ########## 2. Prepare raw OD data from MPLD ##########
    od_flowss = []
    hourly_flows = []
    for file in tqdm(all_od_files):
        # Read OD flow
        ng_pattern = pd.read_csv(file)
        ng_pattern = ng_pattern.dropna(subset=['AREA']).reset_index(drop=True)
        ng_pattern = ng_pattern[~ng_pattern['AREA'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        ng_pattern['AREA'] = ng_pattern['AREA'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        if s_unit == 'CBG':
            ng_pattern = ng_pattern[(ng_pattern['AREA'].isin(need_cbg))].reset_index(drop=True)
        else:
            ng_pattern = ng_pattern[(ng_pattern['AREA'].str[0:11].isin(need_cbg))].reset_index(drop=True)

        # Get monthly OD flow
        ng_pattern['DEVICE_HOME_AREAS'] = ng_pattern['DEVICE_HOME_AREAS'].apply(ast.literal_eval).reset_index(drop=True)
        d_explode = pd.DataFrame([*ng_pattern['DEVICE_HOME_AREAS']], ng_pattern.index).stack() \
            .rename_axis([None, 'Origin']).reset_index(1, name='Flow')
        od_flow = ng_pattern[['AREA']].join(d_explode)
        od_flow = od_flow.dropna(subset=['Origin']).reset_index(drop=True)
        od_flow = od_flow[~od_flow['Origin'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        od_flow['Origin'] = od_flow['Origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        if s_unit == 'CBG':
            od_flow = od_flow[(od_flow['Origin'].isin(need_cbg))].reset_index(drop=True)
        else:
            od_flow = od_flow[(od_flow['Origin'].str[0:11].isin(need_cbg))].reset_index(drop=True)
        od_flowss.append(od_flow)

        # Get hourly OD flow
        hourly_visit = pd.DataFrame(ng_pattern['STOPS_BY_EACH_HOUR'].str[1:-1].str.split(',').tolist()).astype(float)
        date_range = [d.strftime('%Y-%m-%d %H:%M:%S')
                      for d in pd.date_range(ng_pattern.loc[0, 'DATE_RANGE_START'].split('T')[0],
                                             ng_pattern.loc[0, 'DATE_RANGE_END'].split('T')[0], freq='h')][0: -1]
        hourly_visit.columns = date_range
        hourly_visit['AREA'] = ng_pattern['AREA']
        hourly_visit_st = pd.melt(hourly_visit, id_vars=['AREA'], value_vars=date_range)
        hourly_visit_st.columns = ['AREA', 'Datetime', 'visits']
        hourly_visit_st['Datetime'] = pd.to_datetime(hourly_visit_st['Datetime'])
        hourly_visit = hourly_visit_st.groupby(['AREA', 'Datetime'])['visits'].sum().reset_index()
        hourly_flows.append(hourly_visit)

    # Output raw OD info
    od_flows = pd.concat(od_flowss, ignore_index=True)
    od_flows.columns = ['destination', 'origin', 'monthly_total']
    od_flows = od_flows.groupby(['destination', 'origin']).sum().reset_index()
    od_flows.to_csv(r'D:\MDLD_OD\MDLDod\raw_data\%s\OD.csv' % msa_name)

    hourly_flows = pd.concat(hourly_flows, ignore_index=True)
    hourly_flows = hourly_flows.groupby(['AREA', 'Datetime'])['visits'].sum().reset_index()
    hourly_flows.columns = ['destination', 'Datetime', 'hourly_flow']
    hourly_flows.to_csv(r'D:\MDLD_OD\MDLDod\raw_data\%s\hourly_ratio.csv' % msa_name)

    # Plot trip density
    pa_flows = od_flows.copy()
    pa_flows['destination'] = pa_flows['destination'].str[0:11]
    pa_flows = pa_flows.groupby(['destination'])['monthly_total'].sum().reset_index()
    pa_flows.columns = ['BGFIPS', 'attraction']
    msa_t_geo = CBG_geo.merge(pa_flows, on='BGFIPS')
    msa_t_geo = msa_t_geo.to_crs('EPSG:3857')
    msa_t_geo['attraction_density'] = msa_t_geo['attraction'] / (msa_t_geo.area * 0.000247105)  # to acre
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    msa_t_geo.plot(column='attraction_density', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=9,
                   legend_kwds={'labelcolor': 'k', "fmt": "{:.2f}", 'ncol': 3, 'title': '', 'loc': 'lower center',
                                'frameon': False, 'facecolor': 'k', 'edgecolor': 'k', 'framealpha': 0.5}, linewidth=0,
                   edgecolor='white', alpha=0.5)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ctx.add_basemap(ax, crs=msa_t_geo.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.title(msa_name)
    plt.tight_layout()
    plt.savefig(r'%s\%s\trip_density.pdf' % (url_r, msa_name))
    plt.close()

    ########## 3. Weight raw OD data in space and time ##########
    # Population weighting
    od_flows.columns = ['destination', 'origin', 'Flow']
    device_ratio.columns = ['destination', 'destination_ratio']
    od_flows = od_flows.merge(device_ratio, on='destination')
    device_ratio.columns = ['origin', 'origin_ratio']
    od_flows = od_flows.merge(device_ratio, on='origin')
    od_flows['Flow_w'] = od_flows['Flow'] / ((od_flows['origin_ratio'] + od_flows['destination_ratio']) / 2)

    # Hour weighting: covert monthly total to hour (7-8 am, weekday)
    hourly_flows['hourly_ratio'] = hourly_flows['hourly_flow'] / hourly_flows.groupby(['destination'])[
        'hourly_flow'].transform('sum')
    hourly_ratio = hourly_flows[
        (hourly_flows['Datetime'].dt.hour == 7) & (~hourly_flows['Datetime'].dt.dayofweek.isin([5, 6]))][[
        'destination', 'hourly_ratio']].groupby('destination').mean().reset_index()
    od_flows = od_flows.merge(hourly_ratio, on='destination')
    od_flows['Flow_w'] = od_flows['Flow_w'] * od_flows['hourly_ratio']
    hourly_flows['hour'] = hourly_flows['Datetime'].dt.hour
    peak_ratio = hourly_flows[(~hourly_flows['Datetime'].dt.dayofweek.isin([5, 6]))][['hourly_flow', 'hour']].groupby(
        'hour').mean().reset_index()
    peak_ratio['ratio'] = peak_ratio['hourly_flow'] / sum(peak_ratio['hourly_flow'])
    p_ratio = peak_ratio.loc[peak_ratio['hour'] == 7, 'ratio'].item()

    if s_unit == 'CTR':
        od_flows['origin'] = od_flows['origin'].str[0:11]
        od_flows['destination'] = od_flows['destination'].str[0:11]
        od_flows = od_flows.groupby(['destination', 'origin']).sum().reset_index()

    # Final ODs
    od_flows = od_flows[od_flows['Flow_w'] > 0].reset_index(drop=True)
    cbg_list = set(od_flows['destination']).union(set(od_flows['origin']))
    print('Number of zones: %s' % len(cbg_list))
    od_flows = od_flows[['destination', 'origin', 'Flow_w']]

    ########## 3. Reformat OD and network data to align with DTALite ##########
    # Change zone id from CBFIPS to int
    zone_ids = pd.DataFrame({'destination': list(cbg_list), 'd_zone_id': range(0, len(cbg_list))})
    od_flows = od_flows.merge(zone_ids, on='destination')
    zone_ids.columns = ['origin', 'o_zone_id']
    od_flows = od_flows.merge(zone_ids, on='origin')
    od_flows = od_flows.drop(['destination', 'origin'], axis=1)
    od_flows.columns = ['volume', 'd_zone_id', 'o_zone_id']
    zone_ids.columns = ['BGFIPS', 'zone_id']

    # Node and CBG join: assign zone id (CBG) to node; connect to the link with the highest class
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

    ## Plot zones and links
    fig, ax = plt.subplots(figsize=(9, 7))
    link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    node[~node['zone_id'].isnull()].plot(ax=ax, markersize=5, color='red', alpha=1)
    plt.title(msa_name)
    # ctx.add_basemap(ax, crs=G.graph['crs'], source=ctx.providers.CartoDB.Positron)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(r"%s\%s\node_link.png" % (url_r, msa_name), dpi=500)
    plt.close()

    ########## 4. Prepare ODME files: sensor data from FHWA AADT ##########
    # Get AADT based on state name: may cover multiple states
    abbr_list = [fips_to_abbr.get(str(f).zfill(2), None) for f in need_st]
    all_aadt = []
    for st_abb in abbr_list:
        state_network = pd.read_pickle(r'D:\MDLD_OD\Volume\HPMS_2020\HPMS_FULL_%s_2020.pkl' % st_abb)
        state_network = state_network.reset_index()
        SInBG = gpd.sjoin(state_network, boundary_gpd, how='inner', predicate='intersects').reset_index(drop=True)
        state_network = state_network[state_network['index'].isin(SInBG['index'])].reset_index(drop=True)
        all_aadt.append(state_network)
    all_aadt = pd.concat(all_aadt, ignore_index=True)
    all_aadt.drop(['index'], axis=1, inplace=True)
    all_aadt = all_aadt.explode(index_parts=False).reset_index(drop=True)
    all_aadt = all_aadt.drop_duplicates(subset=['F_SYSTEM', 'geometry', 'AADT'])
    all_aadt = all_aadt.dropna(subset='AADT').reset_index(drop=True)

    # Split each two-way road into two one-way roads
    for rr in ['THROUGH_LANES', 'AADT']:
        all_aadt.loc[all_aadt['FACILITY_TYPE'] == 2, rr] = (all_aadt.loc[all_aadt['FACILITY_TYPE'] == 2, rr] / 2)
    two_way = all_aadt[all_aadt['FACILITY_TYPE'] == 2].copy()
    two_way['geometry'] = two_way['geometry'].apply(lambda geom: LineString(list(geom.coords)[::-1]))
    all_aadt = pd.concat([all_aadt, two_way], ignore_index=True).reset_index(drop=True).reset_index()
    all_aadt.rename({'index': 'fhwa_id'}, axis=1, inplace=True)

    # AADT --> OSM based on distance, heading, and road type
    # Calculate heading
    link['heading'] = link['geometry'].apply(calculate_heading)
    all_aadt['heading'] = all_aadt['geometry'].apply(calculate_heading)

    # Generate a buffer (5 meter distance)
    link = link.to_crs('EPSG:3857')
    all_aadt = all_aadt.to_crs('EPSG:3857')
    link['buffer'] = link.geometry.buffer(5)
    dta_bf = gpd.GeoDataFrame(link[['link_id', 'buffer', 'heading', 'link_type_name']], geometry='buffer', crs=link.crs)
    osm_fhwa_ty = {'motorway': [1, 2], 'trunk': [1, 2], 'primary': [2, 3], 'secondary': [3, 4],
                   'tertiary': [4, 5, 6, 7], 'residential': [4, 5, 6, 7]}
    valid_linkss = []
    for ty in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
        dta_bf_t = dta_bf[dta_bf['link_type_name'] == ty]
        all_aadt_t = all_aadt[all_aadt['F_SYSTEM'].isin(osm_fhwa_ty[ty])]
        join_link = gpd.sjoin(dta_bf_t, all_aadt_t[['geometry', 'fhwa_id', 'heading', 'Route_ID', 'AADT']],
                              predicate='intersects')
        join_link = join_link.merge(all_aadt_t[['geometry', 'fhwa_id']], on='fhwa_id')
        join_link = join_link.merge(link[['geometry', 'link_id']], on='link_id')
        join_link['distance'] = join_link.apply(lambda row: row['geometry_x'].distance(row['geometry_y']), axis=1)

        valid_links = join_link[(np.abs(join_link['heading_left'] - join_link['heading_right']) < 10)]
        valid_links = valid_links.loc[valid_links.groupby(valid_links.link_id)['distance'].idxmin()]
        valid_links = valid_links[['link_id', 'fhwa_id', 'Route_ID', 'AADT']]
        valid_linkss.append(valid_links)
    valid_linkss = pd.concat(valid_linkss, ignore_index=True)
    link = link.merge(valid_linkss, on=['link_id'], how='left')
    link = link.to_crs('EPSG:4326')
    all_aadt = all_aadt.to_crs('EPSG:4326')

    # Plot matched outcomes
    fig, ax = plt.subplots(figsize=(12, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    aadt_match = link.copy()
    aadt_match[aadt_match['AADT'].isnull()].plot(ax=ax[0], alpha=0.3, lw=0.25, color='gray')
    aadtr = aadt_match[aadt_match['AADT'] > 0].reset_index(drop=True)
    binning = mapclassify.NaturalBreaks(aadtr['AADT'], k=5)  # NaturalBreaks
    aadtr['cut_jenks'] = (binning.yb + 1) * 0.5
    aadtr.plot(column='AADT', cmap='RdYlGn_r', scheme="natural_breaks", k=5, lw=aadtr['cut_jenks'], ax=ax[0],
               alpha=0.4, legend=True, legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ax[0].set_title('AADT (matched to OSM)')
    ax[0].axis('off')

    binning = mapclassify.NaturalBreaks(all_aadt['AADT'], k=5)  # NaturalBreaks
    all_aadt['cut_jenks'] = (binning.yb + 1) * 0.5
    all_aadt.plot(column='AADT', cmap='RdYlGn_r', scheme="natural_breaks", k=5, lw=all_aadt['cut_jenks'], ax=ax[1],
                  alpha=0.4, legend=True,
                  legend_kwds={"fmt": "{:.0f}", 'frameon': False, 'ncol': 1, 'loc': 'upper left'})
    ax[1].set_title('AADT (FHWA)')
    ax[1].axis('off')
    # ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.title(msa_name)
    plt.tight_layout()
    plt.savefig(r'%s\%s\AADT.pdf' % (url_r, msa_name))
    plt.close()

    # Generate sensors
    all_sensor = link[['Route_ID', 'link_type_name', 'from_node_id', 'to_node_id', 'AADT']]
    all_sensor = all_sensor.dropna(subset='AADT').reset_index(drop=True)
    all_sensor['count'] = all_sensor['AADT'] * p_ratio
    all_sensor = all_sensor.drop_duplicates(subset=['Route_ID']).reset_index(drop=True)
    all_sensor = pd.concat([all_sensor[all_sensor['link_type_name'].isin(['motorway', 'secondary', 'primary'])],
                            all_sensor[all_sensor['link_type_name'].isin(['residential', 'tertiary'])].sample(
                                frac=0.2)], axis=0).reset_index(drop=True).reset_index()
    all_sensor['sensor_id'] = all_sensor['index']
    all_sensor['scenario_index'] = 0
    all_sensor['activate'] = 1
    all_sensor['demand_period'] = 'AM'
    all_sensor[['sensor_id', 'from_node_id', 'to_node_id', 'count', 'scenario_index', 'activate',
                'demand_period']].to_csv(r"%s\%s\sensor_data.csv" % (url_r, msa_name), index=False)

    ########## 5. Generate setting for DTALite: A quick run to determine total weight ##########
    assignment_settings = {'number_of_iterations': 2, 'route_output': 0, 'simulation_output': 0,
                           'number_of_cpu_processors': 6, 'length_unit': 'meter', 'speed_unit': 'kmh',
                           'UE_convergence_percentage': 0.01, 'odme_activate': 0}
    mode_types = [{'mode_type': 'auto', 'vot': 10, 'person_occupancy': 1, 'pce': 1}]
    demand_periods = [{'period': 'AM', 'time_period': '0700_0800'}]
    demand_files = [{'file_sequence_no': 1, 'file_name': 'demand.csv', 'demand_period': 'am', 'mode_type': 'auto',
                     'format_type': 'column', 'scale_factor': 1, 'departure_time_profile_no': 1}]
    subarea = [{'activate': 1, 'subarea_geometry': CT_geo_m.geometry.convex_hull.item().wkt}]  # TODO: Check
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
    od_flows_s = od_flows[od_flows['volume'] > 0.1]
    pre_ttod = od_flows['volume'].sum()
    print('Total loss od (pct): %.5f' % (100 * (od_flows_s['volume'].sum() - pre_ttod) / pre_ttod))
    od_flows_s[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"%s\%s\demand.csv" % (url_r, msa_name), index=False)
    node.to_csv(r"%s\%s\node.csv" % (url_r, msa_name), index=False)
    link[['link_id', 'name', 'osm_way_id', 'from_node_id', 'to_node_id', 'directed', 'geometry', 'dir_flag', 'length',
          'link_type_name', 'link_type', 'free_speed', 'lanes', 'capacity', 'allowed_uses']].to_csv(
        r"%s\%s\link.csv" % (url_r, msa_name), index=False)
    zone_ids.to_csv(r"%s\%s\zone_id.csv" % (url_r, msa_name), index=False)

    # Run assignment
    os.chdir(r"%s\%s" % (url_r, msa_name))
    subprocess.call([r"%s\%s\DTALite_0602_2024.exe" % (url_r, msa_name)])

    ########## 6. Calculate total weighting and rerun the DTALite with ODME ##########
    assign_all_bf = pd.read_csv(r'%s\%s\link_performance.csv' % (url_r, msa_name))
    assign_all_bf = assign_all_bf.merge(link[['link_id', 'AADT']], on='link_id', how='left')
    assign_all_bf[['AADT', 'vehicle_volume']].corr()
    tt_weight = (all_aadt['AADT'].sum() * p_ratio / assign_all_bf['vehicle_volume'].sum())
    od_flows_w = od_flows.copy()
    od_flows_w['volume'] = od_flows_w['volume'] * tt_weight
    pre_ttod = od_flows_w['volume'].sum()
    od_flows_w = od_flows_w[od_flows_w['volume'] > 0.1].reset_index(drop=True)
    post_ttod = od_flows_w['volume'].sum()
    print('Total loss od (pct): %.5f' % (100 * (post_ttod - pre_ttod) / pre_ttod))

    # Run again with ODME
    assignment_settings = {'number_of_iterations': 15, 'route_output': 0, 'simulation_output': 0,
                           'number_of_cpu_processors': 8, 'length_unit': 'meter', 'speed_unit': 'kmh',
                           'UE_convergence_percentage': 0.01, 'odme_activate': 1}
    save_settings_yml(r"%s\%s\settings.yml" % (url_r, msa_name), assignment_settings, mode_types,
                      demand_periods, demand_files, subarea, link_types, departure_time_profiles)
    od_flows_w[['o_zone_id', 'd_zone_id', 'volume']].to_csv(r"%s\%s\demand.csv" % (url_r, msa_name), index=False)
    subprocess.call([r"%s\%s\DTALite_0602_2024.exe" % (url_r, msa_name)])

    assign_all = pd.read_csv(r'%s\%s\link_performance.csv' % (url_r, msa_name))
    assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)
    assign_all = assign_all.merge(link[['link_id', 'AADT']], on='link_id', how='left')
    assign_all[['AADT', 'vehicle_volume']].corr()

    # Plot link performance
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

    aadt['AADT_hour'] = aadt['AADT'] * p_ratio
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.set_palette('coolwarm', 7)
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.barplot(aadt, x='link_type_name', y='vehicle_volume')
    plt.ylabel('Volume')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(r'%s\%s\volume_by_roadtype.png' % (url_r, msa_name), dpi=500)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.regplot(data=aadt, x='AADT_hour', y='vehicle_volume', ax=ax, scatter_kws={'alpha': 0.4, 's': 10})
    plt.tight_layout()
    plt.savefig(r'%s\%s\volume_by_aadt.png' % (url_r, msa_name), dpi=500)
    plt.close()
