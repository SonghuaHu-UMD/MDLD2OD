import glob
import osm2gmns as og
import pandas as pd
import numpy as np
import matplotlib
import geopandas as gpd
import matplotlib.pyplot as plt
import os

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

# Creat the url to download osm files from bbbike
osm_url = []
for e_name in set(msa_pop.head(50)['CBSA_Name']):
    # e_name = 'New York-Newark-Jersey City, NY-NJ-PA'
    msa_cbsa = msa_pop.loc[msa_pop['CBSA_Name'] == e_name, 'CBSA'].values[0]
    msa_need = MSA_geo[MSA_geo['CBSAFP'] == msa_cbsa]
    msa_need = msa_need.to_crs('EPSG:4326')
    bbox = list(msa_need.bounds[['miny', 'minx', 'maxy', 'maxx']].values[0])  # min lat, min long, max lat, max long
    corrd = msa_need.geometry.convex_hull.get_coordinates().reset_index(drop=True)
    corrd['xy'] = corrd['x'].round(3).astype(str) + '%2C' + corrd['y'].round(3).astype(str)

    # Generate boundary for each MSA
    url_bbk = r'https://extract.bbbike.org/?sw_lng=%s&sw_lat=%s&ne_lng=%s&ne_lat=%s&format=osm.pbf&coords=%s&city=%s' % (
        msa_need.bounds.values[0][0], msa_need.bounds.values[0][1], msa_need.bounds.values[0][2],
        msa_need.bounds.values[0][3], corrd['xy'].str.cat(sep='%7C'), msa_cbsa)
    osm_url.append([e_name, msa_cbsa, url_bbk])
osm_url = pd.DataFrame(osm_url, columns=['name', 'cbsa', 'url'])
osm_url.to_csv(r'D:\MDLD_OD\Others\osm_url.csv')

# Generate road network from osm files
for ef in all_files:
    e_cbsa = ef.split('\\')[-1].split('.')[0]
    if e_cbsa + '.pbf_link.csv' not in out_files:
        print('Start processing %s--------------' % e_cbsa)
        net = og.getNetFromFile(ef, network_types=('auto',), default_lanes=True, default_speed=True,
                                default_capacity=True,
                                link_types=['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
                                            'unclassified', 'connector'], )
        og.outputNetToCSV(net, output_folder=url_r, prefix=ef + '_')
    else:
        print('%s already exist--------------' % e_cbsa)

# Link OD data to road network
for ef in all_files:
    # ef=all_files[0]
    e_cbsa = ef.split('\\')[-1].split('.')[0]
    e_name = msa_pop.loc[msa_pop['CBSA'] == e_cbsa, 'CBSA_Name'].values[0]
    print('Start processing %s--------------' % e_name)
    msa_need = MSA_geo[MSA_geo['CBSAFP'] == e_cbsa]
    msa_need = msa_need.to_crs('EPSG:4326')

    node = pd.read_csv(url_r + e_cbsa + '.pbf_node.csv')
    link = pd.read_csv(url_r + e_cbsa + '.pbf_link.csv')
    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord),
                            crs="EPSG:4326")
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    # fig, ax = plt.subplots(figsize=(12, 8))
    # # node.plot(ax=ax, markersize=1, color='k', alpha=0.5)
    # link.plot(ax=ax, lw=1, color='g', alpha=0.5)
    # msa_need.boundary.plot(ax=ax, color='red')
    # plt.axis('off')
    # plt.tight_layout()

    # check link and node: all link's node should be found in node.csv
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    node_node = set(node['node_id'])
    print('Pct of nodes in links: %s' % (len(link_node & node_node) / len(link_node)))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)
    # check link type
    link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()

    # read od we need
    od_raw = pd.read_csv('D:\MDLD_OD\MDLDod\data\\%s_OD.csv' % e_name, index_col=0)
    od_raw['destination'] = od_raw['destination'].astype(str).apply(lambda x: x.zfill(12))
    od_raw['origin'] = od_raw['origin'].astype(str).apply(lambda x: x.zfill(12))
    cbg_list = set(od_raw['destination']).union(set(od_raw['origin']))
    print('Number of zones: %s' % len(cbg_list))

    cbg_need = CBG_geo[CBG_geo['BGFIPS'].isin(cbg_list)].reset_index(drop=True)

    # node and CBG join: assign zone id (CBG) to node
    SInBG = gpd.sjoin(node, cbg_need, how='inner', op='within').reset_index(drop=True)
