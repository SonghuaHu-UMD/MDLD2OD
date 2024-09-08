import glob
import osm2gmns as og
import pandas as pd
import geopandas as gpd
import os
import fiona
from tqdm import tqdm

url_r = r'D:\MDLD_OD\Netmob\\network\\'

# Read cities
city_list = pd.read_excel(r'D:\MDLD_OD\Netmob\City_list.xlsx')
city_list = city_list.join(
    city_list['Latitude/Longitude'].str.split(' / ', n=1, expand=True).rename(columns={0: 'Latitude', 1: 'Longitude'}))

# Creat the url to download osm files from bbbike
osm_url = []
for e_name in set(city_list['Name']):
    # e_name = 'New York-Newark-Jersey City, NY-NJ-PA'
    node = city_list.loc[city_list['Name'] == e_name, :].reset_index(drop=True)

    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.Longitude, node.Latitude), crs="EPSG:4326")
    node = node.to_crs("EPSG:3857")
    node['geometry'] = node['geometry'].buffer(100000, cap_style='square').to_crs("EPSG:4326")

    bbox = list(node.bounds[['miny', 'minx', 'maxy', 'maxx']].values[0])  # min lat, min long, max lat, max long
    corrd = node.geometry.convex_hull.get_coordinates().reset_index(drop=True)
    corrd['xy'] = corrd['x'].round(3).astype(str) + '%2C' + corrd['y'].round(3).astype(str)

    # Generate boundary for each MSA
    url_bbk = r'https://extract.bbbike.org/?sw_lng=%s&sw_lat=%s&ne_lng=%s&ne_lat=%s&format=osm.pbf&coords=%s&city=%s' % (
        node.bounds.values[0][0], node.bounds.values[0][1], node.bounds.values[0][2],
        node.bounds.values[0][3], corrd['xy'].str.cat(sep='%7C'), e_name)
    osm_url.append([e_name, node['Population'].values[0], url_bbk, node.geometry.convex_hull.values[0]])
osm_url = pd.DataFrame(osm_url, columns=['name', 'population', 'url', 'geometry'])
osm_url.to_csv(r'D:\MDLD_OD\Netmob\osm_url_netmob.csv')
osm_url_gpd = gpd.GeoDataFrame(osm_url)
osm_url_gpd.to_file(r'D:\MDLD_OD\Netmob\osm_url_netmob.shp')

# Generate road network from osm files
all_files = glob.glob(url_r + '*.pbf')
out_files = os.listdir(url_r)
for ef in all_files:
    # ef=all_files[0]
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
