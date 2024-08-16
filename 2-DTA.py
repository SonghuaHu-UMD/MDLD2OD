import glob
import osm2gmns as og
import pandas as pd
import numpy as np
import matplotlib
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# Get network from osm
url_r = r'D:\MDLD_OD\Roadosm\\'
all_files = glob.glob(url_r + '*.osm.pbf')

for ef in all_files:
    print(ef)
    e_name = ef.split('\\')[-1].split('.')[0]
    net = og.getNetFromFile(ef, default_lanes=True, default_speed=True, default_capacity=True)
    og.outputNetToCSV(net, output_folder=url_r, prefix=e_name)

# Read MSA
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\Google_Review\Parking\tl_2019_us_cbsa\tl_2019_us_cbsa.shp')
# Read CBG Features
smart_loc = pd.read_pickle(r'F:\Research_Old\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(
    drop=True)
# smart_loc.groupby(['CBSA_Name'])['BGFIPS'].count().sort_values(ascending=False)
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])[['CBSA_Name', 'CBSA', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

# Read all road network belong to that MSA
msa_need = msa_pop.loc[msa_pop['CBSA_Name'] == 'Phoenix-Mesa-Chandler, AZ', 'CBSA'].values[0]

node_raw = pd.read_csv(url_r + 'arizona-latestnode.csv')
node_raw = gpd.GeoDataFrame(node_raw, geometry=gpd.points_from_xy(node_raw.x_coord, node_raw.y_coord), crs="EPSG:4326")

link_raw = pd.read_csv(url_r + 'arizona-latestlink.csv')
link_raw["geometry"] = gpd.GeoSeries.from_wkt(link_raw["geometry"])
link_raw = gpd.GeoDataFrame(link_raw, geometry='geometry', crs='EPSG:4326')
