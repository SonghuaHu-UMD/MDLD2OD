import glob
import osm2gmns as og
import pandas as pd
import geopandas as gpd
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
