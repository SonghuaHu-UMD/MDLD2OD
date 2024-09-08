import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import glob
import contextily as ctx
import h3
import pygeohash as pgh
import rasterio
import rasterio.plot
import numpy as np
import libgeohash as gh

data_url = r'D:\MDLD_OD\Netmob\OD\weekly\H37\\'
ct_n = 'in'
# # Read OD
# OD_weekly_h37 = pd.read_csv(data_url + 'od_week_h37_mx_2019.csv')
# OD_weekly_h37['start_h3_7_cor'] = OD_weekly_h37['start_h3_7'].apply(h3.h3_to_geo)
# OD_weekly_h37['end_h3_7_cor'] = OD_weekly_h37['end_h3_7'].apply(h3.h3_to_geo)

# # Get hourly ratio
# OD_3h = pd.read_csv(r'D:\MDLD_OD\Netmob\OD\3h\GH5\od_3h_gh5_mx_2019.csv')
# OD_3h['start_geohash5_cor'] = OD_3h['start_geohash5'].apply(pgh.decode)
# OD_3h['end_geohash5_cor'] = OD_3h['end_geohash5'].apply(pgh.decode)
# OD_3h['local_time_dt'] = pd.to_datetime(OD_3h['local_time'].str[0:-11])
# # - pd.to_timedelta(OD_3h['local_time'].str[-11:])
# OD_3h['hour'] = OD_3h['local_time_dt'].dt.hour
# OD_3h['dayofweek'] = OD_3h['local_time_dt'].dt.dayofweek
# hour_ratio = OD_3h.groupby(['dayofweek', 'hour'])['trip_count'].mean()
# hour_ratio.plot(marker='o', color='blue')
# hour_ratio = (hour_ratio / hour_ratio.groupby('dayofweek').sum()).reset_index()

# Get device count
device_count = pd.read_csv(r'D:\MDLD_OD\Netmob\PD\daily\pd_2019_daily_gh5\pd_%s_2019_agg5_daily.csv' % ct_n)
device_count['local_date'] = pd.to_datetime(device_count['local_date'], format='%Y%m%d')
# device_count.groupby('local_date')['no_of_unique_users'].sum().plot()

# Get population count
tiff = rasterio.open(r"D:\MDLD_OD\Netmob\pop\%s_ppp_2020_constrained.tif" % ct_n)
band = tiff.read(1)
left_bound, right_bound, top_bound, bottom_bound = tiff.bounds.left, tiff.bounds.right, tiff.bounds.top, tiff.bounds.bottom
# all_gh5 = set(list(OD_3h['start_geohash5']) + list(OD_3h['end_geohash5']))
all_gh5 = set(device_count['geohash_5'])
print(len(all_gh5))
all_rowcol_pd = pd.DataFrame()
for ghs in tqdm(all_gh5):
    bbox = gh.bbox(ghs)
    all_rowcols = []
    for lat in np.arange(bbox['s'], bbox['n'], 0.0007):
        for lng in np.arange(bbox['w'], bbox['e'], 0.0001):
            if lng < left_bound or lng > right_bound or lat > top_bound or lat < bottom_bound:
                print('out of bounds')
            else:
                row, col = tiff.index(lng, lat)
                population = band[row, col]
                h3_code = h3.geo_to_h3(lat, lng, 7)
                all_rowcols.append([ghs, row, col, h3_code, population, lat, lng])
    all_rowcols = pd.DataFrame(all_rowcols, columns=['geohash_5', 'row', 'col', 'h3_code', 'population', 'lat', 'lng'])
    all_rowcols = all_rowcols.drop_duplicates(subset=['row', 'col'])
    all_rowcol_pd = pd.concat([all_rowcol_pd, all_rowcols])
    all_rowcol_pd = all_rowcol_pd[all_rowcol_pd['population'] > 0]
all_rowcol_pd.to_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_%s.pkl' % ct_n)

# Compare
all_pop = all_rowcol_pd[all_rowcol_pd['population'] > 0]
all_pop_gh5 = all_pop.groupby('geohash_5')['population'].sum().reset_index()
device_count = device_count.merge(all_pop_gh5, on='geohash_5', how='left')
device_count_avg = device_count.groupby('geohash_5').mean()
# plt.plot(device_count_avg['population'], device_count_avg['no_of_unique_users'], 'o')
print(device_count_avg[['population', 'no_of_unique_users']].corr())
# sns.distplot(device_count_avg['population'] / device_count_avg['no_of_unique_users'], kde=False, rug=True)
