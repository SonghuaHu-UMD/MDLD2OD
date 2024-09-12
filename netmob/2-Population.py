import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import glob
import contextily as ctx
import h3
import pygeohash as pgh
import rasterio
import rasterio.plot
import os
import numpy as np
import libgeohash as gh

plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Get population and device for each G5
data_url = r'D:\MDLD_OD\Netmob\OD\weekly\H37\\'
for ct_n in ['co', 'id', 'mx', 'in']:
    # Get device count
    device_count = pd.read_csv(r'D:\MDLD_OD\Netmob\PD\daily\pd_2019_daily_gh5\pd_%s_2019_agg5_daily.csv' % ct_n)
    device_count['local_date'] = pd.to_datetime(device_count['local_date'], format='%Y%m%d')
    device_count_avg = device_count.groupby('geohash_5').mean().reset_index()
    # daily_pop_weight = 4345714 / 394269.96875  # daily adjust
    # device_count_avg['no_of_unique_users'] = device_count_avg['no_of_unique_users'] * daily_pop_weight
    # device_count.groupby('local_date')['no_of_unique_users'].sum().plot()

    # Get population count for each country
    tiff = rasterio.open(r"D:\MDLD_OD\Netmob\pop\%s_ppp_2020_constrained.tif" % ct_n)
    band = tiff.read(1)
    left_bound, right_bound, top_bound, bottom_bound = tiff.bounds.left, tiff.bounds.right, tiff.bounds.top, tiff.bounds.bottom
    # all_gh5 = set(list(OD_3h['start_geohash5']) + list(OD_3h['end_geohash5']))
    all_gh5 = list(set(device_count['geohash_5']))
    print(len(all_gh5))
    all_rowcol_pd = pd.DataFrame()
    for ghs in tqdm(all_gh5):
        bbox = gh.bbox(ghs)
        all_rowcols = []
        # population count
        row0, col0 = tiff.index(bbox['e'], bbox['s'])
        row1, col1 = tiff.index(bbox['w'], bbox['n'])
        all_pop = pd.DataFrame(band[row1:row0, col1:col0])
        all_pop[all_pop < 0] = 0
        total_pop = all_pop.sum().sum()
        total_device = device_count_avg.loc[device_count_avg['geohash_5'] == ghs, 'no_of_unique_users'].values[0]
        # h3_code = h3.geo_to_h3(pgh.decode_exactly(ghs)[0], pgh.decode_exactly(ghs)[1], 7)
        all_rowcols.append([ghs, total_pop, total_device, pgh.decode_exactly(ghs)[0], pgh.decode_exactly(ghs)[1]])
        all_rowcols = pd.DataFrame(all_rowcols, columns=['geohash_5', 'population', 'devices', 'lat', 'lng'])
        all_rowcol_pd = pd.concat([all_rowcol_pd, all_rowcols])

    # Calculate weighting
    # daily_pop_weight = 4345714 / 394269.96875  # daily adjust
    # all_rowcol_pd['devices'] = all_rowcol_pd['devices'] * daily_pop_weight
    all_rowcol_pd['weights'] = all_rowcol_pd['population'] / all_rowcol_pd['devices']
    all_rowcol_pd.to_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_new_%s.pkl' % (ct_n))
    print(all_rowcol_pd['weights'].describe())
    print(all_rowcol_pd[['population', 'devices']].corr())

    # Plot two figures
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.regplot(data=all_rowcol_pd, x='devices', y='population', ax=ax,
                label='$Pearson = $' + str(round(all_rowcol_pd[['devices', 'population']].corr().values[1][0], 3)),
                color='royalblue', scatter_kws={'alpha': 0.25})
    plt.xlabel('Device count')
    plt.ylabel('Total population')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Devices_%s.pdf' % ct_n)
    plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.distplot(all_rowcol_pd.loc[all_rowcol_pd['weights'] < np.nanpercentile(all_rowcol_pd['weights'],
                                                                               99), 'weights'], ax=ax, rug=True)
    plt.axvline(x=all_rowcol_pd['weights'].mean(), color='k', linestyle='--')
    plt.axvline(x=all_rowcol_pd['weights'].median(), color='r', linestyle='--')
    plt.xlabel('Population weight')
    plt.tight_layout()
    plt.legend(['density', 'mean', 'median'])
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Weights_%s.pdf' % ct_n)
    plt.close()

# Get total population count for each study area
study_area = pd.read_excel(r"D:\MDLD_OD\Netmob\osm_url_netmob.xlsx")
study_area["geometry"] = gpd.GeoSeries.from_wkt(study_area["geometry"])
study_area = gpd.GeoDataFrame(study_area, geometry='geometry', crs='EPSG:4326')
city_list = pd.read_excel(r"D:\MDLD_OD\Netmob\City_list.xlsx")
study_area = study_area.merge(city_list, left_on='name', right_on='Name')
all_rowcols = []
for ct_n in ['co', 'id', 'mx', 'in']:
    tiff = rasterio.open(r"D:\MDLD_OD\Netmob\pop\%s_ppp_2020_constrained.tif" % ct_n)
    band = tiff.read(1)
    band[band < 0] = 0
    left_bound, right_bound, top_bound, bottom_bound = tiff.bounds.left, tiff.bounds.right, tiff.bounds.top, tiff.bounds.bottom
    # all_gh5 = set(list(OD_3h['start_geohash5']) + list(OD_3h['end_geohash5']))
    cct = 0
    study_area_need = study_area[study_area['Country_code'] == ct_n].reset_index(drop=True)
    for each in tqdm(range(0, len(study_area_need))):
        bbox = study_area_need.loc[each, 'geometry'].bounds
        city_n = study_area_need.loc[each, 'name']
        # population count
        row0, col0 = tiff.index(bbox[0], bbox[1])
        row1, col1 = tiff.index(bbox[2], bbox[3])
        all_pop = pd.DataFrame(band[row1:row0, col0:col1])
        all_pop[all_pop < 0] = 0
        total_pop = all_pop.sum().sum()
        # all_pop.shape

        # Device count
        device_count = pd.read_csv(r'D:\MDLD_OD\Netmob\PD\daily\pd_2019_daily_gh5\pd_%s_2019_agg5_daily.csv' % ct_n)
        device_count['local_date'] = pd.to_datetime(device_count['local_date'], format='%Y%m%d')
        device_count_avg = device_count.groupby('geohash_5').mean().reset_index()
        device_count_avg['geohash_5_cor'] = device_count_avg['geohash_5'].apply(pgh.decode_exactly)
        device_count_corr = pd.DataFrame(device_count_avg['geohash_5_cor'].tolist(),
                                         index=device_count_avg.index)
        device_count_corr.columns = ['lat', 'lng', 'err1', 'err2']
        device_count_avg = device_count_avg.join(device_count_corr)
        all_devices = device_count_avg[(device_count_avg['lng'] > bbox[0]) & (device_count_avg['lng'] < bbox[2]) & (
                device_count_avg['lat'] > bbox[1]) & (device_count_avg['lat'] < bbox[3])]
        total_device = all_devices['no_of_unique_users'].sum().sum()

        # # Plot city
        all_devices_df = gpd.GeoDataFrame(all_devices, geometry=gpd.points_from_xy(all_devices.lng, all_devices.lat),
                                          crs="EPSG:4326")
        fig, ax = plt.subplots(figsize=(9, 7))
        # study_area_need[study_area_need['name'] == city_n].plot(ax=ax)
        study_area_need[study_area_need['name'] == city_n].boundary.plot(ax=ax, color='k', lw=0.2)
        all_devices_df.plot(column='no_of_unique_users', cmap='RdYlGn_r', scheme="naturalbreaks", k=10, ax=ax,
                            alpha=0.6)
        ctx.add_basemap(ax, crs=all_devices_df.crs, source=ctx.providers.CartoDB.Positron, alpha=1)
        rasterio.plot.show(tiff.read(1)[row1:row0, col0:col1], extent=[bbox[0], bbox[2], bbox[1], bbox[3]], ax=ax,
                           cmap="Purples", alpha=0.5)
        plt.tight_layout()
        plt.savefig(r'D:\MDLD_OD\Netmob\results\pop_%s_%s.pdf' % (ct_n, city_n))
        plt.close()

        info_list = [total_pop, total_device, total_device / total_pop, city_n,
                     study_area_need.loc[each, 'Country_code'], study_area_need.loc[each, 'population']]
        all_rowcols.append(info_list)

all_rowcols = pd.DataFrame(all_rowcols,
                           columns=['Total_pop', 'Total_device', 'sampling_rate', 'name', 'country_code', 'pop'])
all_rowcols.to_excel(r'D:\MDLD_OD\Netmob\results\city_wt.xlsx', index=False)
# plt.plot(all_rowcols['Total_pop'],all_rowcols['Total_device'],'o')
