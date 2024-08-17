import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString
import os
from scipy.ndimage.filters import gaussian_filter1d
import contextily as ctx
import requests
import seaborn as sns
import jenkspy
import imageio
import mapclassify
import datetime
import matplotlib as mpl

pd.options.mode.chained_assignment = None

# Style for plot
plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# 5. Plot assignment outcome: speed and volume
link = gpd.read_file(r'D:\NY_Emission\Shp\osmdta_ritis.shp')
link = link.rename({'from_node_': 'from_node_id', 'link_type_': 'link_type_name'}, axis=1)
link = link.to_crs('EPSG:4326')
speed = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT\NY_TT.csv')
speed['measurement_tstamp'] = pd.to_datetime(speed['measurement_tstamp'])
# speed.groupby(['measurement_tstamp'])['speed'].mean().plot()
speed = speed[speed['measurement_tstamp'].dt.date == datetime.date(2023, 12, 5)].reset_index(drop=True)
speed['Hour'] = speed['measurement_tstamp'].dt.hour
speed = speed.groupby(['tmc_code', 'Hour'])['speed'].mean().reset_index()
binning = mapclassify.NaturalBreaks(speed['speed'], k=10)  # NaturalBreaks
speed['cut_jenks_speed'] = (binning.yb + 1) * 0.5
# speed_a = osm.merge(speed[['tmc_code', 'Hour', 'speed']], on=['tmc_code'], how='left')
# temp = speed_a.groupby(['fclass', 'Hour'])['speed'].mean().reset_index()
assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % '')
link['Start_Lon'] = link["geometry"].apply(lambda g: g.coords[0][0])
link['Start_Lat'] = link["geometry"].apply(lambda g: g.coords[0][1])
aadt_avg = pd.DataFrame()
for n_h in np.arange(0, 24):
    cams_gpd_n = cams_gpd[cams_gpd['Hour'] == n_h]
    assign_r = assign_all[assign_all['hour'] == n_h]
    speed_r = speed[speed['Hour'] == n_h]
    aadt = link.merge(speed_r[['tmc_code', 'speed', 'cut_jenks_speed']], on=['tmc_code'], how='left')
    aadt = aadt.merge(assign_r[['from_node_id', 'to_node_id', 'cut_jenks', 'volume_hourly']],
                      on=['from_node_id', 'to_node_id'], how='left')
    aadt_n = aadt[
        (aadt['Start_Lon'] < cams_gpd_n['longitude'].max()) & (aadt['Start_Lon'] > cams_gpd_n['longitude'].min())
        & (aadt['Start_Lat'] < cams_gpd_n['latitude'].max()) & (aadt['Start_Lat'] > cams_gpd_n['latitude'].min())]
    aadt_avg_t = aadt_n[aadt_n['volume_hourly'] > 1].groupby(['link_type_name'])[
        ["speed", 'volume_hourly']].mean().reset_index()
    aadt_avg_t['hour'] = n_h
    aadt_avg = pd.concat([aadt_avg, aadt_avg_t], axis=0)

    aadt["volume_hourly"] = aadt.groupby("link_type_name")["volume_hourly"].transform(lambda x: x.fillna(x.mean()))
    aadt["cut_jenks"] = aadt.groupby("link_type_name")["cut_jenks"].transform(lambda x: x.fillna(x.mean()))
    aadt["speed"] = aadt.groupby("link_type_name")["speed"].transform(lambda x: x.fillna(x.mean()))

    aadt = aadt.to_crs('EPSG:32618')
    cams_gpd_n = cams_gpd_n.to_crs('EPSG:32618')
    # aadt["speed"] = aadt["speed"].astype(int)
    # aadt["cut_jenks_speed"] = aadt.groupby("fclass")["cut_jenks_speed"].transform(lambda x: x.fillna(x.mean()))
    # aadt = aadt.dropna(subset='cut_jenks').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.5, 7))
    mpl.rcParams['text.color'] = 'white'
    # aadt.plot(ax=ax, alpha=0.3, lw=1)
    aadt.plot(column='speed', cmap='RdYlGn', scheme="user_defined",
              classification_kwds={'bins': [10, 15, 20, 30, 45]}, lw=aadt['cut_jenks'], ax=ax, alpha=0.6,
              legend=True, legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Speed (mph)',
                                        'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
                                        'framealpha': 0.5})  # 25
    # cams_gpd_n.plot(column='c3_total_s', cmap='RdYlGn_r', scheme="natural_breaks", k=6,
    #                 markersize=200 * (cams_gpd_n['c3_total_s'] / max(cams_gpd['c3_total_s'])), ax=ax, alpha=0.9,
    #                 legend=True, legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Volume',
    #                                           'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
    #                                           'framealpha': 0.5})
    plt.xlim(cams_gpd_n.bounds['minx'].min(), cams_gpd_n.bounds['maxx'].max())
    plt.ylim(cams_gpd_n.bounds['miny'].min(), cams_gpd_n.bounds['maxy'].max())
    # plt.title('Hour %s:00' % n_h)
    mpl.rcParams['text.color'] = 'k'
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
    plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    # plt.tight_layout()
    plt.axis('off')
    # plt.savefig(r'D:\NY_Emission\Figure\LVolume_camera.pdf')
    plt.savefig(r'D:\NY_Emission\Figure\LVolume_Plot_OSM_%s_%s.pdf' % (n_h, t_t))
    plt.close()
