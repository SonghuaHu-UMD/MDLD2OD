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

url_r = r'D:\MDLD_OD\Roadosm\\'
e_cbsa = '12060'
link = pd.read_csv(r"D:\MDLD_OD\Simulation\%s\link.csv" % e_cbsa)
link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

# link performance
assign_all = pd.read_csv(r'D:\MDLD_OD\Simulation\%s\link_performance_s0_25nb.csv' % e_cbsa)
assign_all['volume'] = assign_all['volume'].fillna(0)
binning = mapclassify.NaturalBreaks(assign_all['volume'], k=10)  # NaturalBreaks
assign_all['cut_jenks'] = (binning.yb + 1) * 0.5
aadt = link.merge(assign_all[['from_node_id', 'to_node_id', 'cut_jenks', 'volume']],
                  on=['from_node_id', 'to_node_id'], how='left')

fig, ax = plt.subplots(figsize=(9, 7))
aadt.plot(column='volume', cmap='RdYlGn_r', scheme="natural_breaks", k=10, lw=aadt['cut_jenks'], ax=ax,
          alpha=0.6, legend=False,
          legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'loc': 'upper left', 'frameon': True,
                       'facecolor': 'k', 'edgecolor': 'k', 'framealpha': 0.5})
# ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
# plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
plt.tight_layout()
plt.axis('off')
# plt.savefig(r'D:\NY_Emission\Figure\LVolume_Plot_OSM_%s_%s.pdf' % (n_h, t_t))
# plt.close()
