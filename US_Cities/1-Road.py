import glob
import matplotlib.pyplot as plt
import osm2gmns as og
import pandas as pd
import geopandas as gpd
import os
import fiona
import contextily as ctx
from tqdm import tqdm
import osmnx as ox

# url_r = r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\\'
# # Read shapefile
# shp_layer = gpd.read_file(url_r + r'nhgis0011_shape\US_place_2023.shp')
# shp_layer = shp_layer.to_crs("EPSG:4326")
# # Read population
# total_pop = pd.read_csv(url_r + r'nhgis0010_csv\nhgis0018_ds267_20235_place.csv')
# total_pop = total_pop[['GISJOIN', 'ASN1E001']]
# # Merge
# shp_layer = shp_layer.merge(total_pop, on='GISJOIN')
# # Extract centroids for each city
# shp_layer['geo_lat'] = shp_layer.geometry.centroid.y  # Latitude
# shp_layer['geo_lon'] = shp_layer.geometry.centroid.x  # Longitude
# shp_layer = shp_layer.sort_values(by=['Country', 'Population'])
# shp_layer.to_file(r'D:\MDLD_OD\resilience\cities_84.shp')

shp_layer = gpd.read_file(r'D:\MDLD_OD\resilience\cities_84.shp')
shp_layer = shp_layer.sort_values(by=['ASN1E001'], ascending=False).reset_index(drop=True)
# 0-10, 100-110, 200-210, 300-310, 400-410
all_files = list(range(0, 10)) + list(range(100, 110)) + list(range(200, 210)) + list(range(300, 310)) + list(
    range(400, 410))
for kk in all_files:
    # Get network by distance
    city_center = (shp_layer.loc[kk, 'geo_lat'], shp_layer.loc[kk, 'geo_lon'])
    city_name = shp_layer.loc[kk, 'NAMELSAD']
    print('-------------- %s: Downloading --------------' % city_name)
    G = ox.graph.graph_from_point(city_center, dist=20 * 1000, dist_type='network', network_type="drive")  # meter
    print('No of edges: %s' % G.number_of_edges())
    G_gpd = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    G_gpd['highway'].value_counts()
    ef = r"D:\MDLD_OD\resilience\road_network\%s.osm" % city_name
    ox.io.save_graph_xml(G, filepath=ef)

    # Plot network
    print('-------------- %s: Plotting --------------' % city_name)
    fig, ax = ox.plot.plot_graph(G, node_size=0, show=False, close=False, edge_linewidth=0.6, edge_color='blue',
                                 edge_alpha=0.2)
    ctx.add_basemap(ax, crs=G.graph['crs'], source=ctx.providers.CartoDB.Positron)
    plt.title(city_name)
    plt.tight_layout()
    plt.savefig(r"D:\MDLD_OD\resilience\road_network\%s.png" % city_name, dpi=500)
    plt.close()

    # Convert the simulation network
    print('-------------- %s: Converting --------------' % city_name)
    net = og.getNetFromFile(
        ef, network_types=('auto',), default_lanes=True, default_speed=True, default_capacity=True,
        link_types=['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified',
                    'connector'])
    og.outputNetToCSV(net, output_folder=r"D:\MDLD_OD\resilience\road_network", prefix=city_name + '_')
