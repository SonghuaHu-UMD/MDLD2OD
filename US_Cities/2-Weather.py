import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from adjustText import adjust_text
import osmnx as ox
from pathlib import Path
import shutil
import contextily as ctx
from shapely.geometry import Point

# Read Cities
shp_layer = gpd.read_file(r'D:\MDLD_OD\resilience\cities_84.shp')
shp_layer = shp_layer.sort_values(by=['ASN1E001'], ascending=False).reset_index(drop=True)
un_st = ['02', '15', '60', '66', '69', '72', '78']
shp_layer = shp_layer[~shp_layer['STATEFP'].isin(un_st)].reset_index(drop=True)

# Weather stations
stations = pd.read_csv(r'G:\Data\Dewey\WEATHER\dewey_US_locations.txt', sep='|', index_col=False)
gstations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(
    stations['longitude coordinates'], stations['latitude coordinates']))
gstations = gstations.set_crs('EPSG:4326')
# all_weather = glob.glob(r'G:\Data\Dewey\WEATHER\Hourly\*')
# all_weather = pd.concat((pd.read_csv(f) for f in all_weather), ignore_index=True)
# all_weather['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'] = all_weather[
#     'OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'].astype(str)
# all_weather['year'] = all_weather['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'].str[0:4].astype(int)
# # # set(all_weather['year'])
# # for kk in [2019, 2020, 2021, 2022, 2023, 2024]:
# #     all_weather19 = all_weather[all_weather['year'] == kk].reset_index(drop=True)
# #     all_weather19['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'] = pd.to_datetime(
# #         all_weather19['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'], format='%Y%m%d%H%M')
# #     all_weather19.to_pickle(r'D:\MDLD_OD\resilience\hourly_weather_%s.pkl' % kk)
# all_weather = all_weather[
#     ['CITY_LOCATION_IDENTIFIER__UP_TO_9_ALPHANUMERIC_CHARACTERS_',
#      'OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_', 'TEMPERATURE__FLOATING_POINT___CELSIUS_',
#      'HOURLY_PRECIP____OPTIONAL___FLOATING__POINT___CENTIMETERS_']]
# all_weather['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'] = pd.to_datetime(
#     all_weather['OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_'], format='%Y%m%d%H%M')
# all_weather.to_pickle(r'D:\MDLD_OD\resilience\hourly_weather_rainfall.pkl')
all_weather = pd.read_pickle(r'D:\MDLD_OD\resilience\hourly_weather_rainfall.pkl')

# all file indexes
all_cities = list(range(0, 20)) + list(range(100, 120)) + list(range(200, 220)) + list(range(300, 320)) + list(
    range(400, 420))
url_r = r'D:\MDLD_OD\resilience\road_network'
for kk in all_cities:
    # Get network by distance
    city_center = (shp_layer.loc[kk, 'geo_lat'], shp_layer.loc[kk, 'geo_lon'])
    city_name = str(kk) + '_' + shp_layer.loc[kk, 'NAMELSAD']
    Path(r"D:\MDLD_OD\resilience\traffic_under_rainfall\%s" % city_name).mkdir(parents=True, exist_ok=True)
    print('-------------- %s: loading --------------' % city_name)
    # G_gpd['highway'].value_counts()
    ef = r"%s\%s\%s.osm" % (url_r, city_name, city_name)
    G = ox.graph_from_xml(ef)
    edge_gpd = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    node_gpd = ox.convert.graph_to_gdfs(G, nodes=True, edges=False).reset_index()

    # Pickup main rainfall events: Only rainfall, no snow.
    # https://ops.fhwa.dot.gov/publications/fhwahop13042/sec5.htm
    # Get weather data
    convex_hull = edge_gpd.geometry.unary_union.convex_hull
    convex_gdf = gpd.GeoDataFrame(geometry=[convex_hull], crs=edge_gpd.crs)
    SInTX = gpd.sjoin(gstations, convex_gdf, how='inner', predicate='within').reset_index(drop=True)
    print('No. of stations: %s' % len(SInTX))
    weather_temp = all_weather.loc[
        all_weather['CITY_LOCATION_IDENTIFIER__UP_TO_9_ALPHANUMERIC_CHARACTERS_'].isin(SInTX['ID']), [
            'OBSERVATION_TIME_IN_LOCAL_TIME__STRING_AS_YYYYMMDDHHMM_', 'TEMPERATURE__FLOATING_POINT___CELSIUS_',
            'HOURLY_PRECIP____OPTIONAL___FLOATING__POINT___CENTIMETERS_']]
    weather_temp.columns = ['Timestamp', 'temperature', 'precipitation']
    weather_temp.loc[weather_temp['precipitation'] == '*', 'precipitation'] = 0
    weather_temp['precipitation'] = weather_temp['precipitation'].astype(float)
    weather_temp['precipitation'] = weather_temp['precipitation'].fillna(0)
    weather_temp = weather_temp.groupby('Timestamp').mean().reset_index()
    weather_temp['precipitation'] = weather_temp['precipitation'] * 0.393701  # cm to inch
    weather_temp['precipitation_1h'] = weather_temp['precipitation'].shift(-1)
    weather_temp['precipitation_2h'] = weather_temp['precipitation'].shift(-2)
    weather_temp['precipitation_-1h'] = weather_temp['precipitation'].shift(1)
    weather_temp['precipitation_-2h'] = weather_temp['precipitation'].shift(2)
    weather_temp['date'] = weather_temp['Timestamp'].dt.date
    weather_temp['hour'] = weather_temp['Timestamp'].dt.hour
    weather_temp['dayofweek'] = weather_temp['Timestamp'].dt.dayofweek
    weather_temp = weather_temp[weather_temp['Timestamp'].dt.year.isin([2019, 2022, 2023, 2024])].reset_index(drop=True)
    if len(weather_temp) > 0:
        # only consider daytime and high temperature
        weather_temp.loc[(weather_temp['temperature'] > 5) & (weather_temp['hour'].isin([10, 11, 12, 13, 14, 15]))
                         & (weather_temp['precipitation'] < 0.5) & (weather_temp['precipitation'] > 0.25)
                         & (weather_temp['precipitation_1h'] < 0.25) & (weather_temp['precipitation_2h'] < 0.1)
                         & (weather_temp['precipitation_-1h'] < 0.25) & (
                                 weather_temp['precipitation_-2h'] < 0.1), 'is_selected'] = 1

        weather_temp.to_csv(r"D:\MDLD_OD\resilience\traffic_under_rainfall\%s\rainfall.csv" % city_name)
        print('No. of dates: %s' % len(sorted(set(weather_temp.loc[weather_temp['is_selected'] == 1, 'date']))))

        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(weather_temp['Timestamp'], weather_temp['precipitation'])
        need_wea = weather_temp[weather_temp['is_selected'] == 1].reset_index(drop=True)
        ax.plot(need_wea['Timestamp'], need_wea['precipitation'], 'o', color='red')
        for _, row in need_wea.iterrows():
            plt.text(row['Timestamp'], row['precipitation'], row['date'], fontsize=10, rotation=30, ha='center',
                     va='bottom')
        plt.axhline(y=0.25, color='red', linestyle='--', linewidth=2)
        plt.tight_layout()
        plt.savefig(r"D:\MDLD_OD\resilience\traffic_under_rainfall\%s\rainfall.png" % city_name)
        plt.close()

        # # Plot network
        print('-------------- %s: Plotting --------------' % city_name)
        G_proj = ox.project_graph(G, to_crs="EPSG:3857")
        gdf_ct = gpd.GeoDataFrame(geometry=[Point(city_center[1], city_center[0])], crs="EPSG:4326")
        gdf_3857 = gdf_ct.to_crs("EPSG:3857")
        fig, ax = ox.plot.plot_graph(G_proj, node_size=0, show=False, close=False, edge_linewidth=0.6, edge_color='blue',
                                     edge_alpha=0.2)
        # ax.plot(city_center[1], city_center[0], 'o', markersize=10, color='red')
        gdf_3857.plot(marker='o', color='red', markersize=100, ax=ax)
        ctx.add_basemap(ax, crs=G_proj.graph['crs'], source=ctx.providers.CartoDB.Positron)
        plt.title(city_name)
        plt.tight_layout()
        plt.savefig(r"D:\MDLD_OD\resilience\traffic_under_rainfall\%s\%s.png" % (city_name, city_name), dpi=500)
        plt.close()

        # shutil.copy(r"D:\MDLD_OD\resilience\road_network\final\%s\%s.png" % (city_name, city_name),
        #             r"D:\MDLD_OD\resilience\traffic_under_rainfall\%s\%s.png" % (city_name, city_name))
