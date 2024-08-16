import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import glob
import contextily as ctx
import matplotlib as mpl

data_url = r'G:\Data\Dewey\Advan\Neighborhood Patterns - US\\'


def plot_od(demand0, boundary_plot, plot_name, o_x, o_y, d_x, d_y):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    boundary_plot.boundary.plot(ax=ax, color='gray', lw=0.2)
    for kk in range(0, len(demand0)):
        ax.annotate('', xy=(demand0.loc[kk, o_x], demand0.loc[kk, o_y]),
                    xytext=(demand0.loc[kk, d_x], demand0.loc[kk, d_y]),
                    arrowprops={'arrowstyle': '->', 'lw': 5 * demand0.loc[kk, plot_name] / max(demand0[plot_name]),
                                'color': 'royalblue', 'alpha': 0.5, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    plt.xlim(demand0[o_x].min(), demand0[o_x].max())
    plt.ylim(demand0[o_y].min(), demand0[o_y].max())
    ctx.add_basemap(ax, crs=boundary_plot.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.tight_layout()


# Read MSA
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\Google_Review\Parking\tl_2019_us_cbsa\tl_2019_us_cbsa.shp')
# Read CBG Geo data
poly = gpd.GeoDataFrame.from_file(
    r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_blck_grp_2019_84.shp')
poly['BGFIPS'] = poly['GISJOIN'].str[1:3] + poly['GISJOIN'].str[4:7] + \
                 poly['GISJOIN'].str[8:14] + poly['GISJOIN'].str[14:15]
poly = poly[~poly['GISJOIN'].str[1:3].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
# Read CBG Features
smart_loc = pd.read_pickle(r'F:\Research_Old\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[0:2].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(
    drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')
# smart_loc.groupby(['CBSA_Name'])['BGFIPS'].count().sort_values(ascending=False)
msa_pop = smart_loc.drop_duplicates(subset=['CBSA_Name'])[['CBSA_Name', 'CBSA_POP']].sort_values(
    by='CBSA_POP', ascending=False).reset_index(drop=True)

all_files = glob.glob(data_url + '*DATE_RANGE_START-2024-05-01.csv.gz')

# Loop for each MSA
for emsa in range(45, 50):
    msa_name = msa_pop.loc[emsa, 'CBSA_Name']
    print("------------------- Start processing MSA: %s -----------------" % msa_name)
    need_cbg = list(smart_loc.loc[smart_loc['CBSA_Name'] == msa_name, 'BGFIPS'])
    od_flows = pd.DataFrame()
    hourly_visit_avgs = pd.DataFrame()
    hourly_visit_days = pd.DataFrame()
    for file in tqdm(all_files):
        # Read OD flow: monthly
        ng_pattern = pd.read_csv(file)
        ng_pattern = ng_pattern.dropna(subset=['AREA']).reset_index(drop=True)
        ng_pattern = ng_pattern[~ng_pattern['AREA'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        ng_pattern['AREA'] = ng_pattern['AREA'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        ng_pattern = ng_pattern[(ng_pattern['AREA'].isin(need_cbg))].reset_index(drop=True)

        # Get monthly OD flow
        ng_pattern['DEVICE_HOME_AREAS'] = ng_pattern['DEVICE_HOME_AREAS'].apply(ast.literal_eval).reset_index(drop=True)
        d_explode = pd.DataFrame([*ng_pattern['DEVICE_HOME_AREAS']], ng_pattern.index).stack() \
            .rename_axis([None, 'Origin']).reset_index(1, name='Flow')
        od_flow = ng_pattern[['AREA']].join(d_explode)
        od_flow = od_flow.dropna(subset=['Origin']).reset_index(drop=True)
        od_flow = od_flow[~od_flow['Origin'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
        od_flow['Origin'] = od_flow['Origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
        od_flow = od_flow[(od_flow['Origin'].isin(need_cbg))].reset_index(drop=True)
        od_flows = pd.concat([od_flows, od_flow])

        # Get hourly ratio
        hourly_visit = pd.DataFrame(ng_pattern['STOPS_BY_EACH_HOUR'].str[1:-1].str.split(',').tolist()).astype(float)
        date_range = [d.strftime('%Y-%m-%d %H:%M:%S')
                      for d in pd.date_range(ng_pattern.loc[0, 'DATE_RANGE_START'].split('T')[0],
                                             ng_pattern.loc[0, 'DATE_RANGE_END'].split('T')[0], freq='h')][0: -1]
        hourly_visit.columns = date_range
        hourly_visit['AREA'] = ng_pattern['AREA']
        hourly_visit_st = pd.melt(hourly_visit, id_vars=['AREA'], value_vars=date_range)
        hourly_visit_st.columns = ['AREA', 'Datetime', 'visits']
        hourly_visit_st['Datetime'] = pd.to_datetime(hourly_visit_st['Datetime'])
        hourly_visit_st['dayofweek'] = hourly_visit_st['Datetime'].dt.dayofweek
        hourly_visit_st['hour'] = hourly_visit_st['Datetime'].dt.hour
        hourly_visit_st['isweekday'] = hourly_visit_st['dayofweek'].isin([0, 1, 2, 3, 4])
        hourly_visit_avg = hourly_visit_st.groupby(['AREA', 'isweekday', 'hour'])['visits'].mean().reset_index()
        hourly_visit_avg['t_ratio'] = hourly_visit_avg['visits'] / hourly_visit_avg.groupby(['AREA', 'isweekday'])[
            'visits'].transform('sum')
        hourly_visit_avgs = pd.concat([hourly_visit_avg, hourly_visit_avgs])

        # Get weekly ratio
        hourly_visit_st['Date'] = pd.to_datetime(hourly_visit_st['Datetime'].dt.date)
        hourly_visit_day = hourly_visit_st.groupby(['AREA', 'Date'])['visits'].sum().reset_index()
        hourly_visit_day['dayofweek'] = hourly_visit_day['Date'].dt.dayofweek
        hourly_visit_day['isweekday'] = hourly_visit_day['dayofweek'].isin([0, 1, 2, 3, 4])
        hourly_visit_day = hourly_visit_day.groupby(['AREA', 'isweekday'])['visits'].mean().reset_index()
        hourly_visit_day = hourly_visit_day.pivot(index='AREA', columns='isweekday', values='visits').reset_index()
        hourly_visit_day['weekday_ratio'] = hourly_visit_day[True] / (hourly_visit_day[False] + hourly_visit_day[True])
        hourly_visit_day.columns = ['BGFIPS', 'weekend_avg', 'weekday_avg', 'weekday_ratio']
        hourly_visit_days = pd.concat([hourly_visit_day, hourly_visit_days])
    od_flows.columns = ['destination', 'origin', 'monthly_total']
    od_flows.to_csv(r'D:\MDLD_OD\MDLDod\data\%s_OD.csv' % msa_name)
    hourly_visit_avgs.columns = ['destination', 'isweekday', 'hour', 'hourly_avg', 'hourly_ratio']
    hourly_visit_avgs.to_csv(r'D:\MDLD_OD\MDLDod\data\%s_hourly_ratio.csv' % msa_name)
    hourly_visit_days.columns = ['destination', 'weekend_avg', 'weekday_avg', 'weekday_ratio']
    hourly_visit_days.to_csv(r'D:\MDLD_OD\MDLDod\data\%s_weekend_ratio.csv' % msa_name)

    # Plot PA
    od_flows = od_flows.groupby(['destination', 'origin']).sum().reset_index()
    pa_flows = od_flows.groupby(['destination'])['monthly_total'].sum().reset_index()
    pa_flows.columns = ['BGFIPS', 'attraction']
    msa_t_geo = poly.merge(pa_flows, on='BGFIPS')
    msa_t_geo = msa_t_geo.to_crs('EPSG:3857')
    msa_t_geo['area'] = msa_t_geo.area
    msa_t_geo['attraction_density'] = msa_t_geo['attraction'] / (msa_t_geo['area'] * 0.000247105)  # to acre
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    msa_t_geo.plot(column='attraction_density', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=6,
                   legend_kwds={'labelcolor': 'k', "fmt": "{:.2f}", 'ncol': 3, 'title': 'Attraction',
                                'loc': 'lower center', 'frameon': False, 'facecolor': 'k', 'edgecolor': 'k',
                                'framealpha': 0.5}, linewidth=0, edgecolor='white', alpha=0.5)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ctx.add_basemap(ax, crs=msa_t_geo.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.title(msa_name)
    plt.tight_layout()
    plt.savefig(r'D:\MDLD_OD\MDLDod\fig\%s_PA.pdf' % msa_name)
    plt.close()

    msa_t_geo['lat'] = msa_t_geo.geometry.centroid.x
    msa_t_geo['lng'] = msa_t_geo.geometry.centroid.y
    msa_t_geo_xt = msa_t_geo[['BGFIPS', 'lat', 'lng']]
    msa_t_geo_xt.columns = ['destination', 'd_lat', 'd_lng']
    od_flows = od_flows.merge(msa_t_geo_xt, on='destination')
    msa_t_geo_xt.columns = ['origin', 'o_lat', 'o_lng']
    od_flows = od_flows.merge(msa_t_geo_xt, on='origin')
    od_flows = od_flows.sort_values(by='monthly_total', ascending=False).reset_index(drop=True)

    # Plot top 1000 OD
    demand0 = od_flows[od_flows['destination'] != od_flows['origin']].head(1000).reset_index(drop=True)
    plot_od(demand0, msa_t_geo, 'monthly_total', 'o_lat', 'o_lng', 'd_lat', 'd_lng')
    plt.savefig(r'D:\MDLD_OD\MDLDod\fig\%s_OD.pdf' % msa_name)
    plt.close()
