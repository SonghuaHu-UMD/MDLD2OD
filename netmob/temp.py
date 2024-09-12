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

# Get population and device for each H3 and G5
data_url = r'D:\MDLD_OD\Netmob\OD\weekly\H37\\'
for ct_n in ['co', 'id', 'mx', 'in']:
    # Get device count
    device_count = pd.read_csv(r'D:\MDLD_OD\Netmob\PD\daily\pd_2019_daily_gh5\pd_%s_2019_agg5_daily.csv' % ct_n)
    device_count['local_date'] = pd.to_datetime(device_count['local_date'], format='%Y%m%d')
    # device_count.groupby('local_date')['no_of_unique_users'].sum().plot()

    # Get population count for each country
    tiff = rasterio.open(r"D:\MDLD_OD\Netmob\pop\%s_ppp_2020_constrained.tif" % ct_n)
    band = tiff.read(1)
    left_bound, right_bound, top_bound, bottom_bound = tiff.bounds.left, tiff.bounds.right, tiff.bounds.top, tiff.bounds.bottom
    # all_gh5 = set(list(OD_3h['start_geohash5']) + list(OD_3h['end_geohash5']))
    all_gh5 = list(set(device_count['geohash_5']))
    print(len(all_gh5))
    all_rowcol_pd = pd.DataFrame()
    cct = 0
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
        all_rowcols = pd.DataFrame(all_rowcols,
                                   columns=['geohash_5', 'row', 'col', 'h3_code', 'population', 'lat', 'lng'])
        all_rowcols = all_rowcols.drop_duplicates(subset=['row', 'col'])
        all_rowcol_pd = pd.concat([all_rowcol_pd, all_rowcols])
        all_rowcol_pd = all_rowcol_pd[all_rowcol_pd['population'] > 0]
        cct += 1
        if (cct % 1000 == 0) and ct_n == 'in':
            all_rowcol_pd.to_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_%s_%s.pkl' % (ct_n, cct))
            all_rowcol_pd = pd.DataFrame()

# all_rowcol_pd = pd.concat(map(pd.read_pickle, glob.glob(r'D:\MDLD_OD\Netmob\pop\all_pop_in*')))
# all_rowcol_pd.to_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_in.pkl')

# Calculate weighting
for ct_n in ['co', 'id', 'mx', 'in']:
    all_rowcol_pd = pd.read_pickle(r'D:\MDLD_OD\Netmob\pop\all_pop_%s.pkl' % ct_n)
    tt_weight = all_rowcol_pd.groupby('geohash_5')['row'].count().mean() / (49 ** 2)  # area adjust
    all_rowcol_pd = all_rowcol_pd[all_rowcol_pd['population'] > 0].reset_index(drop=True)
    device_count = pd.read_csv(r'D:\MDLD_OD\Netmob\PD\daily\pd_2019_daily_gh5\pd_%s_2019_agg5_daily.csv' % ct_n)
    device_count['local_date'] = pd.to_datetime(device_count['local_date'], format='%Y%m%d')
    all_pop = all_rowcol_pd[all_rowcol_pd['population'] > 0]
    all_pop_gh5 = all_pop.groupby('geohash_5')['population'].sum().reset_index()
    all_pop_gh5['population'] = all_pop_gh5['population'] / tt_weight
    device_count = device_count.merge(all_pop_gh5, on='geohash_5', how='left')
    device_count_avg = device_count.groupby('geohash_5').mean()
    daily_pop_weight = 4345714 / 394269.96875  # daily adjust
    device_count_avg['no_of_unique_users'] = device_count_avg['no_of_unique_users'] * daily_pop_weight
    device_count_avg['weights'] = device_count_avg['population'] / device_count_avg['no_of_unique_users']
    # print(device_count_avg['weights'].describe())
    print(device_count_avg[['population', 'no_of_unique_users']].corr())

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    sns.regplot(data=device_count_avg, x='no_of_unique_users', y='population', ax=ax,
                label='$Pearson = $' + str(
                    round(device_count_avg[['no_of_unique_users', 'population']].corr().values[1][0], 3)),
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
    sns.distplot(
        device_count_avg.loc[
            device_count_avg['weights'] < np.nanpercentile(device_count_avg['weights'], 99), 'weights'],
        ax=ax, rug=True)
    plt.axvline(x=device_count_avg['weights'].mean(), color='k', linestyle='--')
    plt.axvline(x=device_count_avg['weights'].median(), color='r', linestyle='--')
    plt.xlabel('Population weight')
    plt.tight_layout()
    plt.legend(['density', 'mean', 'median'])
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Weights_%s.pdf' % ct_n)
    plt.close()

    # Output the population and device count by H37
    h3tgeohash = all_rowcol_pd.drop_duplicates(subset=['h3_code', 'geohash_5'])
    h3tgeohash = h3tgeohash[['geohash_5', 'h3_code']]
    device_count_avg = device_count_avg.reset_index()
    device_count_avg['weights'] = device_count_avg['weights'].fillna(device_count_avg['weights'].mean())
    h3_avg = device_count_avg.merge(h3tgeohash, on='geohash_5')
    h3_avg_weight = h3_avg.groupby('h3_code')[['no_of_unique_users', 'weights', 'population']].mean().reset_index()
    h3_avg_weight.to_pickle(r'D:\MDLD_OD\Netmob\results\h3_avg_weight_%s.pkl' % ct_n)
    print(h3_avg_weight['weights'].describe())

    ## Plot nodes and links
    # fig, ax = plt.subplots(figsize=(9, 7))
    # link.plot(ax=ax, lw=0.2, color='gray', alpha=0.5)
    # node[~node['zone_id'].isnull()].plot(ax=ax, markersize=10, color='red', alpha=1)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.close()
    # plt.show()


study_area = pd.read_excel(r"D:\MDLD_OD\Netmob\osm_url_netmob.xlsx")
study_area["geometry"] = gpd.GeoSeries.from_wkt(study_area["geometry"])
study_area = gpd.GeoDataFrame(study_area, geometry='geometry', crs='EPSG:4326')

city_list = pd.read_excel(r"D:\MDLD_OD\Netmob\City_list.xlsx")
study_area = study_area.merge(city_list, left_on='name', right_on='Name')
study_area[['name', 'population', 'Latitude/Longitude', 'Country']].to_excel(r"D:\MDLD_OD\Netmob\city_info.xlsx")

for kk in set(study_area['Country_code']):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 8))
    tem_plot = study_area[study_area['Country_code'] == kk].reset_index(drop=True)
    tem_plot.plot(ax=ax, alpha=0.1)
    tem_plot.boundary.plot(ax=ax, color='k', lw=0.2)
    tem_plot.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
    ctx.add_basemap(ax, crs=study_area.crs, source=ctx.providers.CartoDB.Positron, alpha=0.9)
    plt.tight_layout()
    plt.axis('off')