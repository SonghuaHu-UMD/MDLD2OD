import pandas as pd
import glob
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

all_dates = pd.date_range(start='2019-04-30', end='2019-06-01', freq='D').strftime('%Y-%m-%d').tolist()
all_files = glob.glob(r'G:\Veraset\Visits\*2019*.snappy.parquet')

# # Export home location
# files_201905 = [var for var in all_files for kk in all_dates if kk in var]
# dfs = []
# for file in tqdm(files_201905):
#     df = pd.read_parquet(file)
#     df = df[['caid', 'safegraph_place_id', 'census_block_group']]
#     df = df[df['safegraph_place_id'] == 'home']
#     df = df.drop_duplicates()
#     dfs.append(df)
# dfs = pd.concat(dfs, ignore_index=True)
# dfs = dfs.drop_duplicates()
# dfs.to_pickle(r'F:\MDLD_OD\MDLDod\shp\hw.pkl')

# Read home location
hw = pd.read_pickle(r'F:\MDLD_OD\MDLDod\shp\hw.pkl')
hw = hw[['caid', 'census_block_group']]
hw.columns = ['caid', 'home_cbg']
hw = hw.drop_duplicates(subset='caid')

# Read visits
need_cbgs = pd.read_pickle(r'F:\MDLD_OD\MDLDod\shp\all_ctrct.pkl')
need_cbgs = need_cbgs[0].values.tolist()
for kk in tqdm(range(7, len(all_dates))):
    need_dates = [all_dates[kk - 1], all_dates[kk], all_dates[kk + 1]]
    need_files = [var for var in all_files for kk in need_dates if kk in var]
    df = pd.concat([pd.read_parquet(f) for f in need_files], ignore_index=True)
    df = df[['caid', 'local_timestamp', 'naics_code', 'minimum_dwell', 'safegraph_place_id', 'census_block_group']]
    df.columns = ['caid', 'end_timestamp', 'end_naics_code', 'minimum_dwell', 'end_place_id', 'end_cbg']

    # Select time interval and space
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], unit='s')
    f_dt = datetime.datetime.strptime(all_dates[kk], '%Y-%m-%d')
    date_lower = f_dt - datetime.timedelta(hours=4)
    date_upper = f_dt + datetime.timedelta(hours=24 + 4)
    df = df[(df['end_timestamp'] < date_upper) & (df['end_timestamp'] > date_lower)]
    df = df.dropna(subset='end_cbg').reset_index(drop=True)
    # df = df.head(50000)

    df['end_cbg'] = df['end_cbg'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
    df = df[df['end_cbg'].str[0:11].isin(need_cbgs)].reset_index(drop=True)
    df = df.sort_values(by=['caid', 'end_timestamp']).reset_index(drop=True)
    df['next_start_timestamp'] = df['end_timestamp'] + pd.to_timedelta(df['minimum_dwell'], unit='m')

    # Hourly rate
    df['hour'] = df['end_timestamp'].dt.hour
    hour_rate = df.groupby(['end_cbg', 'hour'])['caid'].count().reset_index()
    hour_rate['date'] = f_dt
    hour_rate.to_pickle('F:\MDLD_OD\MDLDod\od\hour_rate_%s.pkl' % all_dates[kk])

    # Get trip start
    df['start_place_id'] = df.groupby('caid')['end_place_id'].shift(1)
    df['start_cbg'] = df.groupby('caid')['end_cbg'].shift(1)
    df['start_timestamp'] = df.groupby('caid')['next_start_timestamp'].shift(1)
    # Add start/end time if null: default as 1 hour
    df.loc[df['start_timestamp'].isnull(), 'start_timestamp'] = df['end_timestamp'] - datetime.timedelta(hours=1)
    df.loc[df['end_timestamp'].isnull(), 'end_timestamp'] = df['start_timestamp'] + datetime.timedelta(hours=1)

    # Add missing trip if start/end is not home:
    df['is_imputed'] = False
    date_d = f_dt.date()
    df['dep_date'] = False
    df.loc[(df['start_timestamp'] > (f_dt + datetime.timedelta(hours=3))) & (
            df['start_timestamp'] < (f_dt + datetime.timedelta(hours=24))), 'dep_date'] = True
    first_trips = df[df['dep_date']].groupby('caid').first().reset_index()
    first_indices = df[df['dep_date']].groupby('caid').head(1).index
    df.loc[first_indices, 'start_place_id'] = 'home'
    df.loc[first_indices, 'start_timestamp'] = df['end_timestamp'] - datetime.timedelta(hours=1)
    df.loc[first_indices, 'is_imputed'] = True
    # need_home_start = first_trips[first_trips['start_place_id'] != 'home']
    # print(len(need_home_start) / len(df))

    df['dep_date'] = False
    df.loc[(df['start_timestamp'] > (f_dt + datetime.timedelta(hours=3))) & (
            df['start_timestamp'] < (f_dt + datetime.timedelta(hours=24 + 3))), 'dep_date'] = True
    last_trips = df[df['dep_date']].groupby('caid').last().reset_index()
    # last_trips = last_trips[last_trips['date'] == date_d]
    need_home_end = last_trips[last_trips['end_place_id'] != 'home']
    # print(len(need_home_end) / len(df))

    end_corrections = pd.DataFrame({
        'caid': need_home_end['caid'], 'end_place_id': 'home',
        'end_timestamp': need_home_end['end_timestamp'] + datetime.timedelta(hours=1),
        'start_place_id': need_home_end['end_place_id'], 'start_cbg': need_home_end['end_cbg'],
        'start_timestamp': need_home_end['end_timestamp'], 'is_imputed': True})
    # end_corrections['start_timestamp'].isnull().sum() / len(end_corrections)
    # end_corrections['end_timestamp'].isnull().sum() / len(end_corrections)

    df_fixed = pd.concat([df, end_corrections], ignore_index=True)
    df_fixed = df_fixed.sort_values(['caid', 'end_timestamp']).reset_index(drop=True)
    df_fixed = df_fixed[df_fixed['start_timestamp'].dt.date == date_d]

    # merge with home cbg
    df_fixed = df_fixed.merge(hw, on='caid', how='left')
    # df_fixed['home_cbg'].isnull().sum()/len(df_fixed)
    df_fixed.loc[df_fixed['end_place_id'] == 'home', 'end_cbg'] = df_fixed['home_cbg']
    df_fixed.loc[df_fixed['start_place_id'] == 'home', 'start_cbg'] = df_fixed['home_cbg']

    # df_fixed[['caid', 'start_timestamp', 'start_place_id', 'end_timestamp', 'end_place_id', 'is_imputed']].to_csv(
    #     r'temp.csv')

    # Group-by OD
    df_fixed = df_fixed.dropna(subset=['start_cbg', 'end_cbg'])
    od = (df_fixed.groupby(['start_cbg', 'end_cbg'])['caid'].count()).reset_index()
    od.columns = ['start_cbg', 'end_cbg', 'flow']
    df_fixed2 = df_fixed[(df_fixed['start_timestamp'].dt.date == date_d) & (df_fixed['is_imputed'] == False)]
    od2 = (df_fixed2.groupby(['start_cbg', 'end_cbg'])['caid'].count()).reset_index()
    od2.columns = ['start_cbg', 'end_cbg', 'flow_raw']
    od = od.merge(od2, on=['start_cbg', 'end_cbg'], how='outer')
    od = od.fillna(0)
    od['date'] = f_dt
    od.to_pickle('F:\MDLD_OD\MDLDod\od\od_%s.pkl' % all_dates[kk])
    # plt.plot(od['flow'], od['flow_raw'], 'o')
    del df, df_fixed, df_fixed2

# df_fixed[['caid', 'start_place_id', 'start_cbg', 'start_timestamp', 'end_place_id', 'end_cbg', 'end_timestamp',
#           'is_imputed', 'minimum_dwell']].head(1000).to_csv('temp.csv')

# Generate OD table
all_files = glob.glob(r'F:\MDLD_OD\MDLDod\od\od*.pkl')
od_month = pd.concat([pd.read_pickle(f) for f in all_files], ignore_index=True)
# od_month.groupby(['date'])[['flow', 'flow_raw']].sum().plot()
od_flow = od_month.groupby(['start_cbg', 'end_cbg'])[['flow', 'flow_raw']].sum()
od_flow = od_flow.reset_index()
od_flow.columns = ['origin', 'destination', 'monthly_total', 'monthly_total_raw']
od_flow.to_pickle(r'F:\MDLD_OD\MDLDod\od\od_flow_all.pkl')

# Generate device count by cbg
hw = pd.read_pickle(r'F:\MDLD_OD\MDLDod\shp\hw.pkl')
hw = hw[['caid', 'census_block_group']]
hw.columns = ['caid', 'home_cbg']
hw = hw.drop_duplicates(subset='caid')
devices = hw.groupby('home_cbg')['caid'].count().reset_index()
devices.columns = ['census_block_group', 'number_devices_residing']
devices.to_pickle(r'F:\MDLD_OD\MDLDod\od\device_count.pkl')

# Generate hourly rate
all_files = glob.glob(r'F:\MDLD_OD\MDLDod\od\hour_rate*.pkl')
od_hourly = pd.concat([pd.read_pickle(f) for f in all_files], ignore_index=True)
od_hourly['date'] = od_hourly['date'] + pd.to_timedelta(od_hourly['hour'], unit='h')
od_hourly.drop('hour', axis=1, inplace=True)
od_hourly.columns = ['destination', 'hourly_flow', 'Datetime']
od_hourly.to_pickle(r'F:\MDLD_OD\MDLDod\od\od_hourly.pkl')
