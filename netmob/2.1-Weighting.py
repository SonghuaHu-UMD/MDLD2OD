import pandas as pd
import os
import glob

all_files = glob.glob(r'D:\MDLD_OD\Netmob\Simulation_w\*')
for kk in all_files:
    assign_all = pd.read_csv(r'%s\link_performance.csv' % kk)
    assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)
    assign_all_0 = assign_all[assign_all['vehicle_volume'] > 0]
    VOC = assign_all_0.groupby('link_type')['VOC'].mean().reset_index()
    assign_all_0.groupby('link_type')['vehicle_volume'].mean().reset_index()

