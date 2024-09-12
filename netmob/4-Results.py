import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

pd.options.mode.chained_assignment = None

# Style for plot
plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


# Plot scatter
def plot_scatter(poi_msa, x_t, y_t, x_name, y_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # poi_msa = bg_pois[(bg_pois['Categories'] == poi_t)].reset_index(drop=True)
    poi_msa['size'] = (poi_msa['Total_device'] / max(poi_msa['Total_device']) * 800)
    poi_msa.loc[poi_msa['size'] < 5, 'size'] = 5
    # poi_msa = poi_msa[(poi_msa[x_t] < np.percentile(poi_msa[x_t], 99)) &
    #                   (poi_msa[x_t] > np.percentile(poi_msa[x_t], 1))].reset_index(drop=True)
    color_m = ['green', 'gray', 'royalblue', 'orange']
    ct = 0
    for kk in set(poi_msa['country']):
        poi_temp = poi_msa[poi_msa['country'] == kk]
        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=poi_temp[x_t], y=poi_temp[y_t])
        sns.regplot(x=poi_temp[x_t], y=poi_temp[y_t], color=color_m[ct],
                    scatter_kws={'s': (poi_temp['size']), 'alpha': 0.3}, ax=ax, label=kk)
        ct += 1
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)
    ax.legend()
    plt.tight_layout()


# Read all metrics
all_files = glob.glob(r'D:\MDLD_OD\Netmob\Simulation\*')
all_metrics = pd.DataFrame()
for kk in all_files:
    temp = pd.read_excel(r'%s\all_metrics.xlsx' % kk)
    all_metrics = pd.concat([all_metrics, temp])

# Merge with metro name
metro_info = pd.read_excel(r'D:\MDLD_OD\Netmob\metros.xlsx', index_col=0)
metro_info = metro_info[['Metro', 'Name', 'Population', 'Country']]
metro_info.columns = ['Metro', 'name', 'metro_population', 'country']
all_metrics = all_metrics.merge(metro_info, on='name')

all_metrics_od = all_metrics.drop_duplicates(subset='name')
all_metrics_od = all_metrics_od[
    ['country', 'Total_pop', 'Total_device', 'sampling_rate', 'name', 'country_code', 'diameter_d', 'density_v',
     'transitivity_v',
     'communities_d', 'max_comp_d', 'num_comp_d', 'max_comm_d', 'diameter_v', 'communities_v', 'max_comp_v',
     'num_comp_v', 'max_comm_v', 'self_loop', 'od_count', 'avg_dist', 'avg_duration', 'total_trips', 'PA_count']]
all_metrics_od['diameter_d'] = all_metrics_od['diameter_d'] / 1e3
all_metrics_od['avg_dist'] = all_metrics_od['avg_dist'] / 1e3
all_metrics_od['total_trips'] = all_metrics_od['total_trips'] * 10  # daily
all_metrics_od['trip_rate'] = all_metrics_od['total_trips'] / all_metrics_od['Total_device']

# Describe OD network metrics
need_li = ['total_trips', 'sampling_rate', 'trip_rate', 'Total_device', 'PA_count', 'avg_duration', 'avg_dist',
           'max_comp_v', 'max_comm_v', 'communities_v', 'num_comp_v', 'diameter_d', 'transitivity_v', 'density_v',
           'self_loop', 'Total_pop']
corr_features = all_metrics_od.groupby('country').corr(numeric_only=True, method='pearson')['Total_pop'].reset_index()
corr_features = corr_features[corr_features['level_1'].isin(need_li)]
corr_features.columns = ['country', 'variable', 'corr']
mean_features = all_metrics_od.groupby('country').mean(numeric_only=True)
mean_features = mean_features.loc[:, mean_features.columns.isin(need_li)].reset_index()
mean_features['Total_pop'] = mean_features['Total_pop'] / 1e6
mean_features = pd.melt(mean_features, id_vars='country', value_vars=need_li)
mean_features.columns = ['country', 'variable', 'mean']
corr_features = corr_features.merge(mean_features, on=['country', 'variable'])

corr_features.pivot(index='variable', columns='country', values=['mean', 'corr']).to_csv(
    r'D:\MDLD_OD\Netmob\results\corr_mean_features1.csv')

# Plot
tt_lb = ['Total Trips', 'Total Devices', 'No. ODs (Non-self-loop)', 'Avg. Duration (minute)', 'Avg. Distance (km)',
         'Max. Component Size', 'Max. Community Size', 'No. Communities', 'No. Components', 'Diameter (km)',
         'Transitivity', 'Density', 'Self-loop Rate', 'Penetration Rate']
cct = 0
for kk in ['total_trips', 'Total_device', 'PA_count', 'avg_duration', 'avg_dist', 'max_comp_v', 'max_comm_v',
           'communities_v', 'num_comp_v', 'diameter_d', 'transitivity_v', 'density_v', 'self_loop', 'sampling_rate']:
    plot_scatter(all_metrics_od, 'Total_pop', kk, 'Total Population', tt_lb[cct])
    cct += 1
    plt.savefig(r'D:\MDLD_OD\Netmob\results\%s.pdf' % tt_lb[cct])
    plt.close()

# OD and Assignment outcome
all_metrics['zero_pct'] = all_metrics['zero_length'] / all_metrics['length']
plot_scatter(all_metrics[all_metrics['link_type_name'] == 'tertiary'], 'Total_pop', 'zero_pct', 'Total Population',
             'zero_pct')
