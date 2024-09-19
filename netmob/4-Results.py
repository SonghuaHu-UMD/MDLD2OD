import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import igraph
from collections import Counter

from cairosvg.defs import marker
from tqdm import tqdm

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
def plot_scatter(poi_msa, x_t, y_t, x_name, y_name, x_sci=True, y_sci=True):
    fig, ax = plt.subplots(figsize=(5, 5))
    if x_sci:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    if y_sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    poi_msa['size'] = (poi_msa['Total_device'] / max(poi_msa['Total_device']) * 800)
    poi_msa.loc[poi_msa['size'] < 5, 'size'] = 5
    color_m = ['green', 'gray', 'royalblue', 'orange']
    ct = 0
    for kk in set(poi_msa['country']):
        poi_temp = poi_msa[poi_msa['country'] == kk]
        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=poi_temp[x_t], y=poi_temp[y_t])
        sns.regplot(x=poi_temp[x_t], y=poi_temp[y_t], color=color_m[ct], ci=80,
                    scatter_kws={'s': (poi_temp['size']), 'alpha': 0.3}, ax=ax, label=kk)
        ct += 1
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)
    ax.legend()
    plt.tight_layout()


def bpr_function(x, ffs, alpha, beta):
    return ffs / (1 + alpha * np.power(x, beta))


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
all_metrics['length'] = all_metrics['length'] / 1e3  # to km
all_metrics['zero_length'] = all_metrics['zero_length'] / 1e3  # to km

# # Car ownership estimation
# car_ower = pd.DataFrame([{'India': 57, 'Mexico': 370, 'Indonesia': 82, 'Colombia': 296}]).T
# car_ower = car_ower.reset_index()
# car_ower.columns = ['country', 'car_per1000']
# all_metrics = all_metrics.merge(car_ower, on='country')
# all_metrics['car_total'] = all_metrics['car_per1000'] * all_metrics['Total_pop'] / 1e3
# car_owers = all_metrics[all_metrics['link_type_name'] == 'primary'].reset_index(drop=True)
# car_owers['car_total_rt'] = np.log(car_owers['car_total'] / 1e3) / np.log(car_owers['car_total'].max() / 1e3)
# car_owers['primary_volume'] = 2000 * car_owers['car_total_rt']
# car_owers['ct_weight'] = car_owers['primary_volume'] / car_owers['volume1']
# car_owers[['ct_weight', 'name', 'country_code']].to_excel(r'D:\MDLD_OD\Netmob\car_weight.xlsx')

######## 1. Analyze OD network ########
all_metrics_od = all_metrics.drop_duplicates(subset='name')
all_metrics_od = all_metrics_od[
    ['country', 'Total_pop', 'Total_device', 'sampling_rate', 'name', 'country_code', 'diameter_d', 'density_v',
     'transitivity_v', 'communities_d', 'max_comp_d', 'num_comp_d', 'max_comm_d', 'diameter_v', 'communities_v',
     'max_comp_v', 'num_comp_v', 'max_comm_v', 'self_loop', 'od_count', 'avg_dist', 'avg_duration', 'total_trips',
     'PA_count']]
all_metrics_od['diameter_d'] = all_metrics_od['diameter_d'] / 1e3  # km
all_metrics_od['avg_dist'] = all_metrics_od['avg_dist'] / 1e3  # km
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

# Plot OD
tt_lb = ['Total Trips', 'Total Devices', 'No. ODs (Non-self-loop)', 'Avg. Duration (minute)', 'Avg. Distance (km)',
         'Max. Component Size', 'Max. Community Size', 'No. Communities', 'No. Components', 'Diameter (km)',
         'Transitivity', 'Density', 'Self-loop Rate', 'Penetration Rate']
cct = 0
for kk in ['total_trips', 'Total_device', 'PA_count', 'avg_duration', 'avg_dist', 'max_comp_v', 'max_comm_v',
           'communities_v', 'num_comp_v', 'diameter_d', 'transitivity_v', 'density_v', 'self_loop', 'sampling_rate']:
    plot_scatter(all_metrics_od, 'Total_pop', kk, 'Total Population', tt_lb[cct])
    # cct += 1
    plt.savefig(r'D:\MDLD_OD\Netmob\results\%s.pdf' % tt_lb[cct])
    plt.close()
    cct += 1

######## 2. Analyze Link features ########
all_metrics_link = all_metrics.groupby(['country', 'link_type_name'])[['length', 'link_count']].mean().reset_index()
all_metrics.groupby(['link_type_name'])[['length']].sum() / all_metrics.groupby(['link_type_name'])[
    ['length']].sum().sum()
all_metrics.groupby(['country', 'link_type_name'])[['length']].sum() / all_metrics.groupby(['country'])[
    ['length']].sum()
all_metrics_links = all_metrics_link.groupby(['country'])[['length', 'link_count']].sum().reset_index()
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics, x='link_type_name', y='length', hue='country', palette=sns.color_palette('coolwarm', 4))
plt.ylabel('Length (km)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\Link_length.pdf')

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics, x='link_type_name', y='link_count', hue='country', palette=sns.color_palette('coolwarm', 4))
plt.ylabel('No. links')
plt.xlabel('')
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\Link_number.pdf')

# Analyze link as a network
link_networks = pd.DataFrame()
all_files = glob.glob(r'D:\MDLD_OD\Netmob\Simulation\*')
city_list = pd.read_excel(r'D:\MDLD_OD\Netmob\City_list.xlsx')
for kk in tqdm(all_files):
    # kk=all_files[0]
    e_name = kk.split('\\')[-1].split('_')[0]
    e_ct = city_list.loc[city_list['Name'] == e_name, 'Country'].values[0]
    assign_all = pd.read_csv(r'%s\link_performance.csv' % kk)
    assign_all['vehicle_volume'] = assign_all['vehicle_volume'].fillna(0)
    assign_all_2 = assign_all[assign_all['vehicle_volume'] > 0]

    # Two graph
    graph_all1 = assign_all.drop_duplicates(subset=['from_node_id', 'to_node_id'])[
        ['from_node_id', 'to_node_id', 'distance_km']]
    OD_Graph1 = igraph.Graph.TupleList([tuple(x) for x in graph_all1.values], directed=True,
                                       edge_attrs=['distance_km'])
    # communities1 = OD_Graph1.community_infomap(edge_weights='distance_km')
    components1 = OD_Graph1.connected_components(mode='weak')

    graph_all2 = assign_all_2.drop_duplicates(subset=['from_node_id', 'to_node_id'])[
        ['from_node_id', 'to_node_id', 'distance_km']]
    OD_Graph2 = igraph.Graph.TupleList([tuple(x) for x in graph_all2.values], directed=True,
                                       edge_attrs=['distance_km'])
    # communities2 = OD_Graph2.community_infomap(edge_weights='distance_km')
    components2 = OD_Graph2.connected_components(mode='weak')

    # OD_Graph1.diameter(directed=False, weights='distance_km'),
    dist_nt = [e_name, e_ct, OD_Graph1.density(), OD_Graph1.transitivity_undirected(), max(components1.sizes()),
               len(components1.sizes()), sum(graph_all1['distance_km']), len(graph_all1),
               OD_Graph2.density(), OD_Graph2.transitivity_undirected(), max(components2.sizes()),
               len(components2.sizes()), sum(graph_all2['distance_km']), len(graph_all2)]
    od_network = pd.DataFrame([dist_nt])
    od_network.columns = ['name', 'country', 'Density_b', 'Transitivity_b', 'Max_component_size_b', 'No_components_b',
                          'Total_road_length_b', 'No_links_b', 'Density_a', 'Transitivity_a', 'Max_component_size_a',
                          'No_components_a', 'Total_road_length_a', 'No_links_a', ]
    link_networks = pd.concat([od_network, link_networks])

# Before after comparison: road network
for kk in ['Density', 'Transitivity', 'Max_component_size', 'No_components', 'Total_road_length', 'No_links']:
    link_networks[kk] = 100 * (link_networks[kk + '_a'] - link_networks[kk + '_b']) / link_networks[kk + '_b']
link_networks_diff = pd.melt(link_networks, id_vars=['name', 'country'],
                             value_vars=['Density', 'Transitivity', 'Max_component_size', 'No_components',
                                         'Total_road_length', 'No_links'])
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(link_networks_diff, y='variable', x='value', hue='country', palette=sns.color_palette('coolwarm', 4),
            orient='h')
plt.xlabel('Difference (%)')
plt.ylabel('')
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig(r'D:\MDLD_OD\Netmob\results\Link_diff.pdf')

# By link type
all_metrics_ass = all_metrics[
    ~((all_metrics['link_type_name'] == 'motorway') & (all_metrics['country'] == 'Colombia'))]
all_metrics_ass['zero_pct'] = 100 * (all_metrics_ass['zero_length'] / all_metrics_ass['length'])
all_metrics_ass.groupby(['link_type_name'])['zero_pct'].mean()
all_metrics_ass.groupby(['country'])['zero_pct'].mean()
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics_ass, y='link_type_name', x='zero_pct', hue='country', palette=sns.color_palette('coolwarm', 4),
            orient='h')
plt.xlabel('Pct. of inactive links (%)')
plt.ylabel('')
plt.tight_layout()
plt.legend(loc='upper right')
plt.xlim([0, 130])
plt.savefig(r'D:\MDLD_OD\Netmob\results\Link_diff_type.pdf')

# Relationship with other factors
corr_features = all_metrics_ass.groupby(['country', 'link_type_name']).corr(numeric_only=True, method='pearson')[
    'zero_pct'].reset_index()
corr_features = corr_features[corr_features['level_2'].isin(['sampling_rate', 'max_comp_v', 'Total_pop', ])]
corr_features.groupby('link_type_name')['zero_pct'].mean()
for roadty in set(all_metrics_ass['link_type_name']):
    plot_scatter(all_metrics_ass[all_metrics_ass['link_type_name'] == roadty], 'sampling_rate', 'zero_pct',
                 'Penetration rate', 'Inactive %s ' % roadty + 'roads (%)', False, False)
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Inactive_%s_%s.pdf' % (roadty, 'sampling_rate'))
    plt.close()
    plot_scatter(all_metrics_ass[all_metrics_ass['link_type_name'] == roadty], 'max_comp_v', 'zero_pct',
                 'Max. component size', 'Inactive %s ' % roadty + 'roads (%)', True, False)
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Inactive_%s_%s.pdf' % (roadty, 'max_comp_v'))
    plt.close()
    plot_scatter(all_metrics_ass[all_metrics_ass['link_type_name'] == roadty], 'Total_pop', 'zero_pct',
                 'Total population', 'Inactive %s ' % roadty + 'roads (%)', True, False)
    plt.savefig(r'D:\MDLD_OD\Netmob\results\Inactive_%s_%s.pdf' % (roadty, 'Total_pop'))
    plt.close()
    # plt.ylim([0, 1])

######## 3. Analyze assignment outcome ########
all_metrics.groupby(['country', 'link_type_name'])[['volume', 'speed', 'VOC_mean']].mean().reset_index()
all_metrics.groupby(['link_type_name'])[['volume', 'speed', 'VOC_mean']].mean().reset_index()

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics_ass, x='link_type_name', y='volume', hue='country', palette=sns.color_palette('coolwarm', 4))
plt.ylabel('Volume')
plt.xlabel('')
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\Volume.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics_ass, x='link_type_name', y='speed', hue='country', palette=sns.color_palette('coolwarm', 4))
plt.ylabel('Speed (kmh)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\Speed.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
sns.barplot(all_metrics_ass, x='link_type_name', y='VOC_mean', hue='country', palette=sns.color_palette('coolwarm', 4))
plt.ylabel('VOC')
plt.xlabel('')
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\VOC.pdf')
plt.close()

# Get UE curve
link_networks = pd.DataFrame()
all_files = glob.glob(r'D:\MDLD_OD\Netmob\Simulation\*')
city_list = pd.read_excel(r'D:\MDLD_OD\Netmob\City_list.xlsx')
ue_crs = pd.DataFrame()
for kk in tqdm(all_files):
    # kk=all_files[0]
    e_name = kk.split('\\')[-1].split('_')[0]
    e_ct = city_list.loc[city_list['Name'] == e_name, 'Country'].values[0]
    ue_cr = pd.read_csv(r"D:\MDLD_OD\Netmob\Simulation\%s\log_DTA.txt" % e_name, header=None, delimiter="\t")
    cpu_time = ue_cr[ue_cr[0].str.contains('CPU running time for the entire process')]
    print(e_name)
    print(cpu_time.values[1])
    ue_cr = ue_cr[ue_cr[0].str.contains('DATA INFO')]
    ue_cr = ue_cr[ue_cr[0].str.contains(r'[DATA INFO] \d+')]
    ue_cr = ue_cr[~ue_cr[0].str.contains('Cumulative|zone id')]
    ue_cr = ue_cr[0].str.split(' +', expand=True)
    ue_cr = ue_cr.loc[:, 2:7]
    ue_cr.columns = ['Iteration', 'CPU', 'Travel time', 'TT', 'Least', 'UE_Gap']
    ue_cr['Metro'] = city_list.loc[city_list['Name'] == e_name, 'Metro'].values[0]
    ue_cr['country'] = e_ct
    ue_crs = pd.concat([ue_crs, ue_cr], ignore_index=True)

ue_crs['UE_Gap'] = ue_crs['UE_Gap'].astype(float)
ue_crs['Iteration'] = ue_crs['Iteration'].astype(int)
metro_pop = all_metrics[['Metro', 'Total_pop', 'Total_device']].drop_duplicates(subset='Metro')
ue_crs = ue_crs.merge(metro_pop, on='Metro')

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.lineplot(ue_crs, x='Iteration', y='UE_Gap', hue='country', palette='Set2', marker='o')
plt.ylabel('UE Gap (%)')
plt.xlabel('Iteration')
plt.ylim([0, 15])
plt.tight_layout()
plt.savefig(r'D:\MDLD_OD\Netmob\results\UE_Gap.pdf')
plt.close()

ue_crsl = ue_crs[ue_crs['Iteration'] == 19]
ue_crsl = ue_crsl[ue_crsl['UE_Gap'] < 0.8]
plot_scatter(ue_crsl, 'Total_pop', 'UE_Gap', 'Total population', 'UE Gap (%)', x_sci=True, y_sci=True)
plt.savefig(r'D:\MDLD_OD\Netmob\results\Last_UE_Gap.pdf')
plt.close()

# fig, ax = plt.subplots(figsize=(6.5, 5))
# sns.barplot(ue_crsl, y='city', x='UE_Gap', hue='country', orient='h')
# plt.ylabel('UE Gap (%)')
# plt.xlabel('Iteration')
# # plt.ylim([0, 15])
# plt.tight_layout()
# plt.savefig(r'D:\MDLD_OD\Netmob\results\UE_Gap.pdf')
# plt.close()
