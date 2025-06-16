# # Generate demand period
# t_p = 'am'
# demand_period = pd.DataFrame(
#     {'first_column': [0], "demand_period_id": 1, "demand_period": t_p, "notes": 'weekday',
#      "time_period": '0600_0900', "peak_time": '0830'})
# demand_period.to_csv(r"D:\MDLD_OD\Simulation\%s\demand_period.csv" % e_cbsa, index=False)
# demand_file_list = pd.DataFrame(
#     {'first_column': [0], "file_sequence_no": 1, "scenario_index_vector": 0, "file_name": "demand.csv",
#      "demand_period": t_p, "mode_type": 'auto', "format_type": "column", "scale_factor": 1,
#      "departure_time_profile_no": 1})
# demand_file_list.to_csv(r"D:\MDLD_OD\Simulation\%s\demand_file_list.csv" % e_cbsa, index=False)

# # Get network for the whole MSA
# msa_e_geo = MSA_geo.loc[MSA_geo['NAME'] == msa_name, 'geometry'].item()
# # msa_e_geo.plot()
# print('-------------- %s: Downloading Network--------------' % msa_name)
# G = ox.graph.graph_from_polygon(msa_e_geo, network_type="drive")
# print('No of edges: %s' % G.number_of_edges())
# ox.io.save_graph_xml(G, filepath=r'D:\MDLD_OD\MDLDod\network\%s_whole.osm' % msa_name)

# od_flows = pd.read_csv(r'D:\MDLD_OD\MDLDod\od\%s_OD.csv' % msa_name, index_col=0)
# od_flows['origin'] = od_flows['origin'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
# od_flows['destination'] = od_flows['destination'].astype('int64').astype(str).apply(lambda x: x.zfill(12))


if tt_weight1 > 1.5:
    print('Run DTALite Again to align the total weight!')
    mape_tts = [[1, tt_weight, tt_weight1, assign_all_nn['mape_l'].mean(), assign_all_nn['mae_l'].mean()]]
    w_start = min(tt_weight1 * 0.8, tt_weight * 1.2)
    w_end = max(tt_weight1 * 0.8, tt_weight * 1.2)
    for kk_ww in np.linspace(w_start, w_end, 5):
        od_flows_w = od_flows_s.copy()
        od_flows_w['volume'] = od_flows_w['volume'] * kk_ww

        # Run again with ODME
        output_dta_file(urk_k, 5)
        od_flows_w.to_csv(r"%s\%s\%s\demand.csv" % (url_r, urk_k, msa_name), index=False)
        subprocess.call([r"%s\%s\%s\DTALite_230915.exe" % (url_r, urk_k, msa_name)])

        # Calculate weight
        assign_all = pd.read_csv(r'%s\%s\%s\link_performance_s0_25nb.csv' % (url_r, urk_k, msa_name))
        assign_all = assign_all.merge(link[['link_id', 'AADT']], on='link_id', how='left')
        tt_weight = (((all_aadt['AADT'] * all_aadt['aadt_length']).sum() * p_ratio) /
                     (assign_all['volume'] * assign_all['distance_km'] * 1000).sum())
        assign_all_nn = assign_all[~assign_all['AADT'].isnull()]
        tt_weight1 = ((assign_all_nn['AADT'].sum() * p_ratio) / (assign_all_nn['volume']).sum())
        assign_all_nn['mape_l'] = (abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume']) /
                                   (assign_all_nn['AADT'] * p_ratio))
        assign_all_nn['mae_l'] = abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume'])
        mape_tts.append(
            [kk_ww, tt_weight, tt_weight1, assign_all_nn['mape_l'].mean(), assign_all_nn['mae_l'].mean()])

    # Get the best ratio and run again
    mape_tts = pd.DataFrame(mape_tts, columns=['kk_ww', 'weight_vmt', 'weight_vmta', 'mape_l', 'mae_l'])
    print('Best ratio: %s' % mape_tts.loc[mape_tts['mae_l'].idxmin(), 'kk_ww'])
    mape_tts.to_csv(r"%s\%s\%s\weight_loop.csv" % (url_r, urk_k, msa_name), index=False)
    # mape_tts['mape_tt'] = mape_tts['mape_tt'].abs()
    od_flows_w = od_flows_s.copy()
    od_flows_w['volume'] = od_flows_w['volume'] * mape_tts.loc[mape_tts['mae_l'].idxmin(), 'kk_ww']

    # Run again with ODME
    output_dta_file(urk_k, 10)
    od_flows_w.to_csv(r"%s\%s\%s\demand.csv" % (url_r, urk_k, msa_name), index=False)
    subprocess.call([r"%s\%s\%s\DTALite_230915.exe" % (url_r, urk_k, msa_name)])

## Align with total weight
assign_all = assign_all.merge(link[['link_id', 'AADT']], on='link_id', how='left')
# assign_all_bf.to_csv(r'F:\MDLD_OD\MDLDod\raw_data\%s\link_performance_before_w.csv' % msa_name)
tt_weight = (((all_aadt['AADT'] * all_aadt['aadt_length']).sum() * p_ratio) /
             (assign_all['volume'] * assign_all['distance_km'] * 1000).sum())
assign_all_nn = assign_all[~assign_all['AADT'].isnull()]
tt_weight1 = ((assign_all_nn['AADT'].sum() * p_ratio) / (assign_all_nn['volume']).sum())
wt_s.extend([tt_weight, tt_weight1])
assign_all_nn['mape_l'] = (abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume']) /
                           (assign_all_nn['AADT'] * p_ratio))
assign_all_nn['mae_l'] = abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume'])

urk_k = 'TWeighted_OD'
w_start = min(tt_weight1 * 0.8, tt_weight * 1.2)
w_end = max(tt_weight1 * 0.8, tt_weight * 1.2)
mape_tts = [[1, tt_weight, tt_weight1, assign_all_nn['mape_l'].mean(), assign_all_nn['mae_l'].mean()]]
for kk_ww in np.linspace(w_start, w_end, 3):
    od_flows_w = od_flows_s.copy()
    od_flows_w['volume'] = od_flows_w['volume'] * kk_ww

    # Run again with ODME
    output_dta_file(urk_k, 5)
    od_flows_w.to_csv(r"%s\%s\%s\demand.csv" % (url_r, urk_k, msa_name), index=False)
    os.chdir(r"%s\%s\%s" % (url_r, urk_k, msa_name))
    subprocess.call([r"%s\%s\%s\DTALite_230915.exe" % (url_r, urk_k, msa_name)])

    # Calculate weight
    assign_all = pd.read_csv(r'%s\%s\%s\link_performance_s0_25nb.csv' % (url_r, urk_k, msa_name))
    assign_all = assign_all.merge(link[['link_id', 'AADT']], on='link_id', how='left')
    tt_weight = (((all_aadt['AADT'] * all_aadt['aadt_length']).sum() * p_ratio) /
                 (assign_all['volume'] * assign_all['distance_km'] * 1000).sum())
    assign_all_nn = assign_all[~assign_all['AADT'].isnull()]
    tt_weight1 = ((assign_all_nn['AADT'].sum() * p_ratio) / (assign_all_nn['volume']).sum())
    assign_all_nn['mape_l'] = (abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume']) /
                               (assign_all_nn['AADT'] * p_ratio))
    assign_all_nn['mae_l'] = abs(assign_all_nn['AADT'] * p_ratio - assign_all_nn['volume'])
    mape_tts.append(
        [kk_ww, tt_weight, tt_weight1, assign_all_nn['mape_l'].mean(), assign_all_nn['mae_l'].mean()])
mape_tts = pd.DataFrame(mape_tts, columns=['kk_ww', 'weight_vmt', 'weight_vmta', 'mape_l', 'mae_l'])

import geopandas as gpd
import glob
from tqdm import tqdm
import fiona

layers = fiona.listlayers(r'D:\MDLD_OD\Volume\HPMS_2020.gdb')
for e_layer in tqdm(layers[50:]):
    shp_layer = gpd.read_file(r'D:\MDLD_OD\Volume\HPMS_2020.gdb', layer=e_layer)
    # shp_layer.to_file(r'D:\MDLD_OD\Volume\AADT\road_%s.shp' % e_layer)
    shp_layer.to_pickle(r'D:\MDLD_OD\Volume\HPMS_2020\%s.pkl' % e_layer.split('\\')[-1].split('.')[0])
