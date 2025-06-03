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

import geopandas as gpd
import glob
from tqdm import tqdm
import fiona

layers = fiona.listlayers(r'D:\MDLD_OD\Volume\HPMS_2020.gdb')
for e_layer in tqdm(layers[50:]):
    shp_layer = gpd.read_file(r'D:\MDLD_OD\Volume\HPMS_2020.gdb', layer=e_layer)
    # shp_layer.to_file(r'D:\MDLD_OD\Volume\AADT\road_%s.shp' % e_layer)
    shp_layer.to_pickle(r'D:\MDLD_OD\Volume\HPMS_2020\%s.pkl' % e_layer.split('\\')[-1].split('.')[0])
