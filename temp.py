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
