preproc_calendar_sales:
	nohup python3 data-preproc.py > data-preproc.txt & 

feature_gen:
	nohup python3 create_features.py > create_features_results.txt & 

xgb_regress:
	nohup python3 xgb_regress_all_levels.py > xgb_regress_all_levels.txt & 
