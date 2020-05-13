import pandas as pd
import numpy as np
import pickle
# import squared residuals
frcst_sq_resid = np.array(pickle.load(open("frcst_sq_resid_200512_itemlist_all.pkl", "rb")))

# import point forecasts
frcst_h = np.array(pickle.load(open("frcst_h_200512_itemlist_all.pkl", "rb")))

# get labels
labels = pickle.load(open("agg_sales_all_lvl.pkl", 'rb'))['id'].values

print([i.shape for i in [frcst_sq_resid, frcst_h, labels]])

# change squared residuals to sigma
sigma_all = np.sqrt(frcst_h)



