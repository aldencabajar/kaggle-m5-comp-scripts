import pandas as pd
import numpy as np
import pickle
from scipy import stats

# import squared residuals
frcst_sq_resid = np.array(pickle.load(open("frcst_sq_resid_200512_itemlist_all.pkl", "rb")))

# import point forecasts
frcst_h = np.array(pickle.load(open("frcst_h_200512_itemlist_all.pkl", "rb")))

# get labels
labels = pickle.load(open("agg_sales_all_lvl.pkl", 'rb'))['id'].values

print([i.shape for i in [frcst_sq_resid, frcst_h, labels]])

# change squared residuals to sigma
sigma_all = np.sqrt(frcst_sq_resid)


nitem = len(frcst_h)
pred_int = [0.5, 0.67, 0.95, 0.99]
#nitem = 100

item_list_arr = []

for frcst_set, sigma_set in zip(frcst_h[0:nitem], sigma_all[0:nitem]):
    item_array = []
    
    for frcst_h, sigma_h in zip(frcst_set, sigma_set):
        h_array = np.sort(np.insert(np.concatenate(
                stats.norm.interval(pred_int, frcst_h, sigma_h), axis = 0
                ), int(4), frcst_h))
        item_array.append(h_array)
        
    item_list_arr.append(np.transpose(item_array))

df_list = []

prc_lbl = ['0.005', '0.025', '0.165','0.250', '0.500', 
'0.750', '0.835', '0.975','0.995']

for lbl, item in zip(labels, item_list_arr):    
    df_tmp = pd.DataFrame(item)
    id_label = [lbl +'_' + i for i in prc_lbl]
    df_tmp.insert(loc = 0, column = 'id', value = id_label)
    df_list.append(df_tmp)
    
df_list_bind = pd.concat(df_list).reset_index().drop('index', axis = 1)
df_list_bind.columns = ['id'] + ['F' + str(i) for i in range(1,29)] 

# change all negative values to 0
df_list_bind.iloc[:,1:] = df_list_bind.iloc[:,1:] \
.apply(lambda x: np.where(x < 0, 0, x))

print(df_list_bind.head())
print(df_list_bind.shape)

# check if all values are non-negative
(df_list_bind.iloc[:,1:].apply(lambda x: np.all(x >= 0))).all()

lbl_validation = df_list_bind['id'].apply(lambda x: x + '_validation') \
.to_frame(name = "id")

lbl_evaluation = df_list_bind['id'].apply(lambda x: x + '_evaluation').to_frame(name = "id")

submission = pd.concat(
        [pd.concat([lbl_validation, df_list_bind.iloc[:,1:]], axis = 1),
         pd.concat([lbl_evaluation, df_list_bind.iloc[:,1:]], axis = 1)]
    )

submission.to_csv("submission.csv", index = False)

print(submission.drop('id', axis = 1).isnull().sum())
print(submission.head())
print(submission.tail())
