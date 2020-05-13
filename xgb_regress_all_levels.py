import pandas as pd
import numpy as np
import pickle
import time
import xgboost as xgb

agg_sales_all_lvl = pickle.load(open("agg_sales_all_lvl.pkl", "rb")) \
.reset_index().drop('index', axis = 1) 
calendar_w_price = pickle.load(open("calendar_processed.pkl", "rb")) 

days_col = ['d_'+ str(i+1) for i in range(1913)]

arr_min_sales = agg_sales_all_lvl[days_col] \
.apply(lambda x: x.values[np.min(np.where(x > 0)):], 
axis = 1)

lag = 4 # time lag to be considered
lg_sales_arr, lbl = [], []
for arr in arr_min_sales:
    lg_sales_arr.append(np.array([arr[i:i + lag] for i in range(len(arr) - lag)]))
    lbl.append(np.array([arr[i + lag] for i in range(len(arr) - lag)]))
lg_sales_arr = np.array(lg_sales_arr)
lbl = np.array(lbl)

del arr_min_sales


var_list = ['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI',
'year_frm', 'day_of_year', 'day_num', 'snap_all', 'Chanukah End',
'Christmas', 'Cinco De Mayo', 'ColumbusDay', 'Easter', 'Eid al-Fitr',
'EidAlAdha', "Father's day", 'Halloween', 'IndependenceDay', 'LaborDay',
'LentStart', 'LentWeek2', 'MartinLutherKingDay', 'MemorialDay',
"Mother's day", 'NBAFinalsEnd', 'NBAFinalsStart', 'NewYear',
'OrthodoxChristmas', 'OrthodoxEaster', 'Pesach End', 'PresidentsDay',
'Purim End', 'Ramadan starts', 'StPatricksDay', 'SuperBowl',
'Thanksgiving', 'ValentinesDay', 'VeteransDay', 'num_events']


## FUNCTION FOR GENERATING FEATURES
def feature_gen(lag_arr, id_, lvl) :
    start_day = 1913 - len(lag_arr) 
    if lvl in (2, 6, 7, 11):
        if id_.find('CA') == 0:
            snap = 'snap_CA'
        elif id_.find('WI') == 0:
            snap = 'snap_WI'
        elif id_.find('TX') == 0:
            snap = 'snap_TX'
    else:
        # for levels other than 2,6, 7 and 11, adapt the snap_all var
        snap = 'snap_all'
    
    if lvl >= 11:
        new_var_list = var_list +[snap] + [id_]
    else:
        new_var_list = var_list + [snap]
    
    feat_train = np.concatenate((lag_arr, calendar_w_price.loc[start_day:1912, new_var_list].values, 
                np.reshape(np.arange(start_day + 1, 1914), (len(range(start_day, 1912)) + 1, 1))), axis = 1)
    h_feat = np.append(calendar_w_price.loc[1913:,new_var_list].values,
                       np.reshape(np.arange(1914, 1970), (len(range(1914, 1969)) + 1, 1)), axis = 1)
    
    return feat_train, h_feat

preds_list  = []
log_resid_list = []
frcst_h = []
frcst_sq_resid = []

# number of items to be fit
nitem = len(lg_sales_arr) 

# model for mean regression
xgb_m = xgb.XGBRegressor(n_estimators = 350, max_depth = 2, learning_rate = 0.1,
                             booster = 'gbtree')
# model for variance regression
xgb_res = xgb.XGBRegressor(n_estimators = 350, max_depth = 2, learning_rate = 0.1,booster = 'gbtree')

def fit_pred_multistep(arr, lbl_, x_feat_mt):    

    xgb_m.fit(arr, lbl_)
       
    # getting residuals and fitting it  on the model
    preds = xgb_m.predict(arr)
    sqr_resid = np.square(lbl_ - preds)
    l2 = np.min(sqr_resid[sqr_resid > 0])/2
    log_resid = np.log(np.square(lbl_ - preds) + l2)    
    xgb_res.fit(X = arr, y = log_resid)
    
    frcst_val = np.empty(28)
    sq_resid_val = np.empty(28)
    lg_arr_frcst = list(np.append(arr[len(arr) - 1][1:4], lbl_[len(arr) - 1]))
    for i in range(28):
        X = np.append(np.array(lg_arr_frcst), x_feat_mt[i])
        X = np.reshape(X, (1, len(X)))
        
        #### for point forecast ####
        pred_h = xgb_m.predict(X)
        frcst_val[i] = pred_h
        lg_arr_frcst.append(pred_h) #append at the tail the new prediction
        lg_arr_frcst.pop(0) # remove 0 index forecast
        
        
        #### for variance forecast ####
        resid_h = np.exp(xgb_res.predict(X) - l2)
        sq_resid_val[i] = resid_h
        
    
    # fitting the squared residuals
    return preds, log_resid, frcst_val, sq_resid_val

start = time.time()
for i in range(nitem):
    feat_train, h_feat = feature_gen(lg_sales_arr[i], agg_sales_all_lvl.loc[i,'id'], 
                                     agg_sales_all_lvl.loc[i, 'lvl'])
    p, r, f, rf = fit_pred_multistep(feat_train, lbl[i], h_feat)
    preds_list.append(p)
    log_resid_list.append(r)
    frcst_h.append(f)
    frcst_sq_resid.append(rf)
    
end = time.time()
print(end - start)

pickle.dump(preds_list, open('preds_200512_itemlist_all.pkl', 'wb'))
pickle.dump(log_resid_list, open('log_resid_200512_itemlist_all.pkl', 'wb'))
pickle.dump(frcst_h, open('frcst_h_200512_itemlist_all.pkl', 'wb'))
pickle.dump(frcst_sq_resid, open('frcst_sq_resid_200512_itemlist_all.pkl', 'wb'))

