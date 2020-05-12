import numpy as np
import pandas as pd
import re
import pickle
from datetime import datetime


sales = pd.read_csv('sales_train_validation.csv')
prices = pd.read_csv('sell_prices.csv')
calendar = pd.read_csv('calendar.csv')

# create other calendar variables
calendar['year_frm'] = calendar['year'] - 2011
calendar['date1'] = calendar.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
calendar['start_date_year'] = calendar.year.apply(lambda x: datetime.strptime(str(x) + '-01-01', '%Y-%m-%d'))
calendar['day_of_year'] = calendar['date1'] - calendar['start_date_year']
calendar['day_of_year'] = calendar['day_of_year'].apply(lambda x: x.days + 1)
calendar['day_num'] = range(1, calendar.shape[0] + 1)

#create df for holiday events
calendar_event_1 = pd.get_dummies(calendar.event_name_1)
calendar_event_2 = pd.get_dummies(calendar.event_name_2)
calendar_event_1['Cinco De Mayo'] = calendar_event_1['Cinco De Mayo'] + calendar_event_2['Cinco De Mayo']
calendar_event_1['Easter'] = calendar_event_1['Easter'] + calendar_event_2['Easter']
calendar_event_1["Father's day"] = calendar_event_1["Father's day"] + calendar_event_2["Father's day"]
calendar_event_1["OrthodoxEaster"] = calendar_event_1["OrthodoxEaster"] + calendar_event_2["OrthodoxEaster"]
calendar_event_dm = calendar_event_1
calendar_event_dm['num_events'] = calendar_event_dm.apply(sum, axis = 1).values
calendar_all = pd.concat([calendar, calendar_event_dm], axis = 1)
calendar_all['snap_all'] = calendar_all[['snap_CA', 'snap_WI', 'snap_TX']].apply(sum, axis = 1) 
calendar_all['snap_all'].loc[calendar_all['snap_all'] > 0] = 1                       

## drop unnecessary columns
calendar_all.drop(
['event_name_1', 'event_type_1', 'event_name_2',
 'event_type_2','weekday', 'd','date1', 'start_date_year',
'date', 'year'],
axis = 1, inplace = True
)

days_col =  sales.columns[np.where(np.array([x.find('d_') for x in  sales.columns]) > -1)[0]]

def ts_agg(df, grp, i):
    if 'X' in grp and 'Total' not in grp:
        agg_col = grp[np.where(np.array(grp) != 'X')[0][0]]
        fn = df[np.append(agg_col, days_col)].set_index(agg_col).groupby(agg_col).apply(sum).reset_index()
        fn['id2'] = 'X'
        fn['id'] = fn[np.array([agg_col, 'id2'])].apply(lambda x: '_'.join(x), axis = 1).values
        fn.drop([agg_col, 'id2'], inplace = True, axis = 1)
        
    elif 'Total' in grp:
        df['id'] = 'Total_X'
        fn = df[np.append('id', days_col)].set_index('id').groupby('id').apply(sum).reset_index()
        
        
    else:
        fn = df[np.append(grp, days_col)].set_index(grp).groupby(grp).apply(sum).reset_index()
        fn['id'] = fn[grp].apply(lambda x: '_'.join(x), axis = 1).values
        fn.drop(grp, inplace = True, axis = 1)
    
    fn['lvl'] = i 
        
    return(fn)

def price_agg(df, grp):
    if 'X' in grp and 'Total' not in grp:
        agg_col = grp[np.where(np.array(grp) != 'X')[0][0]]
        fn = prices.groupby(grp + ['wm_yr_wk'])['sell_price'] \
        .agg({'mean'}).rename(columns = {'mean' :'sell_price'}).reset_index()
        fn['id2'] = 'X'
        fn['id'] = fn[np.array([agg_col, 'id2'])].apply(lambda x: '_'.join(x), axis = 1).values
        fn.drop([agg_col, 'id2'], inplace = True, axis = 1)
        
    elif 'Total' in grp:
        df['id'] = 'Total_X'
        fn = prices['sell_price'].agg({'mean'}).rename(columns = {'mean' :'sell_price'}).reset_index()
        
        
    else:
        fn = prices.groupby(grp + ['wm_yr_wk'])['sell_price'] \
        .agg({'mean'}).rename(columns = {'mean' :'sell_price'}).reset_index()
        fn['id'] = fn[grp].apply(lambda x: '_'.join(x), axis = 1).values
        fn.drop(grp, inplace = True, axis = 1)
    
        
    return(fn)

### DETERMINE AGGREGATED SALES ####
levels = (['Total','X'], ['state_id', 'X'], ['store_id', 'X'], ['cat_id', 'X'], ['dept_id', 'X'], 
          ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'], ['store_id', 'dept_id'], 
          ['item_id', 'X'], ['state_id', 'item_id'], ['item_id', 'store_id'])

agg_sales_list = []
for i, lv in enumerate(levels):
    agg_sales_list.append(ts_agg(sales, lv, i + 1))

agg_sales_all = pd.concat(agg_sales_list)

### DETERMINE AGGREGATED PRICE SALES ###
prices['state_id'] = prices.store_id.str.replace(r'_\d', '')
prices_item_all = []

LevelsForPrices = (['state_id','item_id'], ['item_id', 'store_id'])
for grp in LevelsForPrices:
    prices_item_all.append(price_agg(prices, grp))
    
prices_item_all = pd.concat(prices_item_all)

tmp = prices_item_all.pivot(index = "wm_yr_wk", columns ="id", values = "sell_price")
calendar_all.set_index("wm_yr_wk", inplace = True)
calendar_w_price = calendar_all.merge(tmp, right_index = True, left_index= True).reset_index()
calendar_w_price.drop('wm_yr_wk', axis = 1, inplace = True)


pickle.dump(agg_sales_all, open("agg_sales_all_lvl.pkl", "wb"))
pickle.dump(calendar_w_price, open("calendar_processed.pkl", "wb"))







