from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
This script will put items that are sold in the suspecious time intervals to zero
'''

import pandas as pd
import numpy as np

print 'read train'
train = pd.read_csv('../data/train.csv')

print 'read zero items'
zero_items = pd.read_csv('../data/zero_items_solid.csv')

print 'read non filtered submission'
temp = pd.read_csv('predictions/1432595750.71.csv')

print 'modifying temp'

def get_date(x):
    result = pd.to_datetime(x.strip().split('_')[-1])
    return result

print 'keep only nonzero temp'
temp = temp[temp['units'] > 0]

temp['date'] = temp['id'].apply(get_date, 1)


def get_item(x):
    return int(x.strip().split('_')[1])

def get_store(x):
    return int(x.strip().split('_')[0])

temp['store_nbr'] = temp['id'].apply(get_store, 1)

temp['item_nbr'] = temp['id'].apply(get_item, 1)

print 'merging train'
train_zero = train.merge(zero_items)
train_zero = train_zero[train_zero['units_mean'] > 0]

print 'transform train date'
train_zero['date'] = train_zero['date'].apply(pd.to_datetime, 1)

def filter_bigger(df_s, store_nbr, item_nbr):
  '''
  All units in the submission for given store_nbr and item_nbr will be put to zero for any date that is
    bigger than maximum nonzero date in the training set

  :param df_s: dateframe that will be submitted
  :param store_nbr: store number that is needed to be filtered
  :param item_nbr: item number that needs to be set units to zero
  :return: dataframe with zeros in the nonsold time period
  '''
  print df_s.shape
  max_train_date = max(train_zero[(train_zero['item_nbr'] == item_nbr) &
                                  (train_zero['store_nbr'] == store_nbr) &
                                  (train_zero['units'] > 0)]['date'])

  df_s.loc[(df_s['item_nbr'] == item_nbr) & (df_s['store_nbr'] == store_nbr) & (df_s['date'] > max_train_date), 'units'] = 0
  return  df_s



list_to_filter = [(1, 89),
                  (5, 20), #Maybe
                  (5, 105),
                  (8, 84), #Maybe volume is low
                  (9, 42), #Maybe volume is relatively high
                  (10, 21), #Maybe volume is low
                  (10, 86),
                  (11, 86),
                  (12, 86),
                  (15, 17),
                  (15, 86),
                  (19, 69), #maybe
                  (20, 18),
                  (23, 29),
                  (25, 84),
                  (27, 21),
                  (31, 98),#slightly questinable
                  (32, 19),
                  (32, 86),
                  (34, 52),
                  (34, 84),
                  (37, 15),
                  (38, 52),
                  (38, 86),
                  (40, 51),
                  (41, 86),
                  ]

print 'filtering'

for store_nbr, item_nbr in list_to_filter:
  print store_nbr, item_nbr
  temp = filter_bigger(temp, store_nbr, item_nbr)


temp = temp.drop(['store_nbr', 'item_nbr', 'date'], 1)

submission = pd.read_csv('../data/sampleSubmission.csv')
result = temp.merge(submission, on=['id'], how='outer')

print result.head()
result = result.rename(columns={'units_x': 'units', 'units_y': 'units_x'})
result = result.drop('units_x', 1)
result['units'] = result['units'].fillna(0)


print 'saving to file'

result.to_csv('predictions/best_bigger.csv', index=False)

