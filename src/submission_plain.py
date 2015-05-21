from __future__ import division
__author__ = 'Vladimir Iglovikov'

# import pandas as pd
import numpy as np
import os
import graphlab as gl
from graphlab import SFrame
import pandas as pd

def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])



ind = False

weather = SFrame.read_csv(os.path.join('..', "data", "weather_modified_3.csv"))

if not ind:
  test = SFrame.read_csv(os.path.join('..', "data", "test.csv"))


train = SFrame.read_csv(os.path.join('..', "data", "train.csv"))
key = SFrame.read_csv(os.path.join('..', "data", "key.csv"))

zero_items = SFrame.read_csv(os.path.join('..', 'data', 'zero_items_solid.csv'))

train_new = train.join(zero_items)

if not ind:
  test_new = test.join(zero_items)

train_to_fit = train_new[train_new['units_mean'] != 0]

if not ind:
  test_to_fit = test_new[test_new['units_mean'] != 0]

weather_new = weather.join(key)
training = train_to_fit.join(weather_new)

if not ind:
  testing = test_to_fit.join(weather_new)

def f(x):
    return int(x.strip().split('-')[0])

training['year'] = training['date'].apply(f)

if not ind:
  testing['year'] = testing['date'].apply(f)

features = [
#     'date',
            'store_nbr',
            'item_nbr',
#             'units',
#             'units_mean',
#             'station_nbr',
            'tmax',
            'tmin',
            'tavg',
            'depart',
            'dewpoint',
            'wetbulb',
            'heat',
            'cool',
            'sunrise',
            'sunset',
            'snowfall',
            'preciptotal',
            'stnpressure',
            'sealevel',
            'resultspeed',
            'resultdir',
            'avgspeed',
            'HZ',
            'FU',
            'UP',
#             'TSSN',
            'VCTS',
            'DZ',
            'BR',
            'FG',
            'BCFG',
            'DU',
            'FZRA',
            'TS',
            'RA',
            'PL',
            'GS',
            'GR',
            'FZDZ',
            'VCFG',
            'PRFG',
            'FG+',
            'TSRA',
            'FZFG',
            'BLDU',
            'MIFG',
            'SQ',
            'BLSN',
            'SN',
            'SG',
            'days',
            'year']

for column in features:
  a = training[column].mean()
  training = training.fillna(column, a)
  if not ind:
    testing = testing.fillna(column, a)

import math
training['units'] = training['units'].apply(lambda x: math.log(1 + x))

sf_train, sf_test = training.random_split(0.7, seed=5)

params = {'target': 'units',
          'features':features,
          'max_iterations': 500,
          'max_depth': 8,
          'min_loss_reduction':1,
          'step_size': 0.1,
          'row_subsample': 0.8,
          'column_subsample': 0.5,
           }

if not ind:
  model = gl.boosted_trees_regression.create(sf_train, validation_set=sf_test, **params)
else:
  model = gl.boosted_trees_regression.create(training, validation_set=None, **params)
  prediction_testing = model.predict(testing)
  temp = pd.DataFrame()
  temp['id'] = testing[["store_nbr", "item_nbr", "date"]].to_dataframe().apply(merge_data, 1)
  temp['units'] = prediction_testing.apply(lambda x: math.exp(x) - 1)
  submission = pd.read_csv('../data/sampleSubmission.csv')
  result = temp.merge(submission, on=['id'], how='outer')
  result.columns = ['id', 'units', 'units_x']
  result = result.drop('units_x', 1)
  result['units'] = result['units'].fillna(0)
  result['units'] = result['units'].apply(lambda x: max(0, x))
  result.to_csv(os.path.join("predictions", "full_mean_filtered_solid_log_xgbt_mls1_ss01_md8_rs_08_station_nbr_added_89dropped.csv"), index=False)

