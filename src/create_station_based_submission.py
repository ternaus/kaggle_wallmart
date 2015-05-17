from __future__ import division

__author__ = 'Vladimir Iglovikov'

# import pandas as pd
import numpy as np
import os
import graphlab as gl
from graphlab import SFrame
import pandas as pd

submission = pd.read_csv('../data/sampleSubmission.csv')
def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])

weather = SFrame.read_csv(os.path.join('..', "data", "weather_modified_3.csv"))
train = SFrame.read_csv(os.path.join('..', "data", "train.csv"))
key = SFrame.read_csv(os.path.join('..', "data", "key.csv"))
test = SFrame.read_csv(os.path.join('..', "data", "test.csv"))

weather_new = weather.join(key)

training = train.join(weather_new)
# training = training.join(weather)
testing = test.join(weather_new)
# testing = testing.join(weather)

features = [
    # 'store_nbr',
    # 'station_nbr',
            'item_nbr',
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
            # 'TSSN',
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
    'days']



predictions = []
for station in range(1, 21):
  print 'station = ' + str(station)
  train_x = training[training['station_nbr'] == station]
  test_x = testing[testing['station_nbr'] == station]

  for column in features:
    a = train_x[column].mean()
    train_x = train_x.fillna(column, a)
    test_x = test_x.fillna(column, a)

  model = gl.boosted_trees_regression.create(train_x, target='units',
                                           features=features,
                                           max_iterations=300,
#                                            max_depth=10,
#                                            row_subsample=0.8,
                                           min_loss_reduction=1,
                                            step_size=0.01,
                                           validation_set=None)
  prediction_testing = model.predict(test_x)
  temp = pd.DataFrame()
  temp['units'] = prediction_testing
  test_x_pd = test_x.to_dataframe()
  temp['id'] = test_x_pd[["store_nbr", "item_nbr", "date"]].apply(merge_data)

  predictions += [temp]

result = pd.concat(temp)
result[result.units < 0]['units'] = 0
result.to_csv(os.path.join("predictions", "xgbt_stations_mean_ss0.01_mls1_mi300.csv"), index=False)

result_int = result.copy()
result_int['units'] = result_int['units'].astype(int)
result_int.to_csv(os.path.join("predictions", "xgbt_stations_mean_ss0.01_mls1_mi300.csv"), index=False)


