#!/usr/bin/env python

from __future__ import division

__author__ = 'Vladimir Iglovikov'

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import cross_validation

train = pd.read_csv('../data/train_empty.csv')

features = ['store_nbr',  'item_nbr', #  'units',  'station_nbr',
 'tmax',  'tmin',  'tavg',  'depart',  'dewpoint',  'wetbulb',
 'heat',  'cool',  'snowfall',  'preciptotal',  'stnpressure',
 'sealevel',  'resultspeed',  'resultdir',  'avgspeed',
 'HZ',  'FU',  'UP',  'TSSN',  'VCTS',  'DZ',  'BR',  'FG',
 'BCFG',  'DU',  'FZRA',  'TS',  'RA',  'PL',  'GS',  'GR',
 'FZDZ',  'VCFG',  'PRFG',  'FG+',  'TSRA',  'FZFG',  'BLDU',
 'MIFG',  'SQ',  'BLSN',  'SN',  'SG',
#  'month',
#  'day',
 'day_length']
#  'sunset_hour',
#  'sunset_minute',
#  'sunrise_hour',
#  'sunrise_minute']

import xgboost

X = xgboost.DMatrix(train[features].values, missing=np.nan)
y = train["units"].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
clf = XGBRegressor(silent=False)


print clf.score(X_test, y_test)
