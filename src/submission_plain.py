from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import os
import cPickle as pickle
import gzip
import xgboost
import gl_wrapper
import numpy as np
import math
from sklearn.metrics import mean_squared_error

print 'reading train'

weather = pd.read_csv('../data/weather_new_md10.csv')
train = pd.read_csv('../data/train.csv')
key = pd.read_csv('../data/key.csv')
training = train.merge(key)
training = training.merge(weather)


target = training["units"]
training = training.drop(["units", 'date'], 1).values

scaler = StandardScaler()
training = scaler.fit_transform(training)

print 'reading test'
test = pd.read_csv('../data/test.csv')

testing = test.merge(key)
testing = testing.merge(weather)
testing = testing.drop(["date"], 1).values

testing = scaler.transform(testing)

random_state = 42

# model = 'rf' #RandomForest
#model = 'gb' #GradientBoosting
model = 'xgb' #eXtremeGradient Boosting
# model = 'xgbt'
#model = 'svm'


def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])

def make_submission(m, test1, filename):
    prediction = m.predict(test1)
    submission = pd.DataFrame(prediction)
    submission.columns = ['units']

    submission['units'] = submission['units']
    submission[submission['units'] < 0]['units'] = 0
    submission['id'] = test[["store_nbr", "item_nbr", "date"]].apply(merge_data, 1)
    submission.to_csv(os.path.join('predictions', filename), index=False)

try:
    os.mkdir('predictions')
except:
    pass


if model == 'rf':
    params =  {'max_features': 9,
               'min_samples_split': 1,
               'n_estimators': 100,
               'max_depth': None,
               'min_samples_leaf': 7,
               'n_jobs': -1,
               'random_state': random_state}

    method = 'rf_{n_estimators}_max_features{max_features}_min_samples_split{min_samples_split}_max_depth{max_depth}_min_samples_leaf{min_samples_leaf}'.format(n_estimators=params['n_estimators'],  max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], min_samples_split=params['min_samples_split'])
    clf = RandomForestRegressor(**params)
elif model == 'gb':
    params = {'n_estimators': 1000,
              'random_state': random_state}
    method = 'gb_{n_estimators}'.format(n_estimators=params['n_estimators'])
    clf = GradientBoostingRegressor(**params)
elif model == 'xgb':
    params = {'max_depth': 10,
                    'n_estimators': 100}

    method = 'xgb_{n_estimators}_md{md}'.format(md=params['max_depth'], n_estimators=params['n_estimators'])
    clf = xgboost.XGBCRegressor(**params)
elif model == 'xgbt':
    params = {'max_iterations': 300, 'max_depth': 8, 'min_child_weight': 4, 'row_subsample': 0.9, 'min_loss_reduction': 1, 'column_subsample': 0.8}
    method = 'xgbt_{max_iterations}_max_depth{max_depth}_min_loss_reduction{min_loss_reduction}_min_child_weight{min_child_weight}_row_subsample{row_subsample}_column_subsample{column_subsample}'.format(max_depth=params['max_depth'],
                                                                                                  max_iterations=params['max_iterations'],
                                                                                                  min_loss_reduction=params['min_loss_reduction'],
                                                                                                  min_child_weight=params['min_child_weight'],
                                                                                                  row_subsample=params['row_subsample'],
                                                                                                  column_subsample=params['column_subsample'])
    clf = gl_wrapper.BoostedTreesClassifier(**params)

elif model == 'svm':
    params = {'C': 5, 'cache_size': 2048}
    method = 'svm_{C}'.format(C=params['C'])
    clf = SVR(**params)

print 'fit the model'

fit = clf.fit(training, target)

print 'calculating score'
score = mean_squared_error(target, fit.predict(training))
print score


make_submission(clf, testing, method + '.csv')



try:
    os.mkdir('logs')
except:
    pass

#save score to log
fName = open(os.path.join('logs', method + '.log'), 'w')
print >> fName, 'mean squared error on the training set is: ' + str(score)
print >> fName, score
fName.close()
