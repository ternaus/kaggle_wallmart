#!/usr/bin/env python
__author__ = 'Vladimir Iglovikov'

'''
This script will do randomized search to find the best or almost the best parameters for this problem
for sklearn package
'''

import xgboost as xgb
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
import math
from sklearn.grid_search import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

weather = pd.read_csv('../data/weather_new3_md10.csv')
train = pd.read_csv('../data/train.csv')
key = pd.read_csv('../data/key.csv')
# test = pd.read_csv('../data/test.csv')

training = train.merge(key)
training = training.merge(weather)

target = training["units"].apply(lambda x: math.log(1 + x))
training = training.drop(["units", "date"], 1).values

def train_wm(X, y, params):
    train_idx, eval_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size=0.1, random_state=0)))
    X_train, X_eval = X[train_idx], X[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    params['validation_set'] = deval
    evals = dict()
    watchlist = [ (dtrain, 'train'), (deval, 'eval') ]
    return xgb.train(params, dtrain, 10000, watchlist, feval=evalerror,
                    early_stopping_rounds=100, evals_result=evals)




# testing = test.drop("id", 1)


from operator import itemgetter
# Utility function to report best scores
def report(grid_scores, n_top=25):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# clf = XGBRegressor()
clf = RandomForestRegressor(n_jobs=-1)
from scipy.stats import randint as sp_randint

# print help(clf)
param_dist = {"n_estimators": [200, 300],
              "max_depth": [None, 20, 30],
              "max_features": [8, 9, 10, 12],
              "min_samples_split": [1, 2, 5, 4],
              "min_samples_leaf": [5, 7, 8],
              }

random_search = RandomizedSearchCV(clf, param_dist, random_state=42, cv=3, verbose=3, n_iter=20, scoring='mean_squared_error')
fit = random_search.fit(training, target)
report(fit.grid_scores_, 20)