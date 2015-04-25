#!/usr/bin/env python
__author__ = 'Vladimir Iglovikov'

'''
This script will do randomized search to find the best or almost the best parameters for this problem
for sklearn package
'''

# import xgboost as xgb
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn import cross_validation
# from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import math
from sklearn.grid_search import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier

weather = pd.read_csv('../data/weather_filled_gl300_md10.csv')
train = pd.read_csv('../data/train.csv')
key = pd.read_csv('../data/key.csv')
# test = pd.read_csv('../data/test.csv')


training = train.merge(key)
training = training.merge(weather)

target = training["units"].values

training = training.drop(["units", "date"], 1).values
# testing = test.drop("id", 1)


from operator import itemgetter
# Utility function to report best scores
def report(grid_scores, n_top=5):
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

clf = XGBRegressor()
# clf = RandomForestRegressor(n_jobs=2)
from scipy.stats import randint as sp_randint

# print help(clf)
param_dist = {"n_estimators": [10, 20, 30],
              "max_depth": [None, 8, 10, 12, 20],
              # "max_features": sp_randint(1, 11),
              # "min_samples_split": sp_randint(1, 11),
              # "min_samples_leaf": sp_randint(1, 11),
              }

random_search = RandomizedSearchCV(clf, param_dist, random_state=42, cv=5, verbose=3)
fit = random_search.fit(training, target)
report(fit.grid_scores_)