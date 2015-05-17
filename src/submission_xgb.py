from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit

import pandas as pd
import os
import xgboost as xgb
import numpy as np
import math
from sklearn.metrics import mean_squared_error
#
# def evalerror(preds, dtrain):
#   labels = dtrain.get_label()
#   return 'rmse', (dtrain - preds)**2

def train_wd(X, y, params):
    train_idx, eval_idx = next(iter(ShuffleSplit(len(y), 1, test_size=0.1, random_state=random_state)))
    X_train, X_eval = X[train_idx], X[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    params['validation_set'] = deval
    evals = dict()
    watchlist = [(dtrain, 'train'), (deval, 'eval')]
    return xgb.train(params, dtrain, 10000, watchlist, early_stopping_rounds=10, evals_result=evals)

def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])

def make_submission(m, test1, filename):
    prediction = m.predict(test1)
    submission = pd.DataFrame(prediction)
    submission.columns = ['units']

    submission['units'] = submission['units'].apply(lambda x: math.exp(x) - 1)
    submission['id'] = test[["store_nbr", "item_nbr", "date"]].apply(merge_data, 1)
    submission.to_csv(os.path.join('predictions', filename), index=False)

try:
    os.mkdir('predictions')
except:
    pass


print 'reading train'

weather = pd.read_csv('../data/weather_new3_mi100_md10.csv')
train = pd.read_csv('../data/train.csv')
key = pd.read_csv('../data/key.csv')
training = train.merge(key)
training = training.merge(weather)


target = training["units"].apply(lambda x: math.log(1 + x))
training = training.drop(["units", "date"], 1).values

scaler = StandardScaler()
training = scaler.fit_transform(training)

print 'reading test'
test = pd.read_csv('../data/test.csv')

testing = test.merge(key)
testing = testing.merge(weather)
testing = testing.drop(["date"], 1).values

testing = scaler.transform(testing)

random_state = 42

model = 'xgb' #eXtremeGradient Boosting

print 'separating part of the set for later cross_validation',
sss = ShuffleSplit(len(target), 1, test_size=0.2, random_state=random_state)
data_index, hold_index = next(iter(sss))

X_train, X_valid = training[data_index], training[hold_index]
y_train, y_valid = target[data_index], target[hold_index]

params = {'max_depth': 10,
          # 'target': 'units',
          'silent': 1}

method = 'xgb_md{md}'.format(md=params['max_depth'])


print 'fit the model'

clf = train_wd(X_train, y_train, params)

print 'calculating score'

score = mean_squared_error(y_valid, clf.predict(X_valid))

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
