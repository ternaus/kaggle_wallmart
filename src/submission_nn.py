from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano

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
def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])

def make_submission(m, test1, filename):
    prediction = m.predict(test1)
    submission = pd.DataFrame(prediction)
    submission.columns = ['units']

    submission['units'] = submission['units'].apply(lambda x: math.exp(x) - 1)
    submission['id'] = test1[["store_nbr", "item_nbr", "date"]].apply(merge_data, 1)
    submission.to_csv(os.path.join('predictions', filename), index=False)

try:
    os.mkdir('predictions')
except:
    pass


print 'reading train'
#
# weather = pd.read_csv('../data/weather_new3_mi100_md10.csv')
# train = pd.read_csv('../data/train.csv')
# key = pd.read_csv('../data/key.csv')
# training = train.merge(key)
# training = training.merge(weather)
#

weather = pd.read_csv(os.path.join('..', "data", "weather_modified_3.csv"))
train = pd.read_csv(os.path.join('..', "data", "train.csv"))
key = pd.read_csv(os.path.join('..', "data", "key.csv"))
test = pd.read_csv(os.path.join('..', "data", "test.csv"))
zero_items = pd.read_csv(os.path.join('..', 'data', 'zero_items_solid.csv'))

train_new = train.merge(zero_items)
test_new = test.merge(zero_items)
train_to_fit = train_new[train_new['units_mean'] != 0]
test_to_fit = test_new[test_new['units_mean'] != 0]

weather_new = weather.merge(key)
training = train_to_fit.merge(weather_new)
testing = test_to_fit.merge(weather_new)

features = [
#     'date',
#             'store_nbr',
            'item_nbr',
#             'units',
#             'units_mean',
            'station_nbr',
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
            'TSSN',
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

for column in features:
#     print column
    a = training[column].mean()
    training[column] = training[column].fillna(a)
    testing[column] = testing[column].fillna(a)


ly = training["units"].apply(lambda x: math.log(1 + x)).values.astype(np.float32)

ym = ly.mean()
ly.shape = (ly.shape[0], 1)
ys = ly.std()

# training = training.drop(["units", "date"], 1).values

X = training[features]

scaler = StandardScaler()
# training = scaler.fit_transform(training)
X = scaler.fit_transform(X)
#
# print 'reading test'
# test = pd.read_csv('../data/test.csv')
#
# testing = test.merge(key)
# testing = testing.merge(weather)
# testing = testing.drop(["date"], 1).values
#
# testing = scaler.transform(testing)

random_state = 42

# model = 'xgb' #eXtremeGradient Boosting

# print 'separating part of the set for later cross_validation',
# sss = ShuffleSplit(len(target), 1, test_size=0.2, random_state=random_state)
# data_index, hold_index = next(iter(sss))
#
# X_train, X_valid = training[data_index], training[hold_index]
# y_train, y_valid = target[data_index], target[hold_index]

method = 'nn'


print 'fit the model'
# layers0 = [('input', InputLayer),
#            ('dense0', DenseLayer),
#            ('dropout', DropoutLayer),
#            ('dense1', DenseLayer),
#            # ('dropout', DropoutLayer),
#            # ('dense2', DenseLayer),
#            ('output', DenseLayer),
#            ]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           # ('dropout', DropoutLayer),
           # ('dense2', DenseLayer),
           ('output', DenseLayer),
           ]



num_units = 100
num_features = X.shape[1]

clf = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=num_units,
                 dropout_p=0.5,
                 dense1_num_units=num_units,
                 # dense2_num_units=num_units,
                 output_num_units=1,

                 output_nonlinearity=None,

                 regression=True,

                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 # allow_input_downcast=True,

                 eval_size=0.2,
                 verbose=1,
                 max_epochs=100)

clf.fit(X.astype(np.float32), (ly-ym) / ys)

# make_submission(clf, testing, method + '.csv')
