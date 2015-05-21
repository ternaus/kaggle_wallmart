from __future__ import division
__author__ = 'Vladimir Iglovikov'
import pandas as pd
import numpy as np
import os

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

from pylab import *
import seaborn as sns
import pandas as pd
import os
import xgboost as xgb
import numpy as np
import math


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

weather = pd.read_csv(os.path.join('..', "data", "weather_modified_3.csv"))

#change some columns of weather

# weather['sunrise'] = weather['sunrise'].apply(lambda x: math.log(1 + x), 1)
# weather['sunset'] = weather['sunset'].apply(lambda x: math.log(1 + x), 1)
# weather['sealevel'] = weather['sealevel'].apply(lambda x: math.log(1 + x), 1)


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

def f(x):
    return int(x.strip().split('-')[0])

training['year'] = training['date'].apply(f)
testing['year'] = testing['date'].apply(f)

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
            'days',
            'year'
]

for column in features:
#     print column
    a = training[column].mean()
    training[column] = training[column].fillna(a)
    testing[column] = testing[column].fillna(a)

import math
ly = training['units'].apply(lambda x: math.log(1 + x)).values

ym = ly.mean()
ys = ly.std()

ly.shape = (ly.shape[0], 1)

X = training[features].values

num_features = X.shape[1]

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

from lasagne import layers
net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden0', layers.DenseLayer),
        ('dropout', DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, num_features),  # 96x96 input pixels per batch
    hidden0_num_units=200,  # number of units in hidden layer
    dropout_p=0.5,
    hidden1_num_units=200,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,
    eval_size=0.2,
    max_epochs=200,  # we want to train this many epochs
    verbose=1,
    regression=True,  # flag to indicate we're dealing with regression problem

    )

target = (ly-ym) / ys


net1.fit(X.astype(np.float32), target.astype(np.float32))

import cPickle as pickle

try:
  os.mkdir('models')
except:
  pass

with open('models/net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

try:
  os.mkdir('logs')
except:
  pass

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plot(train_loss, linewidth=3, label="train")
plot(valid_loss, linewidth=3, label="valid")
grid()
legend()
xlabel("epoch")
ylabel("loss")
ylim(0.5, 1)
# yscale("log")
savefig('logs/double.eps', bbox_inches='tight')
ylim(ymax=1, ymin=0)

fName = open('logs/double.log', 'w')
print >> fName, 'train mean = ', ym
print >> fName, 'train std = ', ys
fName.close()

