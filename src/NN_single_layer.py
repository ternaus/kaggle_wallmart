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

def merge_data(df):
    return ''.join([str(df["store_nbr"]), "_", str(df["item_nbr"]), "_", df["date"]])

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
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, num_features),  # 96x96 input pixels per batch
    hidden_num_units=50,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,
    eval_size=0.2,
    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
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
ylim(1e-3, 1e-2)
yscale("log")
savefig('logs/net1.eps', bbox_inches='tight')

fName = open('logs/net1.log', 'w')
print >> fName, 'train mean = ', ym
print >> fName, 'train std = ', ys
fName.close()

