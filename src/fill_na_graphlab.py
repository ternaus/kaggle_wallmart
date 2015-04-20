from __future__ import division
__author__ = 'Vladimir Iglovikov'

import numpy as np
# import pandas as pd
import sys
import graphlab as gl

'''
This file will contain functions that will fill missed values in the dataframe
'''

def fill_missed(df, label, features, verbose=False, summary=True):
    '''

    :param df: dataframe with missed values
    :param label: column to fill
    :param method: classifier or regressor to use
    :return: df with filled na values
    '''

    #split df into df with missing values in the desired column and rest of it


    train, test = df.dropna_split(label)
    features = features[:]


    model = gl.boosted_trees_regression.create(train, target=label,
                                               features=features,
                                               # max_iterations=300,
                                               verbose=verbose,
                                               # max_depth=10,
                                               # step_size=0.1
                                               )

    if summary:
        print model.summary()

    prediction = model.predict(test)

    test[label] = prediction

    result = train.append(test)
    return result

def fill_missed_all(df, features, verbose=False, summary=True):
    '''

    :param df: dataframe with missed values    
    :return: df with filled na values
    '''
    #create list with tuples (# of missed values, column name)

    list_with_na = []
    features_new = features[:]

    for column in features:                      
        num_na = sum(df[column].apply(lambda x: np.isnan(x)))
        print column, num_na
        if num_na > 0:
            list_with_na += [(num_na, column)]
	    features_new.remove(column)

    features = features_new
    list_with_na.sort()

    if list_with_na == []:
        return df	
    print features
    temp_df = df
    for i, column in list_with_na:
        print "filling " + column + " na = " + str(i)
        a = temp_df[column].apply(lambda x: np.isnan(x)).astype(int)        

        temp_df = fill_missed(temp_df, column, features, verbose=verbose, summary=summary)
        temp_df[ column + "NAN" ] = a
        features += [column]
        features += [column+"NAN"]

    return temp_df
