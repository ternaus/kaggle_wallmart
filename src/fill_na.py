from __future__ import division
__author__ = 'Vladimir Iglovikov'

import numpy as np
import pandas as pd
import sys
'''
This file will contain functions that will fill missed values in the dataframe
'''

def fill_missed(df, label, method):
    '''

    :param df: dataframe with missed values
    :param label: column to fill
    :param method: classifier or regressor to use
    :return: df with filled na values
    '''

    #split df into df with missing values in the desired column and rest of it

    id = range(len(df.index))

    df["id"] = id

    train = df[df[label].notnull()]
    test = df[df[label].isnull()]

    train_id = train["id"]
    test_id = test["id"]

    target = train[label]
    train = train.drop(["id", label], 1)
    test = test.drop(["id", label], 1)

    cf = method()
    fit = cf.fit(train.values, target.values)
    prediction = fit.predict(test.values)

    test[label] = prediction
    train[label] = target

    train["id"] = train_id
    test["id"] = test_id
    result = pd.concat([train, test])
    result = result.sort("id")
    result = result.drop("id", 1)
    return result

def fill_missed_all(df, method):
    '''

    :param df: dataframe with missed values
    :param method: classifier or regressor to use
    :return: df with filled na values
    '''
    #create list with tuples (# of missed values, column name)

    list_with_na = []
    to_drop = []
    for column in df.columns:
        num_na = (sum(df[column].apply(np.isnan)))
        if num_na > 0:
            list_with_na += [(num_na, column)]
            to_drop += [column]

    list_with_na.sort()

    if list_with_na == []:
        return df

    temp_df = df.copy()
    temp_df = temp_df.drop(to_drop, 1)

    for i, column in list_with_na:
        print "filling " + column
        temp_df[column] = df[column]
        temp_df = fill_missed(temp_df, column, method)

    return temp_df
