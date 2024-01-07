# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import pandas as pd
from data_prep.data_settings import *
#from dotenv import find_dotenv, load_dotenv


def transform_choice(choice = "minmax"):
    if choice == "standard":
        # Z transform with mean of zero and variance based on residual
        trans_function = StandardScaler()
    else:
        # range standardization
        trans_function = MinMaxScaler()
    return trans_function

def anomaly_function(method="isolation"):
    if method == "elliptical":
        anomaly_model = EllipticEnvelope(contamination = 0.05)
    else:
        anomaly_model = IsolationForest(contamination = 0.05)
    return anomaly_model

def transform(X_train, X_test, y_train, y_test,logger,outlier_choice = True):

    logger = logging.getLogger(__name__)
    
    # reading the raw dataset
    logger.info('Transforming the split data')
    trans_function = transform_choice(TRANSFORM_METHOD)
    
    # Seperating Numeric and categorical variables
    X_train_num,X_train_cat = X_train[NUMERICAL_VAR],X_train[CATEGORICAL_VAR]
    X_test_num,X_test_cat = X_test[NUMERICAL_VAR],X_test[CATEGORICAL_VAR]
    
    # One hot encoding of categorical variables
    X_train_cat1,X_test_cat1 = pd.get_dummies(X_train_cat.astype('category')),pd.get_dummies(X_test_cat.astype('category'))
    
    # Normalizing the data
    scaled_X_train = trans_function.fit_transform(X_train_num)
    scaled_X_test = trans_function.transform(X_test_num)
    
    # combine the normalized numerical data with one hot encoded categorical data
    X_train_comb = np.concatenate([scaled_X_train,X_train_cat1],axis=1)
    X_test_comb = np.concatenate([scaled_X_test,X_test_cat1],axis=1)
    
    # Choice of dropping anamoly rows from training data
    if outlier_choice:
        anomaly_model = anomaly_function(ANAMOLY_MODEL_TYPE)
        # Using IsolationForest or EllipticEnvelope method to drop the anamoly outlier rows
        yhat = anomaly_model.fit_predict(X_train_comb)
        mask = yhat != -1
        # dropping anamoly rows from the training data
        X_train_comb, y_train = X_train_comb[mask, :], y_train[mask]
    logger.info('Transformation of the data is completed')
    return X_train_comb, y_train,X_test_comb,y_test

    