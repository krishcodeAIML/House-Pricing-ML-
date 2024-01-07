# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split(data,logger):
    """ 
    splitting the data to train and test    
    """
    
    # reading the raw dataset
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=41)
    
    logger.info('Completed data split')
    return X_train, X_test, y_train, y_test
    