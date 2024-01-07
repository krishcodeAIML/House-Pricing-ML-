# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from data_prep.dataanalysis_vis import clean_analysis_main
from data_prep.data_split import split
from data_prep.data_transform import transform
#from dotenv import find_dotenv, load_dotenv

def data_main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    # reading the raw dataset
    logger.info('starting to read the raw data')
    raw_df = pd.read_csv(input_filepath)
    logger.info('Read complete, data analysis from the raw data started')
    
    input_df = clean_analysis_main(raw_df,logger)
    logger.info('Analysis complete, data split starts')
    
    X_train, X_test, y_train, y_test = split(input_df,logger)
    logger.info('data split completes, transformation starts')
    
    X_train, y_train,X_test, y_test = transform(X_train, X_test, y_train, y_test,logger, True)
    logger.info('data transformation completes and data preparation for model completes')
    return X_train, y_train,X_test, y_test

    
