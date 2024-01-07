# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep.data_settings import *
#from dotenv import find_dotenv, load_dotenv

def distplot(data,var):
    fig, axes = plt.subplots(nrows = 3, ncols = 3)    # axes is 2d array (3x3)
    axes = axes.flatten()         # Convert axes to 1d array of length 9
    fig.set_size_inches(30, 20)
    for ax, col in zip(axes, var):
        sns.distplot(data[col], ax = ax)
        ax.set_title(col)
    plt.savefig('reports/figures/distribution_plot.jpg')
    
def scatterplot(data,var):
    fig, axes = plt.subplots(nrows = 2, ncols = 2)    # axes is 2d array (3x3)
    axes = axes.flatten()  
    fig.set_size_inches(20,20)
    for ind, (ax, col) in enumerate(zip(axes, var)):
        sns.scatterplot(x=data[col],y=data['SalePrice'],color='red',ax=ax)
        ax.set(xlabel = col,
            ylabel = 'SalePrice')
            #title = f"Scatter Plot of {col} and SalePrice")
    plt.savefig('reports/figures/scatter_plot.jpg')
    
def clean_analysis_main(input_df,logger):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    input_df = input_df.drop(['ID'],axis=1)
    # Checking the data type of the columns, also finding the number of null values
    input_df.info()
    
    # Basic Descriptive Analysis of the raw data
    descriptive_df = input_df.describe().transpose()
    descriptive_df.to_csv("data/analysis_result/basic_desc_analysis.csv")
    
    # Distribution plot and scatter plot
    distplot(input_df,ALL_VAR)
    scatterplot(input_df,NUMERICAL_VAR)
    
    # Correlation between the independent & dependent variables 
    # None of the variables have strong correlation with each other
    print(f"Number of null values {pd.isnull(input_df).sum()}")
    input_df.corr().to_csv("data/analysis_result/correlation_analysis.csv")
    input_df.corr()['SalePrice'].sort_values().to_csv("data/analysis_result/Target_corr_analysis.csv")
    
    logger.info('Data analysis complete')

    return input_df
#data_main("data/raw/sample_dataset_house.csv","data/raw/sample_dataset_house.csv")