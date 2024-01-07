import os 
import logging
from data_prep.data_main import data_main
from models_train.train_model import trainer
def main_train_pipeline():
    logger = logging.getLogger(__name__)
    #invoking the main training pipeline
    X_train, y_train,X_test,  y_test = data_main("data/raw/sample_dataset_house.csv")
    results = trainer(X_train, y_train,X_test,  y_test)
    results.to_csv("results/result.csv")
main_train_pipeline()