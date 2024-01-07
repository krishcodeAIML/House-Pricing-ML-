from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,IsolationForest,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
import joblib
import numpy as np
import pandas as pd
from models_train.model_settings import *

def param_function(model_name):
    if model_name == "KNeighborsRegressor":
        param_grid = {"n_neighbors":[8,10,15]}
        model_id = KNeighborsRegressor()
    elif model_name == "RandomForestRegressor":
        param_grid = {"n_estimators":[50,80],'max_depth':[3,4]}
        model_id = RandomForestRegressor()
    elif model_name == "GradientBoostingRegressor":
        param_grid = {"n_estimators":[50,80],'max_depth':[3,4], "learning_rate":[0.01,0.015]}
        model_id = GradientBoostingRegressor()
    elif model_name == "XGBRegressor":
        param_grid = {"n_estimators":[50,80],'max_depth':[3,4], "learning_rate":[0.01,0.015]}
        model_id = XGBRegressor()
    elif model_name == "SVR":
        param_grid = {'C':[1,5,10,100,1000],'gamma':['auto','scale']}
        model_id = SVR()
    return param_grid,model_id


def trainer(X_train, y_train,X_test,  y_test):
    mse = make_scorer(mean_squared_error,greater_is_better=False)
    mae_list = []
    rmse_list = []
    param_list = []
    important_var = []
    
    #looping through all the models and testing
    for ind,model_nameitem in enumerate(MODEL_NAME_LIST):
        
        #Getting the parameter for gridsearchcv and model ID
        param_grid,model_id = param_function(model_nameitem)
        grid = GridSearchCV(model_id,param_grid,scoring=mse,cv=4)
        grid.fit(X_train,y_train)
        
        #model predictions
        predictions = grid.predict(X_test)
        
        #model save
        joblib.dump(grid,f'models/{model_nameitem}.pkl')
        
        #model metrics save for all the models
        rmse_list.append(np.sqrt(mean_squared_error(y_test,predictions)))
        mae_list.append(mean_absolute_error(y_test,predictions))
        
        # Best parameter for the given model
        param_list.append(grid.best_params_)
        try:
            important_var.append(grid.best_estimator_.feature_importances_)
        except:
            important_var.append("")
        
    dict_data = {'model_name': MODEL_NAME_LIST, 'best_param': param_list,'RMSE': rmse_list, "MAE":mae_list} 
    result_df = pd.DataFrame(dict_data)
    return result_df