import numpy as np 
import pandas as pd 
import xgboost as xgb
import argparse
from sklearn.model_selection import GridSearchCV
import pickle 
import time
import os 

def grid_search(resp,gpu = True):
    """
    resp : ['resp','resp1','resp2','resp3','resp4']
    """

    # path initialize
    data_path = '/home/yihang_toby/data/train.csv'
    model_path = '/home/yihang_toby/model/xgboost/'

    # prepare data
    data = pd.read_csv(data_path)
    X,y = data.loc[:,'feature_0':'feature_129'] , data[resp]

    # parameters
    args = {'n_estimators': [100, 200, 400, 500, 600, 700, 800], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.1, 0.05, 0.01], 'n_jobs': [150],'tree_method': ['gpu_hist']}
    args1 = {'n_estimators': [100, 200, 400, 500, 600, 700, 800], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.1, 0.05, 0.01], 'n_jobs': [150]}
    # args = {'n_estimators': [100], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.01], 'n_jobs': [10],'tree_method': ['gpu_hist']}
    # args1 = {'n_estimators': [100], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.01], 'n_jobs': [10]}
    

    # model

    model = xgb.XGBRegressor()

    # gridsearch
    if gpu:
        hy_model = GridSearchCV(model, args, cv=5 , verbose = 3)
    else:
        hy_model = GridSearchCV(model, args1, cv=5 , verbose = 3)
    hy_model.fit(X, y)

    # save model
    with open(model_path + resp + '_gridsearch.pickle','wb') as f:
        pickle.dump(hy_model,f)
    
    return

if __name__ == '__main__':
    #,'resp_1','resp_2','resp_3','resp_4'
    os.system('export CUDA_VISIBLE_DEVICES=7')
    for resp in ['resp']:
        grid_search(resp)
    