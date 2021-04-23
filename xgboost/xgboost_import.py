import numpy as np 
import pandas as pd 
import xgboost as xgb
import argparse
from sklearn.model_selection import GridSearchCV
import pickle 
import time
import os 

def run_xgboost(resp,gpu = True):
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
    # args = {'n_estimators': [100, 200, 400, 500, 600, 700, 800], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.1, 0.05, 0.01], 'n_jobs': [150],'tree_method': ['gpu_hist']}
    # args1 = {'n_estimators': [100, 200, 400, 500, 600, 700, 800], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.1, 0.05, 0.01], 'n_jobs': [150]}
    # args = {'n_estimators': [100], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.01], 'n_jobs': [10],'tree_method': ['gpu_hist']}
    # args1 = {'n_estimators': [100], 'max_depth': range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 'learning_rate':[0.01], 'n_jobs': [10]}
    

    # model

    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,
             importance_type='total_gain', interaction_constraints='',
             learning_rate=0.01, max_delta_step=0, max_depth=3,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=700, n_jobs=150, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='gpu_hist', validate_parameters=1, verbosity=None)

    # gridsearch
    model.fit(X,y)

    # save model
    with open(model_path + resp + '_totalgain.pickle','wb') as f:
        pickle.dump(model,f)
    
    return

if __name__ == '__main__':
    #,'resp_1','resp_2','resp_3','resp_4'
    os.system('export CUDA_VISIBLE_DEVICES=7')
    for resp in ['resp','resp_4']:
        run_xgboost(resp)
    