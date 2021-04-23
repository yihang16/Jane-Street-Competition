import numpy as np 
import pandas as pd 
import xgboost as xgb
import argparse
from sklearn.model_selection import GridSearchCV
import pickle 
import time
import os

def add_feature(resp):
    """
    resp : ['resp','resp4']
    method:['+','-','/']
    """

    # path initialize 
    data_path = '/home/yihang_toby/data/train.csv'
    model_path = '/home/yihang_toby/model/xgboost/'

    # prepare data

    data = pd.read_csv('/home/yihang_toby/data/all_features.csv')
    feature_col = [i for i in data.columns if 'feature' in i]
    X,y = data[feature_col] , data[resp]

    # parameter
    args = {'n_estimators': [700,1000], 'max_depth': [6,10,15], 'min_child_weight':[1], 'learning_rate':[0.01], 'n_jobs': [150],'tree_method': ['gpu_hist']}

    model = xgb.XGBRegressor()

    # grid search 
    hy_model = GridSearchCV(model, args, cv=5 , verbose = 3)

    # # generate feature
    # with open('/home/yihang_toby/model/xgboost/resp_totalgain.pickle','rb') as f:
    #     model_import = pickle.load(f)
    #     names = ['feature_'+str(i) for i in range(len(model_import.feature_importances_))]
    #     importance = pd.Series(model_import.feature_importances_,index = names)
    #     features = list(importance[importance > 0.01].index)



    # X.to_csv('/home/yihang_toby/data/all_features.csv')
    
    hy_model.fit(X,y)
    print(hy_model.best_estimator_)
    # # save model
    # model = hy_model.best_estimator_
    # model.fit(X,y)
    # feature_importance = pd.Series(model.feature_importances_,index=  model._Booster.feature_names)
    # feature_importance.to_csv('/home/yihang_toby/results/feature_importance.csv')
    with open(model_path + resp +'_feature_importance1.pickle','wb') as f:
        pickle.dump(hy_model,f)

    return 

if __name__ == '__main__':
    model_path = '/home/yihang_toby/model/xgboost/'
    # ,'resp_1','resp_2','resp_3','resp_4'
    os.system('export CUDA_VISIBLE_DEVICES=7')
    for resp in ['resp']:
        add_feature(resp)

    
    # with open('/home/yihang_toby/model/xgboost/resp_totalgain.pickle','rb') as f:
    #     model = pickle.load(f)
    #     names = ['feature_'+str(i) for i in range(len(model.feature_importances_))]
    #     importance = pd.Series(model.feature_importances_,index = names)
    #     print(importance[importance > 0.01])
    #     print(len(importance[importance > 0.01]))
    #     # print(pd.Series(model.feature_importances_,index = names))
    #     # print()
    #     # print(pd.Series(model.feature_importances_,index = names).sort_values())