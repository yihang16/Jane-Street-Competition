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

    data = pd.read_csv(data_path)
    ispass = True
    while (ispass):
        X,y = data.loc[:,'feature_0':'feature_129'] , data[resp]

        # parameter
        args = {'n_estimators': [700], 'max_depth': [6,10,15], 'min_child_weight':[1], 'learning_rate':[0.01], 'n_jobs': [150],'tree_method': ['gpu_hist']}

        model = xgb.XGBRegressor()

        # grid search 
        hy_model = GridSearchCV(model, args, cv=5 , verbose = 3)

        # generate feature
        with open('/home/yihang_toby/model/xgboost/resp_totalgain.pickle','rb') as f:
            model_import = pickle.load(f)
            names = ['feature_'+str(i) for i in range(len(model_import.feature_importances_))]
            importance = pd.Series(model_import.feature_importances_,index = names)
            features = list(importance[importance > 0.01].index)


        
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2:
                    X[feature1 + '+' +feature2] = X[feature1] + X[feature2]
        
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2:
                    X[feature1 + '-' +feature2] = X[feature1] - X[feature2]

        
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2:
                    X[feature1 + '*' + 'abs('+feature2+')'] = X[feature1] * abs(X[feature2])
        
        
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2:
                    X[feature1 + '/' + 'abs('+feature2+')'] = X[feature1] / abs(X[feature2] + 1e-5)

        all_features =  pd.read_csv('/home/yihang_toby/results/feature_importance.csv',index_col=0)
        all_features['name'] = all_features.index
        all_features.columns = ['importance','name']
        multi = ['*' in i for i in all_features['name']]
        div= ['/' in i for i in all_features['name']]
        plus = ['+' in i for i in all_features['name']]
        minus = ['-' in i for i in all_features['name']]
        print('enter threshold')
        thre = float(input())
        X = X[all_features[all_features['importance'] > thre].index]

        all_features['method'] = np.where(multi,'*',np.where(div,'/',np.where(plus,'+',np.where(minus,'-','normal'))))
        plus_features = all_features[all_features['importance'] > thre][all_features['method'] == '+'].index
        minus_features = all_features[all_features['importance'] > thre][all_features['method'] == '-'].index
        normal_features = all_features[all_features['importance'] > thre][all_features['method'] == 'normal'].index

        for feature1 in plus_features:
            for feature2 in normal_features:
                X[feature1 + '/' + 'abs('+feature2+')'] = X[feature1] / abs(X[feature2] + 1e-5)
                X[feature1 + '*' + 'abs('+feature2+')'] = X[feature1] * abs(X[feature2])
        print(len(X.columns))
        print('enter ispass')
        ispass =int(input())
    for column in data.columns:
        X[column] = data[column]
    X.to_csv('/home/yihang_toby/data/all_features.csv')
    
    # hy_model.fit(X,y)
    # # save model
    # model = hy_model.best_estimator_
    # model.fit(X,y)
    # feature_importance = pd.Series(model.feature_importances_,index=  model._Booster.feature_names)
    # feature_importance.to_csv('/home/yihang_toby/results/feature_importance.csv')
    # # with open(model_path + resp +'_feature_importance.pickle','wb') as f:
    # #     pickle.dump(hy_model,f)

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