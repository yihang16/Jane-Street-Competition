import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import argparse
import sys

parser = argparse.ArgumentParser(description='JS Market Prediction')
parser.add_argument('--weights', type=int, nargs='+', help='input 5 weights, from 1 to 3')
args = parser.parse_args()


file_dir = '/home/yihang_toby/'

train = pd.read_csv(file_dir+'data/train.csv')
'''
train = pd.read_csv(file_dir+'data/train.csv', skiprows = lambda x: x>0 and np.random.rand() > 0.01)
train.to_csv('train_small.csv')
sys.exit()
'''
train = train.query('date > 85').reset_index(drop = True) 
train = train[train['weight'] != 0]

train.fillna(train.mean(),inplace=True)
#train['action'] = ((train['resp'].values) > 0).astype(int)
features = [c for c in train.columns if "feature" in c]
f_mean = np.mean(train[features[1:]].values,axis=0)
resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']
X_train = train.loc[:, train.columns.str.contains('feature')]
#y_train = (train.loc[:, 'action'])
'''
y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
y_train = train.loc[:,'resp']
y_train = np.where(y_train>0,1,0)
'''
yt = np.array(train.loc[:, resp_cols])
# for i in range(1,4):
#     args.weights[-2]=i
for j in range(1,8):
    args.weights[-1]=j
    y_train = yt * (np.array(args.weights)/np.sum(args.weights))
    print(X_train.shape, y_train.shape)
    y_train = np.sum(y_train, axis=1)
    print(args.weights)
#model = XGBRegressor(tree_method='gpu_hist')
    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,importance_type='gain', interaction_constraints='',learning_rate=0.01, max_delta_step=0, max_depth=3,min_child_weight=1, monotone_constraints='()',n_estimators=700, n_jobs=150, num_parallel_tree=1, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,tree_method='gpu_hist', validate_parameters=1, verbosity=None)
    model.fit(X_train, y_train, sample_weight = train['weight'])
    signature = ''
    for weight in args.weights:
        signature += str(weight)
    model.save_model('/home/yihang_toby/model/'+signature+'_all.model')
    y_pred = model.predict(X_train)
    o = open('/home/yihang_toby/results/'+signature+'_all.txt','w')
    o.write(str(np.var(y_pred)))


'''
pd.DataFrame(X_train.iloc[1:2].loc[:, features].values,columns = X_train.columns)
pd.Series(X_train.iloc[1:2].loc[:, features].values,names  = X_train.columns)
pd.DataFrame(X_train.iloc[1])
'''
#model.predict(pd.DataFrame(X_train.iloc[1:2].loc[:, features].values,columns = X_train.columns)).item()


'''
import janestreet
from tqdm import tqdm
env = janestreet.make_env()
for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        pred = model.predict(pd.DataFrame(x_tt,columns = X_train.columns)).item()
        print(pred)
        pred_df.action = int(pred)
    else:
        pred_df.action = 0
    env.predict(pred_df)
'''


