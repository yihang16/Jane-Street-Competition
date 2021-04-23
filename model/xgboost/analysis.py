l
import pickle
import pandas as pd
with open('/home/yihang_toby/model/xgboost/resp_feature_importance1.pickle','rb') as f:
    model_import = pickle.load(f)
    model_import = model_import._best_estimators
    names = ['feature_'+str(i) for i in range(len(model_import.feature_importances_))]
    importance = pd.Series(model_import.feature_importances_,index = names)
    features = list(importance[importance > 0.01].index)
    print(model_import.feature_names)

# feature
# data = pd.read_csv(data_path)
# for resp in ['resp','resp_4']:
#     for method in ['+','-','*','$']:
#         with open(resp + method +'_feature_importance.pickle','rb') as f:
#             # data = pd.read_csv(data_path)
#             X,y = data.loc[:,'feature_0':'feature_129'] , data[resp]
            
#             model = pickle.load(f) 
#             model.best_estimator_.feature_importances_  
#     model = pickle.load(f)
# with open('resp$_feature_importance.pickle','rb') as f:
#     model = pickle.load(f)
#     print(model.best_estimator_.feature_importances_[model.best_estimator_.feature_importances_ > 0.01])

# # with open('/home/yihang_toby/model/xgboost/resp_totalgain.pickle','rb') as f:
# #         model_import = pickle.load(f)
# #         names = ['feature_'+str(i) for i in range(len(model_import.feature_importances_))]
# #         importance = pd.Series(model_import.feature_importances_,index = names)
# #         features = list(importance[importance > 0.01].index)

# # for method in ['+','*','$','/']:
    