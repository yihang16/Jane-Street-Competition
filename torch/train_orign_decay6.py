import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import optuna
from sklearn.model_selection import StratifiedKFold


# GPU_NUM = 8
BATCH_SIZE = 8192# * GPU_NUM
EPOCHS = 200
EARLYSTOP_NUM = 10
NFOLDS = 5

TRAIN = True
CACHE_PATH = './'
torch.multiprocessing.set_sharing_strategy('file_system')


def objective(trial):
    # train_set = MarketDataset(train)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=100)
    # valid_set = MarketDataset(valid)
    # valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=100)
    # lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    # WEIGHT_DECAY = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    # dropout_rate = trial.suggest_int("dropout_rate", 1, 9)
    hidden_size = trial.suggest_int("hidden_size", 256, 400)
    activate = trial.suggest_categorical("non_linear_activation", ["PReLU","LeakyReLU","RReLU"])
    method = trial.suggest_categorical("method", ["mean", "median"])
    lr = 1e-3
    WEIGHT_DECAY = 1e-6
    hidden_size = 256
    dropout_rate = 2
    activate = 'LeakyReLU'
    method = 'median'

    print('lr:',lr)
    print('weight_decay:',WEIGHT_DECAY)
    print('hidden_size:',hidden_size)
    print('dropout_rate:',dropout_rate)
    print('activate:',activate)
    print('method:',method)
    start_time = time.time()
    for _fold, index in enumerate(kf.split(train,train['action'])):
        train_index, valid_index = index
        valid = train.iloc[valid_index]
        train_set = MarketDataset(train.iloc[train_index])
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=100)
        valid_set = MarketDataset(valid)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=100)
        print(f'Fold{_fold}:')
        seed_everything(seed=42+_fold)
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = Model(trial,hidden_size,dropout_rate,activate)
        model.to(device)
        # model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        # optimizer = Nadam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        # optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
        #                                                 max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(train_loader))
        # loss_fn = nn.BCEWithLogitsLoss()
        loss_fn = SmoothBCEwLogits(smoothing=0.005)

        model_weights = f"{CACHE_PATH}/{__file__.split('.')[0]}{_fold}.pth"
        es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)

            valid_pred = inference_fn(model, valid_loader, device)
            # valid_auc = roc_auc_score(valid[target_cols].values, valid_pred)
            valid_logloss = log_loss(valid[target_cols].values, valid_pred)
            valid_pred = eval('np.'+method+'(valid_pred, axis=1)')
            valid_pred = np.where(valid_pred >= .5, 1, 0).astype(int)
            valid_u_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values,
                                                   resp=valid.resp.values, action=valid_pred)
            print(f"FOLD{_fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                      f"valid_u_score={valid_u_score:.5f} valid_logloss={valid_logloss:.5f} "
                      f"time: {(time.time() - start_time) / 60:.2f}min")
            es(valid_u_score, model, model_path=model_weights)
            if es.early_stop:
                print("Early stopping")
                break
        torch.save(model.state_dict(), model_weights)
    if True:
        test_pred = np.zeros((len(test), len(target_cols)))
        test_set = MarketDataset(test)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=100)
        for _fold in range(NFOLDS):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            model = Model(trial,hidden_size,dropout_rate,activate)
            model.to(device)
            model_weights = f"{CACHE_PATH}/{__file__.split('.')[0]}{_fold}.pth"
            model.load_state_dict(torch.load(model_weights))

            test_pred += inference_fn(model, test_loader, device) / NFOLDS
        auc_score = roc_auc_score(test[target_cols].values, test_pred)
        logloss_score = log_loss(test[target_cols].values, test_pred)
        test_pred = eval('np.'+method+'(test_pred, axis=1)')
        test_pred = np.where(test_pred >= .5, 1, 0).astype(int)
        test_score = utility_score_bincount(date=test.date.values, weight=test.weight.values, resp=test.resp.values,
                                             action=test_pred)
        print(f'{NFOLDS} models test score: {test_score}\tauc_score: {auc_score:.4f}\tlogloss_score:{logloss_score:.4f}')
    return logloss_score
    
    
    # return metrics

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
    # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
    # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            # print('validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            # if not DEBUG:
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

train = pd.read_csv('/home/yihang_toby/data/train.csv')

kf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
feat_cols = [i for i in train.columns if 'feature' in i]
# feat_cols = [col for col in train.columns if 'feature' in col]

if TRAIN:
    train = train.loc[train.date > 85].reset_index(drop=True)

    train['action'] = (train['resp'] > 0).astype('int')
    train['action_1'] = (train['resp_1'] > 0).astype('int')
    train['action_2'] = (train['resp_2'] > 0).astype('int')
    train['action_3'] = (train['resp_3'] > 0).astype('int')
    train['action_4'] = (train['resp_4'] > 0).astype('int')
    test = train.loc[(train.date >= 450) & (train.date < 500)].reset_index(drop=True)
target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']

if TRAIN:
    df = pd.concat([train[feat_cols], test[feat_cols]]).reset_index(drop=True)
    f_mean = df.mean()
    f_mean = f_mean.values
    np.save(f'{CACHE_PATH}/f_mean_online.npy', f_mean)

    train.fillna(df.mean(), inplace=True)
    test.fillna(df.mean(), inplace=True)
else:
    f_mean = np.load(f'{CACHE_PATH}/f_mean_online.npy')

##### Making features
# https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference/data
# eda:https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance
# his example:https://www.kaggle.com/gracewan/plot-model
def fillna_npwhere_njit(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array

class RunningEWMean:
    def __init__(self, WIN_SIZE=20, n_size=1, lt_mean=None):
        if lt_mean is not None:
            self.s = lt_mean
        else:
            self.s = np.zeros(n_size)
        self.past_value = np.zeros(n_size)
        self.alpha = 2 / (WIN_SIZE + 1)

    def clear(self):
        self.s = 0

    def push(self, x):

        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s

    def get_mean(self):
        return self.s
all_feat_cols = [col for col in feat_cols]
# if TRAIN:
#     all_feat_cols = [col for col in feat_cols]

#     train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
#     train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
#     valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
#     valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

#     all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

##### Model&Data fnc
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class MarketDataset:
    def __init__(self, df):
        self.features = df[all_feat_cols].values

        self.label = df[target_cols].values.reshape(-1, len(target_cols))

        self.weight = df['weight'].values.reshape(-1, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
            'weight':torch.tensor(self.weight[idx],dtype = torch.float)
        }


class Model(nn.Module):
    def __init__(self,trail,hidden_size,dropout_rate,activate):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate/10)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate/10)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate/10)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate/10)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

        self.activate = activate
        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        exec('x1 = self.'+self.activate+'(x1)')
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        exec('x2 = self.'+self.activate+'(x2)')
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        exec('x3 = self.'+self.activate+'(x3)')
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        exec('x4 = self.'+self.activate+'(x4)')
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        weight = data['weight'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1, len(target_cols))

    return preds

def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    # print('weight: ', weight)
    # print('resp: ', resp)
    # print('action: ', action)
    # print('weight * resp * action: ', weight * resp * action)
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(252 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    
    print("Results:", study.best_params)
    print(study.best_value)
    print(study.best_trial)
