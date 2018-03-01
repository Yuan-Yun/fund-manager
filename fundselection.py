import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
import warnings
import os
from WindPy import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
trade_days = pd.read_excel('tradeday.xlsx')
trade_days = trade_days['TRADE_DAYS']
tmp = np.unique(np.floor(trade_days[0:2128] / 100))
first_day = list()
for i in range(len(tmp)):
    first_day.append(min(trade_days[np.floor(trade_days / 100) == tmp[i]]))


class Para:
    # month_in_sample = first_day[0:120 + 1]
    # month_test = first_day[12:]
    month_number = len(first_day)
    percent_select = [0.3, 0.3]
    percent_cv = 0.1
    path_data = 'AllData/'
    path_results = 'results/'
    seed = 1
    n_stock_select = 10


para = Para()


def label_data(data):
    data['return_bin'] = np.nan
    data = data.sort_values(by='return', ascending=False)
    n_stock_select = np.multiply(para.percent_select, data.shape[0])
    n_stock_select = np.around(n_stock_select).astype(int)
    data.iloc[0:n_stock_select[0], -1] = 1
    data.iloc[-n_stock_select[1]:, -1] = 0
    data = data.dropna(axis=0)
    return data


strategy = pd.DataFrame({'return': [0] * len(first_day)})
for k in range(12, len(first_day)):
    para.month_in_sample = first_day[0:k]
    para.month_test = first_day[k]
    for i_month in para.month_in_sample:
        file_name1 = para.path_data + str(i_month) + 'X.csv'
        file_name2 = para.path_data + str(i_month) + 'Y.csv'
        data_curr_month_X = pd.read_csv(file_name1, header=0)
        data_curr_month = data_curr_month_X.loc[:, 'abs_ret1':'vol3']
        data_curr_month_Y = pd.read_csv(file_name2, header=0)
        data_curr_month['return'] = data_curr_month_Y['Y']
        para.n_stock = data_curr_month.shape[0]
        data_curr_month = data_curr_month.dropna(axis=0)
        data_curr_month = label_data(data_curr_month)
        if i_month == para.month_in_sample[0]:
            data_in_sample = data_curr_month
        else:
            data_in_sample = data_in_sample.append(data_curr_month)

    x_in_sample = data_in_sample.loc[:, 'abs_ret1':'vol3']
    y_in_sample = data_in_sample.loc[:, 'return_bin']
    x_train, x_cv, y_train, y_cv = train_test_split(x_in_sample, y_in_sample, test_size=para.percent_cv, random_state=para.seed)
    idx1 = x_train[np.sum(np.isinf(x_train), axis=1) > 0].index
    idx2 = x_cv[np.sum(np.isinf(x_cv), axis=1) > 0].index
    x_train = x_train.drop(idx1)
    x_cv = x_cv.drop(idx2)
    y_train = y_train.drop(idx1)
    y_cv = y_cv.drop(idx2)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_cv = scaler.transform(x_cv)

    # model = linear_model.SGDClassifier(loss='hinge', alpha=0.0001, penalty='l2', n_iter=5, random_state=para.seed)
    #
    # model.fit(x_train, y_train)
    # y_pred_train = model.predict(x_train)
    # y_score_train = model.decision_function(x_train)
    # y_pred_cv = model.predict(x_cv)
    # y_score_cv = model.decision_function(x_cv)

    # from sklearn.neural_network import MLPClassifier
    #
    # mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=500)
    # mlp.fit(x_train, y_train)



    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)

    bdt.fit(x_train, y_train)

    i = k - 12
    i_month = para.month_test
    file_name1 = para.path_data + str(i_month) + 'X.csv'
    file_name2 = para.path_data + str(i_month) + 'Y.csv'
    data_curr_month_X = pd.read_csv(file_name1, header=0)
    data_curr_month = data_curr_month_X.loc[:, 'abs_ret1':'vol3']
    data_curr_month_Y = pd.read_csv(file_name2, header=0)
    data_curr_month['return'] = data_curr_month_Y['Y']

    data_curr_month = data_curr_month.dropna(axis=0)
    data_curr_month = data_curr_month.drop(data_curr_month[np.sum(np.isinf(data_curr_month), axis=1) > 0].index)
    x_curr_month = data_curr_month.loc[:, 'abs_ret1':'vol3']
    # scaler = preprocessing.StandardScaler().fit(x_curr_month)
    # x_curr_month = scaler.transform(x_curr_month)

    # SP
    # y_pred_curr_month = model.predict(x_curr_month)
    # y_score_curr_month = pd.DataFrame({'value': model.decision_function(x_curr_month)})
    # y_score_curr_month = y_score_curr_month.sort_values(by='value', ascending=False)

    # NN
    # y_pred_curr_month = mlp.predict(x_curr_month)
    # y_score_curr_month = pd.DataFrame({'value': mlp.decision_function(x_curr_month)})
    # y_score_curr_month = y_score_curr_month.sort_values(by='value', ascending=False)

    # for j in range(len(x_curr_month)):
    #     x_curr_month_j = x_curr_month.iloc[j,:]
    #     y_pred_curr_month[j] = nn.predict(x_curr_month_j)
    # y_pred_curr_month = pd.DataFrame({'value': y_pred_curr_month})
    # y_score_curr_month = y_pred_curr_month.sort_values(by='value', ascending=False)

    # ADA
    y_pred_curr_month = bdt.predict(x_curr_month)
    y_score_curr_month = pd.DataFrame({'value': bdt.decision_function(x_curr_month)})
    y_score_curr_month = y_score_curr_month.sort_values(by='value', ascending=False)

    index_select = y_score_curr_month[0:para.n_stock_select].index
    strategy.loc[i, 'return'] = np.mean(np.exp(data_curr_month.loc[index_select, 'return'])) - 1

    print(i)
strategy['value'] = (strategy['return'] + 1).cumprod()
# w.start()  # 启动 Wind API
# x = w.wsd("000300.SH", "close", str(para.month_in_sample[-1]), str(para.month_test[-1]), "Period=M")
# hs300 = pd.DataFrame(x.Data[0])
# hs300.index = x.Times
df = pd.DataFrame({'strategy': np.array(strategy['value'])})
df.to_csv('ada_resultxxxx.csv')
plt.plot(range(len(strategy.loc[:, 'value'])), strategy.loc[:, 'value'], 'r-')
plt.show()
