"""
This program is used to to store and retrieve data.

Authors: Yuan Yun
Date:    2017/09/22
"""

import numpy as np
import pandas as pd
from WindPy import *
from datetime import datetime as dt
import warnings
import os
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm

w.start()  # 启动 Wind API
warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
funddata = pd.read_excel('fund.xlsx')  # 这部分是筛选后的基金，均为主动型、开放型、非分级基金
trade_days = pd.read_excel('tradeday.xlsx')
trade_days = trade_days['TRADE_DAYS']
if not os.path.exists('NewData'):
    os.mkdir('NewData')
if not os.path.exists('AllData'):
    os.mkdir('AllData')


def selection(day, fund):
    """
    筛选在指定日期满足FOF投资条件的偏股型基金
    :param day: 指定日期
    :param fund: 待筛选基金的代码及相关数据
    :return: number-基金数量，code-基金代码, operating_period-基金运营时间
    """
    daydate = dt.strptime(str(day), '%Y%m%d')
    operating_period = np.array([(daydate - dt.strptime(str(i), '%Y%m%d')).days for i in fund['setdate']])
    number = sum(operating_period > 365)
    code = fund['fundcode']
    code = np.array(code[operating_period > 365])
    operating_period = operating_period[operating_period > 365]
    # 最近的两个定期报告日
    if int(str(day)[4:]) < 400:
        startdate = str(int(str(day)[0:4]) - 1) + '0930'
        enddate = str(int(str(day)[0:4]) - 1) + '1231'
    elif int(str(day)[4:]) < 700:
        startdate = str(int(str(day)[0:4]) - 1) + '1231'
        enddate = str(day)[0:4] + '0331'
    elif int(str(day)[4:]) < 1000:
        startdate = str(day)[0:4] + '0331'
        enddate = str(day)[0:4] + '0630'
    else:
        startdate = str(day)[0:4] + '0630'
        enddate = str(day)[0:4] + '0930'
    for i in range(number):
        x = w.wss(code[i], "fund_stm_issuingdate_qty,prt_netasset", "rptDate=" + enddate + ";unit=1")
        if x.Data[1][0] > 0:
            break
    if x.Data[0][0] < daydate:
        x1 = w.wss(",".join(code), "fund_stm_issuingdate_qty,prt_netasset,prt_stocktonav", "rptDate=" + startdate + ";unit=1")
        df1 = pd.DataFrame(x1.Data).T
        df1 = pd.DataFrame(np.column_stack((pd.DataFrame(x1.Codes), df1)))
        df = df1
    else:
        x2 = w.wss(",".join(code), "fund_stm_issuingdate_qty,prt_netasset,prt_stocktonav", "rptDate=" + startdate + ";unit=1")
        df2 = pd.DataFrame(x2.Data).T
        df2 = pd.DataFrame(np.column_stack((pd.DataFrame(x2.Codes), df2)))
        df = df2
    number = sum((df.iloc[:, 2] > 1.e+08) & (df.iloc[:, 3] > 80))
    code = code[(df.iloc[:, 2] > 1.e+08) & (df.iloc[:, 3] > 80)]
    operating_period = operating_period[(df.iloc[:, 2] > 1.e+08) & (df.iloc[:, 3] > 80)]
    asset = df[(df.iloc[:, 2] > 1.e+08) & (df.iloc[:, 3] > 80)]
    asset = pd.Series(asset.iloc[:, 2])
    asset.index = code
    return number, code, operating_period, asset


def net(day, code):
    """
    指定日期前三个月的指定基金的净值数据
    :param day: 指定日期
    :param code: 指定基金代码
    :return: data-基金累计单位净值数据
    """
    enddate = str(day)[0:4] + '-' + str(day)[4:6] + '-' + str(day)[6:]
    x = w.wsd(",".join(code), "NAV_acc", "ED-3M", enddate, "")
    data = pd.DataFrame(x.Data).T
    data.columns = code
    data.index = x.Times
    return data


def available(day, code, data):
    data = data[[list(data.columns).index(i) for i in code]]
    if day not in [int(i.strftime('%Y%m%d')) for i in list(data.index)]:
        print('ATTENTION: %d is not in the data list' % day)
        return
    if data is None:
        print('ATTENTION: data is none')
        return
    if not isinstance(data, pd.DataFrame):
        print('ATTENTION: type of data is ' + str(type(data)) + ', please input data as DataFrame')
        return
    if (list(data.index)[-1] - list(data.index)[0]).days < 80:
        print('ATTENTION: time interval is %d days, too short' % (list(data.index)[-1] - list(data.index)[0]).days)
        return


def date_index(day, data):
    day1m = dt.strptime(str(day), '%Y%m%d') - relativedelta(months=+1)
    temp = np.array([int(i.strftime('%Y%m%d')) for i in list(data.index)])
    temp = temp[temp >= int(day1m.strftime('%Y%m%d'))][0]
    day1m_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(temp)
    day3m = dt.strptime(str(day), '%Y%m%d') - relativedelta(months=+3)
    temp = np.array([int(i.strftime('%Y%m%d')) for i in list(data.index)])
    temp = temp[temp >= int(day3m.strftime('%Y%m%d'))][0]
    day3m_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(temp)
    day_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(day) + 1
    return day1m_index, day3m_index, day_index


def abs_ret(day, code, data):
    """
    收益指标：绝对收益
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: abs_ret1-过去一个月的净值收益, abs_ret3-过去三个月的净值收益
    """
    available(day, code, data)
    data = data[[list(data.columns).index(i) for i in code]]
    day1m_index, day3m_index, day_index = date_index(day, data)
    abs_ret1 = data.iloc[day_index - 1, :] / data.iloc[day1m_index, :] - 1.
    abs_ret2 = data.iloc[day_index - 1, :] / data.iloc[day3m_index, :] - 1.
    return abs_ret1, abs_ret2


def risk(day, code, data, bench):
    """
    风险指标：波动率、最大回撤、亏损频率、下行风险
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: vol1-过去一个月的波动率, vol3-过去三个月的波动率, mdd1-过去一个月的最大回撤, mdd3-过去三个月的最大回撤, loss1-过去一个月的亏损频率, loss3-过去三个月的亏损频率, down1-过去一个月的下行风险, down3-过去三个月的下行风险
    """
    available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    vol1 = np.std(r.iloc[day1m_index:day_index, :])
    vol3 = np.std(r.iloc[day3m_index:day_index, :])
    loss1 = np.sum(r.iloc[day1m_index:day_index, :] < 0, axis=0) / r.iloc[day1m_index:day_index, :].shape[0]
    loss3 = np.sum(r.iloc[day3m_index:day_index, :] < 0, axis=0) / r.iloc[day3m_index:day_index, :].shape[0]
    rm = np.array(bench.iloc[1:, :]) / np.array(bench.iloc[0:-1, :]) - 1.
    rp = r.iloc[day1m_index:day_index, :] - np.tile(rm[day1m_index:day_index], r.shape[1])
    down1 = np.zeros(r.shape[1])
    for i in range(r.shape[1]):
        tmpr = rp.iloc[:, i]
        tmpr = tmpr[tmpr < 0]
        down1[i] = np.power(sum(np.power(tmpr, 2)) / (len(tmpr) - 1), 1 / 2)
    down1 = pd.Series(down1)
    down1.index = vol1.index
    rp = r.iloc[day3m_index:day_index, :] - np.tile(rm[day3m_index:day_index], r.shape[1])
    down3 = np.zeros(r.shape[1])
    for i in range(r.shape[1]):
        tmpr = rp.iloc[:, i]
        tmpr = tmpr[tmpr < 0]
        down3[i] = np.power(sum(np.power(tmpr, 2)) / (len(tmpr) - 1), 1 / 2)
    down3 = pd.Series(down3)
    down3.index = vol1.index
    day1m_index, day3m_index, day_index = date_index(day, data)
    values = data.iloc[day1m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    mdd1 = abs(np.min(dd, axis=0))
    values = data.iloc[day3m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    mdd3 = abs(np.min(dd, axis=0))
    return vol1, vol3, mdd1, mdd3, loss1, loss3, down1, down3


def risk_ret(day, code, data):
    """
    风险调整后收益指标：夏普比率、卡玛比率
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: sharp1-过去一个月的夏普比率, sharp1-过去三个月的夏普比率, calmar1-过去一个月的卡玛比率, calmar3-过去三个月的卡玛比率
    """
    available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    sharp1 = np.mean(r.iloc[day1m_index:day_index, :]) / np.std(r.iloc[day1m_index:day_index, :])
    sharp3 = np.mean(r.iloc[day3m_index:day_index, :]) / np.std(r.iloc[day3m_index:day_index, :])
    day1m_index, day3m_index, day_index = date_index(day, data)
    values = data.iloc[day1m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    calmar1 = (values.iloc[-1] / values.iloc[0] - 1) / values.shape[0] * 244 / abs(np.min(dd, axis=0))
    values = data.iloc[day3m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    calmar3 = (values.iloc[-1] / values.iloc[0] - 1) / values.shape[0] * 244 / abs(np.min(dd, axis=0))
    return sharp1, sharp3, calmar1, calmar3


def tm_model(day, code, data, bench):
    """
    T-M模型指标：选股能力、择时能力、系统风险系数
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :param bench: 比较基准
    :return: stock-选股能力, timing-择时能力, beta-系统风险系数
    """
    available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    rf = 4 / 100 / 365
    rm = (np.array(bench.iloc[1:, :]) / np.array(bench.iloc[0:-1, :]) - 1.)[day3m_index:day_index]
    rp = r.iloc[day3m_index:day_index, :]
    stock = pd.Series(rp.shape[1])
    timing = pd.Series(rp.shape[1])
    beta = pd.Series(rp.shape[1])
    for i in range(rp.shape[1]):
        x = np.column_stack((rm - rf, np.power(rm - rf, 2)))
        x = sm.add_constant(x)
        y = np.array(rp.iloc[:, i]) - rf
        est = sm.OLS(y, x)
        est = est.fit()
        stock[i], beta[i], timing[i] = est.params
    stock.index = data.columns
    timing.index = data.columns
    beta.index = data.columns
    return stock, timing, beta


def manager(day, code):
    """
    基金经理指标：任职时间
    :param day: 截面日期
    :param code: 指定基金代码
    :return: mng-基金在指定日期的任职基金经理
    """
    x = w.wss(",".join(code), "fund_predfundmanager")
    x1 = pd.Series(x.Data[0])
    x1.index = x.Codes
    mng = pd.Series(len(code))
    mng_time = pd.Series(len(code))
    for i in range(len(x1)):
        for k in range(len(x1[i].split('\r\n'))):
            j = x1[i].split('\r\n')[k]
            if j[-9:-1].isdigit() & j[-18:-10].isdigit():
                if (int(j[-9:-1]) >= day) & (int(j[-18:-10]) <= day):
                    mng[i] = j[0:-19]
                    mng_time[i] = (dt.strptime(str(day), '%Y%m%d') - dt.strptime(j[-18:-10], '%Y%m%d')).days
            elif (j[-3:-1] == '至今') & (int(j[-11:-3]) <= day):
                mng[i] = j[0:-12]
                mng_time[i] = (dt.strptime(str(day), '%Y%m%d') - dt.strptime(j[-11:-3], '%Y%m%d')).days
    mng.index = x.Codes
    mng_time.index = x.Codes
    return mng, mng_time


tmp = np.unique(np.floor(trade_days[0:2128] / 100))
first_day = list()
for i in range(len(tmp)):
    first_day.append(min(trade_days[np.floor(trade_days / 100) == tmp[i]]))
last_day = list()
for i in range(len(tmp)):
    last_day.append(max(trade_days[np.floor(trade_days / 100) == tmp[i]]))

for d in reversed(np.array(first_day)):
    print(d)
    if not os.path.exists('NewData/' + str(d)):
        os.mkdir('NewData/' + str(d))
    number, code, operating_period, asset = selection(d, funddata)
    if number <= 0:
        print('ERROR: no data!')
        break
    df_basic = pd.DataFrame({'code': code, 'operating_period': operating_period, 'asset': asset})
    df_basic.to_csv('NewData/' + str(d) + '/basic.csv', index=False)
    data = net(d, code)
    data.to_csv('NewData/' + str(d) + '/netvalue.csv')
    abs_ret1, abs_ret2 = abs_ret(d, code, data)
    df_ret = pd.DataFrame({'abs_ret1': abs_ret1, 'abs_ret2': abs_ret2})
    df_ret.to_csv('NewData/' + str(d) + '/absret.csv')
    # 同期沪深300净值
    x = w.wsd("000300.SH", "close", "ED-3M", str(d), "")
    hs300 = pd.DataFrame(x.Data[0])
    hs300.index = x.Times
    if hs300.shape[0] != data.shape[0]:
        print('ERROR: length not pair!')
        break
    hs300.to_csv('NewData/' + str(d) + '/hs300net.csv')
    vol1, vol3, mdd1, mdd3, loss1, loss3, down1, down3 = risk(d, code, data, hs300)
    df_risk = pd.DataFrame({'vol1': vol1, 'vol3': vol3, 'mdd1': mdd1, 'mdd3': mdd3, 'loss1': loss1, 'loss3': loss3, 'down1': down1, 'down3': down3})
    df_risk.to_csv('NewData/' + str(d) + '/risk.csv')
    sharp1, sharp3, calmar1, calmar3 = risk_ret(d, code, data)
    df_riskret = pd.DataFrame({'sharp1': sharp1, 'sharp3': sharp3, 'calmar1': calmar1, 'calmar3': calmar3})
    df_riskret.to_csv('NewData/' + str(d) + '/riskret.csv')
    stock, timing, beta = tm_model(d, code, data, hs300)
    df_tm = pd.DataFrame({'stock': stock, 'timing': timing, 'beta': beta})
    df_tm.to_csv('NewData/' + str(d) + '/tmmodel.csv')
    mng, mng_time = manager(d, code)
    df_mng = pd.DataFrame({'manager': mng, 'manage_time': mng_time})
    df_mng.to_csv('NewData/' + str(d) + '/manager.csv')
    all_x = pd.DataFrame(
        {'operating_period': operating_period, 'asset': asset, 'abs_ret1': abs_ret1, 'abs_ret2': abs_ret2, 'vol1': vol1, 'vol3': vol3, 'mdd1': mdd1,
         'mdd3': mdd3, 'loss1': loss1, 'loss3': loss3, 'down1': down1, 'down3': down3, 'sharp1': sharp1, 'sharp3': sharp3, 'calmar1': calmar1,
         'calmar3': calmar3, 'stock': stock, 'timing': timing, 'beta': beta, 'manage_time': mng_time})
    all_x.to_csv('AllData/' + str(d) + 'X.csv')

for i in range(len(first_day)):
    fd = first_day[i]
    ld = last_day[i]
    x = pd.read_csv('AllData/' + str(fd) + 'X.csv')
    code = x['Unnamed: 0']
    yfd = w.wsd(",".join(code), "NAV_acc", str(fd), str(fd), "")
    yld = w.wsd(",".join(code), "NAV_acc", str(ld), str(ld), "")
    logret = pd.DataFrame({'code':code,'Y':np.log(np.array(yld.Data[0])/np.array(yfd.Data[0]))})
    logret.to_csv('AllData/' + str(fd) + 'Y.csv')