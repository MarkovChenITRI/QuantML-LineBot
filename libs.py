import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, warnings
import yfinance as yf
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd
from neo4j import GraphDatabase
import scipy.stats as stats
from prompt_toolkit.application.application import E
import yahoo_fin.stock_info as si
import matplotlib as mpl
from matplotlib.font_manager import fontManager
from scipy.optimize import linprog

def numpy_sma(close, timeperiod):
  sma = np.empty_like(close)
  for i in range(len(close)):
    if i < timeperiod:
      sma[i] = np.nan
    else:
      sma[i] = np.mean(close[i-timeperiod+1:i+1])
  return sma

def numpy_stddev(close, timeperiod):
    stddev = np.empty_like(close)
    stddev[:] = np.nan
    for i in range(timeperiod - 1, len(close)):
        window = close[i - timeperiod + 1:i + 1]
        stddev[i] = np.std(window, ddof=0)
    return stddev

def index_summary(index, sensitivity = 90):
  for i in index:
    index[i]['MEAN'] = numpy_sma(index[i].Close, timeperiod=sensitivity)
    index[i]['STD'] = numpy_stddev(index[i].Close, timeperiod=sensitivity)
    index[i]['HCL0'] = np.where(index[i]['Low'] > (index[i]['MEAN'] + 1 * index[i]['STD']), 1, 0)
    index[i]['HCL1'] = np.where(index[i]['Low'] > (index[i]['MEAN'] + 2 * index[i]['STD']), 1, 0)
    index[i]['HCL2'] = np.where(index[i]['Low'] > (index[i]['MEAN'] + 3 * index[i]['STD']), 1, 0)
    index[i]['LCL0'] = np.where(index[i]['High'] < (index[i]['MEAN'] - 1 * index[i]['STD']), 1, 0)
    index[i]['LCL1'] = np.where(index[i]['High'] < (index[i]['MEAN'] - 2 * index[i]['STD']), 1, 0)
    index[i]['LCL2'] = np.where(index[i]['High'] < (index[i]['MEAN'] - 3 * index[i]['STD']), 1, 0)
    index[i]['STATE'] = index[i]['HCL0'] + index[i]['HCL1'] + index[i]['HCL2'] - index[i]['LCL0'] - index[i]['LCL1'] - index[i]['LCL2']
    index[i].index = index[i].index.date
  df = index[list(index.keys())[-1]].copy()
  for i in index:
    df[i] = index[i].STATE
  return df

def marketing_summary():
  market = {}
  for i in ['^HSI', '^HSCE', '000001.SS', '399001.SZ', '^N225', '^DJI', '^GSPC', '^IXIC', '^KS11', '^TWII']:
    market[i] = yf.Ticker(i).history(period='6y')
  market_df = index_summary(market)
  forex = {}
  for i in ['USDTWD=X', 'USDCNY=X', 'USDJPY=X', 'USDEUR=X', 'USDHKD=X', 'USDKRW=X']:
    forex[i] = yf.Ticker(i).history(period='6y')
  forex_df = index_summary(forex)
  summary_df = market_df.iloc[:, 16:].copy()
  for i in forex_df.iloc[:, 16:]:
    summary_df[i] = forex_df[i]
  for i in summary_df:
    summary_df[i] = summary_df[i].ffill()
  return summary_df

def marketing_ratio(summary_df):
  history, indexes = [], []
  for i in range(90, 0, -1):
    temp_df = summary_df.iloc[: -i, :].dropna().copy()
    col = ['^DJI', '^GSPC', '^IXIC', '^TWII']
    X, y, pred_X = np.array(temp_df.iloc[: -1, :]), np.array(temp_df.loc[:, col][1: ]), np.array(temp_df.iloc[-1:, :])
    model = LinearRegression().fit(X, y)
    indexes.append(list(temp_df.index)[-1])
    history.append(model.predict(pred_X)[0])
  summary = pd.DataFrame(np.array(history), columns=col).apply(tanh)
  summary['indexes'] = indexes
  summary = summary.set_index('indexes')
  summary['^TW'] = summary['^TWII']
  summary['^US'] = (summary['^DJI'] + summary['^GSPC'] + summary['^IXIC'])/3
  return summary.loc[:, ["^TW", "^US"]]

def tanh(x):
  x = x
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def forex_risk_test(x, y):
  MEAN, STD = np.mean(x), np.std(x)
  LCL, HCL = (MEAN - STD * 2), (MEAN + STD * 2)
  _, double_p = stats.ttest_ind(np.random.normal(loc=MEAN, scale=STD * MEAN, size=len(TWD.Close)), x * (y / MEAN), equal_var = False)
  if np.mean(x) > np.mean(y):
    p = double_p/2.
  else:
    p = 1.0 - double_p/2.
  return -(HCL / MEAN - 1) * round(p, 2)

def process(option, portfolio):
  labels, columns = {}, []
  for cls in ["field", "topic", "product", "utility"]:
    query = "MATCH (n:{cls}) RETURN n".format(cls=cls)
    with driver.session() as session:
      results = session.run(query).data()
      labels[cls] = [i['n']['name'] for i in results]
      columns += labels[cls]
  transition_matrix, columns = np.zeros((len(columns), len(columns))), np.array(columns)

  level_state = np.zeros((len(columns)))
  pe_pair = []
  for utility in results:
    try:
      if utility['n']['PE_ratio'] != 'nan':
        pe_pair.append([utility['n']['name'], utility['n']['PE_ratio']])
    except:
      pass
  pe_pair = np.array(pe_pair)
  base_count = 0
  for name, pe_ratio in pe_pair:
    _, p = stats.ttest_ind(pe_pair[:,1].astype(float), [float(pe_ratio)])
    #if p > 0.05:
    if 1 == 1:
      i = np.where(columns==str(name))[0][0]
      level_state[i] = pe_ratio; base_count += 1
  level_state = np.array(level_state)/ base_count


  absorbing_node = [np.where(columns==cls)[0][0] for cls in labels[option]]
  #定義吸收節點
  for i in absorbing_node:
    transition_matrix[i][i] = 1

  #定義轉移矩陣
  query = "MATCH p=()-[:dominate]->() RETURN p"
  with driver.session() as session:
    results = session.run(query).data()
    for vertex in results:
      i, j = np.where(columns==vertex['p'][0]['name'])[0][0], np.where(columns==vertex['p'][2]['name'])[0][0]
      if i not in absorbing_node:
        transition_matrix[i][j] = 1
  for i in range(len(columns)):
    if np.sum(transition_matrix[i]) > 0:
      transition_matrix[i] /= np.sum(transition_matrix[i])

  #【投資部位】
  #定義狀態向量
  state = np.zeros((len(columns)))
  for cls in portfolio:
    i = np.where(columns==cls)[0][0]
    state[i] = portfolio[cls]

  #取得穩態機率
  for _ in range(len(columns)):
    state = state.dot(transition_matrix)
  slack = 1
  for i, j in enumerate(absorbing_node):
    slack -= state[absorbing_node][i]
    print(columns[j], ':', round(state[absorbing_node][i] * 100, 2), '%')
  print('Cash :', round(slack * 100, 2), '%')
  #【市場水位】
  for _ in range(len(columns)):
    level_state = level_state.dot(transition_matrix)
  print('\n市場整體本益比:', round(np.sum(level_state * state), 2))

def estimate(df, portfolio):
  beta, sharpe, pe_ratio = 0, 0, 0
  for name in portfolio:
    beta += float(df[df['name']==name]['BETA']) * portfolio[name]
    sharpe += float(df[df['name']==name]['Sharpe']) * portfolio[name]
    pe_ratio += float(df[df['name']==name]['PE_ratio']) * portfolio[name]
  print('-------------------\nBeta:', round(beta, 2), ',Sharpe Ratio:', round(sharpe, 2), ',PE Ratio:', round(pe_ratio, 2))