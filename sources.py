import requests
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from indicators import SMA, STDDEV, SHARPE

def GET(code, timeperiod = 90):
  temp_df = yf.Ticker(code).history(period='6y')
  temp_df.index = temp_df.index.to_period(freq='D')

  temp_df[code] = temp_df.Close
  temp_df[code + '/Mean'] = SMA(temp_df.Close, timeperiod = timeperiod)
  temp_df[code + '/Std'] = STDDEV(temp_df.Close, timeperiod = timeperiod)

  temp_df['HCL0'] = np.where(temp_df['Low'] > (temp_df[code + '/Mean'] + 1 * temp_df[code + '/Std']), 1, 0)
  temp_df['HCL1'] = np.where(temp_df['Low'] > (temp_df[code + '/Mean'] + 2 * temp_df[code + '/Std']), 1, 0)
  temp_df['HCL2'] = np.where(temp_df['Low'] > (temp_df[code + '/Mean'] + 3 * temp_df[code + '/Std']), 1, 0)
  temp_df['LCL0'] = np.where(temp_df['High'] < (temp_df[code + '/Mean'] - 1 * temp_df[code + '/Std']), 1, 0)
  temp_df['LCL1'] = np.where(temp_df['High'] < (temp_df[code + '/Mean'] - 2 * temp_df[code + '/Std']), 1, 0)
  temp_df['LCL2'] = np.where(temp_df['High'] < (temp_df[code + '/Mean'] - 3 * temp_df[code + '/Std']), 1, 0)

  temp_df[code + '/State'] = temp_df['HCL0'] + temp_df['HCL1'] + temp_df['HCL2'] - temp_df['LCL0'] - temp_df['LCL1'] - temp_df['LCL2']
  temp_df[code + '/State1'] = temp_df[code + '/State'].shift(1)
  temp_df[code + '/State2'] = temp_df[code + '/State1'].shift(1)
  temp_df[code + '/State3'] = temp_df[code + '/State2'].shift(1)

  temp_df[code + '/Bias'] = (temp_df[code] - temp_df[code + '/Mean'])/ temp_df[code + '/Std']
  temp_df[code + '/Bias1'] = temp_df[code + '/Bias'].diff(1)
  temp_df[code + '/Bias2'] = temp_df[code + '/Bias1'].shift(1)
  temp_df[code + '/Bias3'] = temp_df[code + '/Bias2'].shift(1)
  
  #temp_df[code + '/Pred'] = list(np.clip((temp_df[code + '/Bias'].diff(1) / temp_df[code + '/Std']), -1, 1))[1:] + [None]
  temp_df[code + '/Pred'] = temp_df[code + '/State'].shift(-1)
  res = temp_df.loc[:, [code, code + '/Bias', code + '/Bias1', code + '/Bias2', code + '/Bias3',
                        code + '/State', code + '/State1', code + '/State2', code + '/State3', code + '/Pred']]
  return res

def Get_Price(temp_df, market, USD):
  data = temp_df.history(period='1d')
  update, price = data.index[-1].strftime('%Y-%m-%d'), float(data.Close[-1])
  if market == 'TWSE':
    price /= USD
  return update, price

def Get_EPS(temp_df, market, USD):
  try:
    state_df = temp_df.income_stmt
    eps = state_df.loc['Diluted EPS', :].dropna()[0]
    if market == 'TWSE':
      eps /= USD
  except:
    eps = None
  return eps

def Get_PE(code):
  try:
    pe = temp_df.info['trailingPE']
  except:
    pe = None
  return pe

def Get_Beta(temp_df):
  try:
    beta = temp_df.info['beta']
  except:
    beta = None
  return beta

def Get_Sharpo(code):
  web = requests.get(f'https://portfolioslab.com/symbol/{code}')
  soup = BeautifulSoup(web.text, "html.parser")
  header_element = soup.find(id='sharpe-ratio')
  try:
      res = str(header_element.find_all('b')[1]).replace('<b>', '').replace('</b>', '')
      return float(res)
  except:
      temp_df = yf.download(code, period='6y')['Adj Close'].pct_change().dropna()
      res = temp_df.rolling(240).apply(SHARPE)[-1]
      return res