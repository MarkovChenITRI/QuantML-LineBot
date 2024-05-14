import yfinance as yf
import numpy as np
from indicators import SMA, STDDEV

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
  return res.ffill()