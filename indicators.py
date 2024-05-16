import numpy as np

def SMA(close, timeperiod):
  sma = np.empty_like(close)
  for i in range(len(close)):
    if i < timeperiod:
      sma[i] = np.nan
    else:
      sma[i] = np.mean(close[i - timeperiod + 1: i + 1])
  return sma

def STDDEV(close, timeperiod):
    stddev = np.empty_like(close)
    stddev[:] = np.nan
    for i in range(timeperiod - 1, len(close)):
        window = close[i - timeperiod + 1:i + 1]
        stddev[i] = np.std(window, ddof=0)
    return stddev

def sharpe_ratio(returns, period = 240, adjustment_factor=0):
    returns_risk_adj = returns - adjustment_factor
    return (returns_risk_adj.mean() / returns_risk_adj.std()) * np.sqrt(period)