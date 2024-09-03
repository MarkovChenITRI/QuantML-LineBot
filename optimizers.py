from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from sources import Get_Sharpo
from indicators import kelly_criterion
import pandas as pd
import numpy as np

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def Fit_Regressor(df, options, test_size = 0.05):
    print(f'[optimizer.py] fit_regression()')
    split_index = int(df.shape[0] * (1 - test_size))
    X_col, y_col = [i for i in df if 'State' in i or 'Bias' in i], []
    for market in options:
        y_col += options[market]
    input, label = np.array(df.copy().loc[: , X_col]), np.array(df.copy().loc[: , y_col])
    X_train, y_train, X_test, y_test = input[: split_index], label[: split_index], input[split_index: ], label[split_index: ]

    model = LinearRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    pred = model.predict(X_test[-1: ])
    
    print(' - Confidence:', score)
    res_df = pd.DataFrame(pred, columns=y_col).apply(tanh) * score
    res_df.loc[len(res_df.index)] = [Get_Sharpo(i.split('/')[0]) for i in res_df]

    leverage = 22
    EXP, POS = [], []
    for i in res_df:
        code = i.strip('/Pred')
        current_price, expected_price = df[code][-1],  df[code + '/Mean'][-1] + df[code + '/Std'][-1] * res_df[i][0] * 3
        diff = expected_price / current_price - 1
        if diff > 0:
          expected_price = str(int(expected_price)) + '/Call'
        else:
          expected_price = str(int(expected_price)) + '/Put'
        EXP.append(expected_price)
        POS.append(str(round(kelly_criterion(score, abs(diff) * leverage + 1) * 100 / leverage, 2)) + '%')

    res_df.loc[len(res_df.index)] = EXP
    res_df.loc[len(res_df.index)] = POS
    res_df.index = ['Trend', 'Beta', 'Point', 'Position']
    return res_df, score

def Shares_Optimizer(df, market_state, options, score, market):
  print(f'[optimizer.py] Shares_Optimizer()')
  risk_ratio = np.mean(market_state.loc[['Beta'], ['^DJI/Pred', '^GSPC/Pred', '^IXIC/Pred', '^TWII/Pred']]) * score
  market_names = {'NASDAQ': 'UnitedStates', 'TWSE': 'Taiwan', 'ACE': 'Universe', 'NYSE': 'Universe'}
  df['weight'] = [max([0, np.mean(market_state.loc['Trend', options[market_names[i]]])]) for i in df['market']] 
  df['chosed'] = [1 if market_names[i] in market else 0 for i in df['market']]
  
  df["X"] = linprog(c      =  list(df['sharpo'] * df['chosed'] * df['weight'] * -1),
                    A_ub   =  [list(df['beta']),   [1 for _ in range(df.shape[0])]], #coefficient of variables for each constraint
                    b_ub   =  [risk_ratio, 1],    #y value of constraints
                    bounds =  [(0, 1) for _ in range(df.shape[0])], #interval of each variable
                    method =  "highs").x
  df["X"] = df["X"] * df['weight']
  return df