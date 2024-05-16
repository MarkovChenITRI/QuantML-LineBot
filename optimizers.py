from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from sources import Get_Sharpo
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
    input, label = np.array(df.loc[: , X_col]), np.array(df.loc[: , y_col])
    X_train, y_train, X_test, y_test = input[: split_index], label[: split_index], input[split_index: ], label[split_index: ]

    model = LinearRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    pred = model.predict(X_test[-1: ])
    df = pd.DataFrame(pred, columns=y_col).apply(tanh)

    df.loc[len(df.index)] = [Get_Sharpo(i.split('/')[0]) for i in df]
    df.index = ['Trend', 'Beta']
    return df, score

def Distribution_Optimizer(df, market_state, score, risk_ratio, market):
  print(f'[optimizer.py] Distribution_Optimizer()')
  df['chosed'] = [1 if i in market else 0 for i in df['market']]
  df["X"] = linprog(c    = list(df['sharpo'] * df['chosed'] * -1),
                    A_ub  =  [list(df['beta']),   [1 for _ in range(df.shape[0])]], #coefficient of variables for each constraint
                    b_ub  =  [risk_ratio, 1],    #y value of constraints
                    bounds =  [(0, 1) for _ in range(df.shape[0])], #interval of each variable
                    method =  "highs").x
  return df