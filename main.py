import gradio as gr
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
from libs import *

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)

AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"
driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))
LOAN_INTEREST_PER_TWD = 0.07
EUR, TWD, KRW = yf.Ticker("EUR=X").history(period='6y'), yf.Ticker("TWD=X").history(period='6y'), yf.Ticker("KRW=X").history(period='6y')
FOREX_RISK_PER_USD = forex_risk_test(TWD.Close, TWD.Close[-1]) / 2
EUR_Close = (TWD.Close / EUR.Close).dropna()
FOREX_RISK_PER_EUR = forex_risk_test(EUR_Close, EUR_Close[-1]) / 2
KRW_Close = (TWD.Close / KRW.Close).dropna()
FOREX_RISK_PER_KRW = forex_risk_test(KRW.Close, KRW.Close[-1]) / 2
query = "MATCH (u:utility) RETURN u"
history_prices = {}

with driver.session() as session:
    results = session.run(query).data()
    for utility in results:
      print(utility['u']['name'], utility['u']['code'])
      df = yf.Ticker(utility['u']['code'])
      data = df.history(period='1y')
      if data.shape[0] > 0:
        data.index = data.index.date
        history_prices[utility['u']['name']] = data.loc[:, ['Close']]
        update, price = data.index[-1].strftime('%Y-%m-%d'), data.Close[-1]
        if update != utility['u']['update']:
          quote = si.get_quote_table(utility['u']['code'])
          try:
            eps = df.income_stmt.loc['Diluted EPS', :].dropna()[0]
          except:
            eps = 0
          beta, pe_ratio = quote['Beta (5Y Monthly)'], quote['PE Ratio (TTM)']
          if utility['u']['code'].endswith('.TW'): #台幣計價
            price /= TWD.Close[-1]; eps /= TWD.Close[-1]
          elif utility['u']['code'].endswith('.TI'): #歐元計價
            price /= EUR.Close[-1]; eps /= EUR.Close[-1]
            beta = beta - FOREX_RISK_PER_EUR
          elif utility['u']['code'].endswith('.KS'): #韓元計價
            price /= KRW.Close[-1]; eps /= KRW.Close[-1]
            beta = beta - FOREX_RISK_PER_KRW
          else: #美金計價
            beta = beta - FOREX_RISK_PER_USD
          price, eps = round(price, 2), round(eps, 2)
          query = "MATCH (u:utility WHERE u.name='{name}') set u.update='{update}' set u.price='{price}' set u.eps='{eps}' set u.beta='{beta}' set u.pe_ratio='{pe_ratio}'".format(name=utility['u']['name'], update=update, price=str(price), eps=eps, beta=beta, pe_ratio=pe_ratio)
          session.run(query); print(query)

def my_func(name):
    update_time = '2024-01-17' # @param {type:"date"}
    Micron = 0.9     # @param {type:"number"}
    Samsung = 0.33    # @param {type:"number"}
    TSMC = 1.25      # @param {type:"number"}
    ASML = 1.49      # @param {type:"number"}
    ARM = 0.7       # @param {type:"number"}
    NVIDIA = 4.69     # @param {type:"number"}
    _2454 = 0.8      # @param {type:"number"}
    AMD = 1.42      # @param {type:"number"}
    Intel = 0.8      # @param {type:"number"}
    _2308 = 0.7      # @param {type:"number"}
    _2356 = 0.7      # @param {type:"number"}
    _2317 = 0.7      # @param {type:"number"}
    _2382 = 0.7      # @param {type:"number"}
    _6669 = 0.7      # @param {type:"number"}
    _2376 = 0.7      # @param {type:"number"}
    TXN = 1        # @param {type:"number"}
    Qualcomm = 1.1    # @param {type:"number"}
    _2412 = 0       # @param {type:"number"}
    Tesla = 2.68     # @param {type:"number"}
    Amazon = 2.54     # @param {type:"number"}
    Microsoft = 1.54   # @param {type:"number"}
    Meta = 4.9      # @param {type:"number"}
    Apple = 1.9      # @param {type:"number"}
    NXP = 1.2      # @param {type:"number"}

    update_time = '20240117'
    sharpe_dict = {"Micron": Micron, "Samsung": Samsung, "TSMC": TSMC, "ASML": ASML, "ARM": ARM, "NVIDIA": NVIDIA,
            "MediaTek": _2454, "AMD": AMD, "Intel":Intel, "台達電": _2308, "英業達": _2356,
            "鴻海": _2317, "廣達": _2382, "緯穎": _6669, "技嘉": _2376, "德州儀器": TXN,
            "Qualcomm": Qualcomm, "中華電": _2412, "Tesla": Tesla, "Amazon": Amazon,
            "Microsoft": Microsoft, "Meta": Meta, "Apple": Apple, "NXP": NXP}
    company_names = list(sharpe_dict.keys())
    option = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0 , 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    sharpe_last = [0.64, 0.71, 1.02, 1.25, 0.87, 1.18, 0.79, 0.96, 0.58, 0.69, 0.52, 0.44, 0.48, 0.46, 0.51, 0.76, 0.83, 0.41, 0.62, 0.91, 0.94, 0.88, 0.86, 0.74]
    Sharpe = list(sharpe_dict.values())

    print(Sharpe)
    plt.bar(np.arange(len(company_names)), sharpe_last, width=0.35, label="last sharpe")
    plt.bar(np.arange(len(company_names)) + 0.35, Sharpe, width=0.35, label="latest sharpe")
    plt.xlabel("Company"); plt.ylabel("Sharpe")
    plt.xticks(np.arange(len(company_names)) + 0.35 / 2, company_names, rotation=90)
    plt.legend(); plt.grid(); plt.show()
    summary_df = marketing_summary()
    summary = marketing_ratio(summary_df)

    Sharpe_temp = Sharpe.copy()
    table = {'name':[], 'price': [], 'eps': [], 'BETA': [], 'PE_ratio': []}
    query = "MATCH (u:utility) RETURN u"
    with driver.session() as session:
    results = session.run(query).data()
    for utility in results:
        if 'eps' in utility['u'].keys():
        i = np.where(np.array(company_names)==str(utility['u']['name']))[0][0]
        if utility['u']['code'].endswith('.TW'): #台幣計價
            Sharpe_temp[i] -= LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_USD)
        elif utility['u']['code'].endswith('.TI'): #歐元計價
            Sharpe_temp[i] -= LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_EUR)
        elif utility['u']['code'].endswith('.KS'): #韓元計價
            Sharpe_temp[i] -= LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_KRW)
        else: #美金計價
            Sharpe_temp[i] -= LOAN_INTEREST_PER_TWD
        table['name'].append(utility['u']['name']); table['price'].append(utility['u']['price']); table['eps'].append(utility['u']['eps'])
        table['BETA'].append(utility['u']['beta']); table['PE_ratio'].append(utility['u']['pe_ratio'])

    table['name'].append('USD=X'); table['price'].append(TWD.Close[-1]); table['eps'].append(0); table['BETA'].append(-FOREX_RISK_PER_USD); table['PE_ratio'].append(1); Sharpe_temp.append(1 - LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_USD))
    table['name'].append('EUR=X'); table['price'].append(EUR.Close[-1]); table['eps'].append(0); table['BETA'].append(-FOREX_RISK_PER_EUR); table['PE_ratio'].append(1); Sharpe_temp.append(1 - LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_EUR))
    table['name'].append('KRW=X'); table['price'].append(KRW.Close[-1]); table['eps'].append(0); table['BETA'].append(-FOREX_RISK_PER_KRW); table['PE_ratio'].append(1); Sharpe_temp.append(1 - LOAN_INTEREST_PER_TWD * (1 - FOREX_RISK_PER_KRW))
    table['name'].append('TWD=X'); table['price'].append(1); table['eps'].append(0); table['BETA'].append(0); table['PE_ratio'].append(1); Sharpe_temp.append(1 - LOAN_INTEREST_PER_TWD)

    df = pd.DataFrame(table)
    df['eps'], df['price'] = df['eps'].astype(float), df['price'].astype(float)
    df['投資報酬率(%)/年'] = (df['eps'].astype(float) * 100 / df['price']).round(2)
    df['Sharpe'] = Sharpe_temp
    df['option'] = option + [1, 1, 1, 1]

    df['PE_ratio'] = df['PE_ratio'].replace('nan', np.mean(df['PE_ratio'][9:23].astype(float).dropna())).astype(float)
    df['BETA'] = df['BETA'].replace('nan', np.mean(df['BETA'].astype(float).dropna())).astype(float)

    df["X"] = linprog(c=list(df['Sharpe'] * -1), A_ub=[list(df['BETA']), [1 for _ in range(df.shape[0])]], b_ub=[1.05, 1], bounds=[(0, 1) for _ in range(df.shape[0])], method="highs").x
    df["X'"] = linprog(c=list(df['Sharpe']), A_ub=[list(df['option']), list(df['BETA'] * -1), [1 for _ in range(df.shape[0])]], b_ub=[0, -1.05, 1], bounds=[(0, 1) for _ in range(df.shape[0])], method="highs").x
    df["X'"] = df["X'"] * -1
    df.to_excel(update_time + '.xlsx')

    asset = 800000
    portfolio_pos, portfolio_neg = {}, {}
    for i, name in enumerate(df["name"][:-4]):
    option, X, X_plus = df['option'][i],df['X'][i], df["X'"][i]
    if round(X, 1) != 0:
        if option == 1:
        portfolio_pos[name] = round(summary["^US"][-1] * X, 2)
        else:
        portfolio_pos[name] = round(summary["^TW"][-1] * X, 2)
    if round(X_plus, 1) != 0:
        if option == 0 and summary["^TW"][-1] < 0:
        portfolio_neg[name] = round(summary["^TW"][-1] * X_plus, 2)
    portfolio_pos, portfolio_neg

    print('美股建議投資比例', summary["^US"][-1] * 100, '%')
    print('台股建議投資比例', summary["^TW"][-1] * 100, '%')
    portfolio_pos, portfolio_neg

    opt = 'field' #@param ["field", "topic", "product"]
    process(opt, portfolio_pos)
    estimate(df, portfolio_pos)
    return "Hello"

demo = gr.Interface(fn=my_func, inputs="text", outputs="text")
    
demo.launch(server_name="0.0.0.0")