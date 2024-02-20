# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, warnings, time
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from neo4j import GraphDatabase
import scipy.stats as stats
from prompt_toolkit.application.application import E
import yahoo_fin.stock_info as si
import matplotlib as mpl
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
import gradio as gr
import altair as alt

#################【initialize】####################
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)
AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"
driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))

query = "MATCH (u:utility) RETURN u"
global sharpe_dict, portfolio
history_prices, sharpe_dict, portfolio = {}, {}, {'category': [], 'value': []}
with driver.session() as session:
    results = session.run(query).data()
    for utility in results:
      print(utility['u'])
      if "sharpo" in utility['u'].keys():
        sharpe_dict[utility['u']['name']], update_time = utility['u']['sharpo'].split('/')
        sharpe_dict[utility['u']['name']] = float(sharpe_dict[utility['u']['name']])
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
print(sharpe_dict)
#################【libs】####################
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
  _, double_p = stats.ttest_ind(np.random.normal(loc=MEAN, scale=STD * MEAN, size=len(x)), x * (y / MEAN), equal_var = False)
  if np.mean(x) > np.mean(y):
    p = double_p/2.
  else:
    p = 1.0 - double_p/2.
  return -(HCL / MEAN - 1) * round(p, 2)

def process(option, distribute, driver):
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
  for i in absorbing_node:
    transition_matrix[i][i] = 1
  query = "MATCH p=()-[:dominate]->() RETURN p"
  with driver.session() as session:
    results = session.run(query).data()
    for vertex in results:
      i, j = np.where(columns==vertex['p'][0]['name'])[0][0], np.where(columns==vertex['p'][2]['name'])[0][0]
      if i not in absorbing_node:
        transition_matrix[i][j] = 1
      if option != "field":
        if vertex['p'][2]['name'] in labels["field"]:
          transition_matrix[j][i] = 1
  for i in range(len(columns)):
    if np.sum(transition_matrix[i]) > 0:
      transition_matrix[i] /= np.sum(transition_matrix[i])
  state = np.zeros((len(columns)))
  cash = 1
  for cls in distribute:
    i = np.where(columns==cls)[0][0]
    state[i] = distribute[cls]
    cash -= state[i]

  res = {'category': [], 'value': []}
  for _ in range(len(columns)):
    state = state.dot(transition_matrix)

  slack = 1 - cash
  for i, j in enumerate(absorbing_node):
    slack -= state[absorbing_node][i]
    print(columns[j], ':', round(state[absorbing_node][i] * 100, 2), '%')
    res['category'].append(columns[j])
    res['value'].append(state[absorbing_node][i] * 100)
  print('Other :', round(slack * 100, 2), '%')

  res['category'].append("其它"); res['value'].append(slack * 100)
  res['category'].append("現金"); res['value'].append(cash * 100)
  for _ in range(len(columns)):
    level_state = level_state.dot(transition_matrix)
  print('\n市場整體本益比:', round(np.sum(level_state * state), 2))
  return res

def estimate(df, portfolio):
  beta, sharpe, pe_ratio = 0, 0, 0
  for name in portfolio:
    beta += float(df[df['name']==name]['BETA']) * portfolio[name]
    sharpe += float(df[df['name']==name]['Sharpe']) * portfolio[name]
    pe_ratio += float(df[df['name']==name]['PE_ratio']) * portfolio[name]
  return 'Beta係數:' + str(round(beta, 2)) +', Sharpe指標:' + str(round(sharpe, 2)) +', 整體本益狀況:' + str(round(pe_ratio, 2))

#################【WebUI】####################

def configuration(Loan_Interest):
  global FOREX_RISK_PER_USD, FOREX_RISK_PER_EUR, FOREX_RISK_PER_KRW, LOAN_INTEREST_PER_TWD, EUR, TWD, KRW, summary
  EUR, TWD, KRW = yf.Ticker("EUR=X").history(period='6y'), yf.Ticker("TWD=X").history(period='6y'), yf.Ticker("KRW=X").history(period='6y')
  FOREX_RISK_PER_USD = forex_risk_test(TWD.Close, TWD.Close[-1]) / 2
  EUR_Close = (TWD.Close / EUR.Close).dropna()
  FOREX_RISK_PER_EUR = forex_risk_test(EUR_Close, EUR_Close[-1]) / 2
  KRW_Close = (TWD.Close / KRW.Close).dropna()
  FOREX_RISK_PER_KRW = forex_risk_test(KRW.Close, KRW.Close[-1]) / 2
  LOAN_INTEREST_PER_TWD = 0.07
  summary = marketing_ratio(marketing_summary())
  US_index, TW_index = list(summary["^US"])[-1], list(summary["^TW"])[-1]
  TWD_std, EUR_std, KRW_std = TWD.Close[-90:] / np.std(TWD.Close), EUR.Close[-90:] / np.std(EUR.Close), KRW.Close[-90:] / np.std(KRW.Close)
  data = pd.DataFrame({'x': list(range(90)), 'TWD': TWD_std / np.mean(TWD_std), 'EUR': EUR_std / np.mean(EUR_std), 'KRW': KRW_std / np.mean(KRW_std)})
  data_melted = data.melt('x', var_name='line', value_name='y')
  chart = alt.Chart(data_melted).mark_line().encode(x='x', y=alt.Y('y', scale=alt.Scale(domain=[0.9, 1.1])), color='line').properties(height=100)
  return chart, {"台幣本位": 0,"美金本位": FOREX_RISK_PER_USD, "歐元本位": FOREX_RISK_PER_EUR, "韓元本位": FOREX_RISK_PER_KRW}, "美股行情指數: " + str(int(US_index * 100)), "台股行情指數: " + str(int(TW_index * 100)), "實質利率: " + str(int(LOAN_INTEREST_PER_TWD * 100)) + "%"

def summary_finance_report(Sharpe):
  Sharpe_temp = list(Sharpe.values())
  company_names = ["Micron", "Samsung", "TSMC", "ASML", "ARM", "NVIDIA", "MediaTek", "AMD", "Intel", "台達電",
                   "英業達", "鴻海", "廣達", "緯穎", "技嘉", "德州儀器", "Qualcomm", "中華電", "Tesla", "Amazon",
                   "Microsoft", "Meta", "Apple", "NXP"]
  option = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0 , 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
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
  return df

def portfolio(*args):
  global sharpe_dict, portfolio_pos
  sharpo_data, new_sharpe_dict = {'company': [], 'sharpo': [], 'scope': []}, {}
  for i, key in enumerate(list(sharpe_dict.keys())):
    new_sharpe_dict[key] = args[i]
    sharpo_data['company'] += [key + '_1', key + '_2']
    sharpo_data['sharpo'] += [args[i], sharpe_dict[key]]
    sharpo_data['scope'] += ['old', 'new']
  sharpo_plot = alt.Chart(pd.DataFrame(sharpo_data)).mark_bar().encode(x='company:O', y='sharpo:Q', color='scope:N').properties(width=600, height=240)
  
  df = summary_finance_report(new_sharpe_dict)
  df_show = df.iloc[:-4, ].loc[:, ["name", "BETA", "PE_ratio", "投資報酬率(%)/年"]].round(1)

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
  return sharpo_plot, df_show, portfolio_pos, portfolio_neg, estimate(df, portfolio_pos)

def portfolio_select(evt: gr.SelectData):
  return alt.Chart(pd.DataFrame(process(evt.value, portfolio_pos, driver))).mark_arc().encode(color='category',theta='value').properties(height=300, width=200)

def respond(message, chat_history):
        bot_message = """系統暫時關閉, 以免用量爆炸₍ᐢ ̥ ̮ ̥ᐢ₎ """
        chat_history.append((message, bot_message))
        return "", chat_history

optimizer = gr.Interface(portfolio,
                        inputs = [gr.Slider(0, 5, value=sharpe_dict[i], label=i, info="") for i in sharpe_dict],
                        outputs = [gr.Plot(), gr.Dataframe(type="pandas"), gr.Label(label='多方投資比例'), gr.Label(label='空方投資比例'), gr.Textbox(label="推薦摘要")],
                        allow_flagging="never",
)

with gr.Blocks() as WebUI:
    gr.Markdown("<span style='font-size:28px; font-weight:bold;'>QuantML </span><span style='font-size:20px; font-weight:bold; color:gray;'>(MarkovChen)</span>")
    with gr.Row():
      with gr.Column():
        forex_label1 = gr.Label(label='世界金融')
        forex_plot = gr.Plot(label='資金流向')
      with gr.Column():
        forex_label2, forex_label3, forex_label4 = gr.Label(label=''), gr.Label(label=''), gr.Label(label='')
      with gr.Column():
        class_option = gr.Dropdown(["field", "topic", "product"], label="投資部位")
        class_plot = gr.Plot(label='')
        class_option.select(portfolio_select, None, class_plot)
    forex_btn = gr.Button("Update")
    forex_btn.click(configuration, None, [forex_plot, forex_label1, forex_label2, forex_label3, forex_label4])

    chatbot = gr.Chatbot(label='AGR Chatbot (powered by OpenAI)')
    msg = gr.Textbox(value="I am looking for the most credible investment rating agencies in the world with financial market expertise and I expect to get some risk assessment indicators from some of their reports or statements, which is very important to me as I need to support my young children and elders at home who are unable to walk. Here are the investments I am looking at:  please help me to investigate different professional organizations' evaluation of Sharp's performance according to their order. Please follow his order to help me survey different professional organizations on their evaluation of Sharpe Indicator (as long as the Sharpe Indicator), and consolidate them into a table according to the evaluation date of this indicator. yahoo and google are not professional enough, please survey more credible organizations (like S&P Global, Moody’s, Morgan Stanley, Google Finance, Barclays, Morningstar, Zacks, BofA Securities, Goldman Sachs, Credit Suisse, PortfoliosLab, PortfoliosViewer) which publish this more approach 2024/02/15 and newer than 2024/02/05 give me the source of the information, so as to help me to take better care of my family. of course, please share me what date does these sharpo ratio be publushed in the table.")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    gr.Markdown("<span style='font-size:24px; font-weight:bold;'>投資組合最佳化</span>")
    optimizer.render()

WebUI.launch()
