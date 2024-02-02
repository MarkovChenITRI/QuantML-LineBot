import altair as alt
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, warnings, random, time
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from neo4j import GraphDatabase
import scipy.stats as stats
from prompt_toolkit.application.application import E
import yahoo_fin.stock_info as si
import matplotlib as mpl
from matplotlib.font_manager import fontManager
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from libs import *
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)

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
    
AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"
driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))

global sharpe_dict, portfolio
history_prices, sharpe_dict, portfolio = {}, {}, {'category': [], 'value': []}
with driver.session() as session:
    results = session.run("MATCH (u:utility) RETURN u").data()
    for utility in results:
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
        time.sleep(5)
        bot_message = """
        ASML是一家全球領先的半導體設備製造商，專門提供光刻機和相關服務。ASML的財務資訊包括年報、季報、財務策略、股票回購等，您可以在ASML的投資者關係網站查看詳細內容。以下是一些ASML的財務概況：\n\n

        1. 2023年，ASML的營收達到1,416億歐元，同比增長28.9%；淨利潤為34.6億歐元，同比增長35.7%；每股收益為8.49歐元，同比增長35.2%。\n
        2. 2023年第四季度，ASML的營收為37.8億歐元，同比增長22.3%；淨利潤為10.3億歐元，同比增長50.8%；每股收益為2.52歐元，同比增長50.6%。\n
        3. ASML的財務策略是通過股息和股票回購向股東回報現金，並保持一定的財務靈活性。2023年，ASML宣布了一項60億歐元的股票回購計劃，預計在2024年底前完成。此外，ASML還提議將2023年的股息提高15%，達到2.75歐元每股。\n\n
        希望這些資訊對您有幫助，如果您還有其他問題，歡迎隨時與我聊天。👋
        """
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
        forex_label1 = gr.Label(label='世界金融', height=200)
        forex_plot = gr.Plot(label='資金流向')
      with gr.Column():
        forex_label2, forex_label3, forex_label4 = gr.Label(label=''), gr.Label(label=''), gr.Label(label='')
      with gr.Column():
        class_option = gr.Dropdown(["field", "topic", "product"], label="投資部位")
        class_plot = gr.Plot(label='', label_font_size=16)
        class_option.select(portfolio_select, None, class_plot)
    forex_btn = gr.Button("Update")
    forex_btn.click(configuration, None, [forex_plot, forex_label1, forex_label2, forex_label3, forex_label4])

    chatbot = gr.Chatbot(label='AGR Chatbot (powered by OpenAI)', default= [('1', 'I love you')])
    msg = gr.Textbox(placeholder="Ex. 請幫我搜尋有關ASML的財務資訊")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    gr.Markdown("<span style='font-size:24px; font-weight:bold;'>投資組合最佳化</span>")
    optimizer.render()

if __name__ == "__main__":
    WebUI.launch()
