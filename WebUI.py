import altair as alt
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, warnings
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
from libs import *
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)

def configuration(Loan_Interest):
  EUR, TWD, KRW = yf.Ticker("EUR=X").history(period='6y'), yf.Ticker("TWD=X").history(period='6y'), yf.Ticker("KRW=X").history(period='6y')
  FOREX_RISK_PER_USD = forex_risk_test(TWD.Close, TWD.Close[-1]) / 2
  EUR_Close = (TWD.Close / EUR.Close).dropna()
  FOREX_RISK_PER_EUR = forex_risk_test(EUR_Close, EUR_Close[-1]) / 2
  KRW_Close = (TWD.Close / KRW.Close).dropna()
  FOREX_RISK_PER_KRW = forex_risk_test(KRW.Close, KRW.Close[-1]) / 2
    
  summary_df = marketing_summary()
  summary = marketing_ratio(summary_df)
  US_index, TW_index = list(summary["^US"])[-1], list(summary["^TW"])[-1]
  return {"台幣本位": 0,"美金本位": FOREX_RISK_PER_USD, "歐元本位": FOREX_RISK_PER_EUR, "韓元本位": FOREX_RISK_PER_KRW}, "美股行情指數: " + str(int(US_index * 100)), "台股行情指數: " + str(int(TW_index * 100))
    
AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"
driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))

global sharpe_dict
history_prices, sharpe_dict = {}, {}
with driver.session() as session:
    results = session.run("MATCH (u:utility) RETURN u").data()
    for utility in results:
      if "sharpo" in utility['u'].keys():
        sharpe_dict[utility['u']['name']], update_time = utility['u']['sharpo'].split('/')
        sharpe_dict[utility['u']['name']] = float(sharpe_dict[utility['u']['name']])
      """
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
      """
print(sharpe_dict, update_time)

def portfolio(*args):
  global sharpe_dict
  new_sharpe_dict = {}
  for i, key in enumerate(list(sharpe_dict.keys())):
    new_sharpe_dict[key] = args[i]
  return "OK"

def on_select(evt: gr.SelectData):
  source = pd.DataFrame({"Target": [12, 23, 47, 6, 52, 19]})
  base = alt.Chart(source).encode(
      theta=alt.Theta("values:Q", stack=True),
      radius=alt.Radius("values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
      color="values:N",
  )
  return base.mark_arc(innerRadius=20, stroke="#fff"), base.mark_text(radiusOffset=10).encode(text="values:Q")

optimizer = gr.Interface(portfolio,
                        inputs = [gr.Slider(0, 5, value=sharpe_dict[i], label=i, info="") for i in sharpe_dict],
                        outputs = [gr.Textbox()],
                        allow_flagging="never",
)

with gr.Blocks() as WebUI:
    gr.Markdown("<span style='font-size:28px; font-weight:bold;'>QuantML </span><span style='font-size:20px; font-weight:bold; color:gray;'>(powered by MarkovChen)</span>")
    with gr.Row():
      forex_label1 = gr.Label(label='世界金融')
      with gr.Column():
        forex_label2, forex_label3 = gr.Label(label=''), gr.Label(label='')
      with gr.Column():
        class_option = gr.Dropdown(["field", "topic", "product"], label="投資部位")
        class_plot = gr.Plot()
        class_option.select(on_select, None, class_plot)
    forex_btn = gr.Button("Update")
    forex_btn.click(configuration, None, [forex_label1, forex_label2, forex_label3])

    gr.Markdown("<span style='font-size:24px; font-weight:bold;'>組合最佳化</span>")
    optimizer.render()
if __name__ == "__main__":
    WebUI.launch()
