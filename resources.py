import requests, smtplib
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from indicators import SMA, STDDEV, SHARPE
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from email.mime.text import MIMEText

def SendEmail(html):
  whitelist = ["markov.chen1996@gmail.com", "Kepitlo@gmail.com", "lzy871119@gmail.com", "sces60206@gmail.com", "sylviia.chan@gmail.com"]

  mime=MIMEText(html, "html", "utf-8")
  mime["Subject"]="本日投資組合及分配比例 (系統自動發送)"
  mime["From"]="QuantML System"
  mime["To"]="markov.chen1996@gmail.com"
  msg=mime.as_string()
  smtp=smtplib.SMTP("smtp.gmail.com", 587)
  smtp.ehlo()
  smtp.starttls()
  smtp.login("markov.chen1996@gmail.com", 'zwkd lwtp vkqs uafa')
  from_addr="markov.chen1996@gmail.com"
  to_addr=whitelist
  status=smtp.sendmail(from_addr, to_addr, msg)
  smtp.quit()
  print(status)
  print(html)

def SendMessage(text='Meow!!!'):
  line_bot_api = LineBotApi('Es+feMvp7Uwg+nIcgB66iAKWVD1dOKRcXzYwPmSbko+b0Vf21iko3s7dRwEFX1tfToR8mrW78XUACEd/uyecCF/Uqd9LgvkchpPEPiODdX4L8BU4b6pXHzFvlDoAfsP9xIFSMG+rmVzQURS+7uBnegdB04t89/1O/w1cDnyilFU=')
  line_bot_api.push_message('Udba3ff0abbe6607af5a5cfc2e2ddc8a1', TextSendMessage(text=text))

def Push(img_str, dist, dist_buffer, html):
  dist_match = False
  for i, j in dist_buffer:
    if i == dist:
      dist_match, image_url = True, j
  if dist_match == False:
    try:
      image_url = UploadImage(img_str)
      dist_buffer.append((dist, image_url))
      html += f'<img src="{image_url}">'
    except:
      html += f'<br>(Rate limiting)<br>'
  else:
    html += f'<img src="{image_url}">'
  print(dist_buffer)
  return html, dist_buffer
  
def UploadImage(img_str):
  response = requests.post(
    "https://api.imgur.com/3/image",
    headers={
      "Authorization": "869c052fc7ddbfe 94de4c79d7c297349dc2ef16aadb283e5ea17b41",
    },
    data={
      "image": img_str,
    },
  )
  data = response.json()
  print(data)
  return data["data"]["link"]

def GetCodeIndexes(code, timeperiod = 90):
  temp_df = yf.Ticker(code).history(period='max')
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
  temp_df[code + '/State2'] = temp_df[code + '/State1'].shift(2)
  temp_df[code + '/State3'] = temp_df[code + '/State2'].shift(3)
  temp_df[code + '/Bias'] = (temp_df[code] - temp_df[code + '/Mean'])/ temp_df[code + '/Std']
  temp_df[code + '/Bias1'] = temp_df[code + '/Bias'].diff(1)
  temp_df[code + '/Bias2'] = temp_df[code + '/Bias1'].shift(2)
  temp_df[code + '/Bias3'] = temp_df[code + '/Bias2'].shift(3)

  temp_df[code + '/Pred'] = temp_df[code + '/State'].shift(-1)
  res = temp_df.loc[:, [code, code + '/Bias', code + '/Bias1', code + '/Bias2', code + '/Bias3',
                        code + '/State', code + '/State1', code + '/State2', code + '/State3', 
                        code + '/Mean', code + '/Std', code + '/Pred']]
  return res

def GetPrice(temp_df, market, USD):
  data = temp_df.history(period='1d')
  update, price = data.index[-1].strftime('%Y-%m-%d'), float(data.Close[-1])
  if market == 'TWSE':
    price /= USD
  return update, price

def GetEPS(temp_df, market, USD):
  try:
    state_df = temp_df.income_stmt
    eps = state_df.loc['Diluted EPS', :].dropna()[0]
    if market == 'TWSE':
      eps /= USD
  except:
    eps = None
  return eps

def GetPE(temp_df):
  try:
    pe = temp_df.info['trailingPE']
  except:
    pe = None
  return pe

def GetBeta(temp_df):
  try:
    beta = temp_df.info['beta']
  except:
    beta = None
  return beta

def GetSharpo(code):
  web = requests.get(f'https://portfolioslab.com/symbol/{code}')
  soup = BeautifulSoup(web.text, "html.parser")
  header_element = soup.find(id='sharpe-ratio')
  try:
      res = str(header_element.find_all('b')[1]).replace('<b>', '').replace('</b>', '')
      return float(res)
  except:
      temp_df = yf.download(code, period='max')['Adj Close'].pct_change().dropna()
      res = temp_df.rolling(240).apply(SHARPE)[-1]
      return res