import numpy as np
import json, io, base64
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def kelly_criterion(win_probability, win_odds):
  return (win_probability * win_odds - (1 - win_probability)) / win_odds

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

def SHARPE(returns, period = 240, adjustment_factor=0):
    returns_risk_adj = returns - adjustment_factor
    return (returns_risk_adj.mean() / returns_risk_adj.std()) * np.sqrt(period)

class Sensitive_Analysis():
  def __init__(self):
    AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
    AURA_USERNAME = "neo4j"
    AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"
    driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))

    labels, columns = {}, []
    for cls in ["field", "topic", "product", "utility"]:
      query = "MATCH (n:{cls}) RETURN n".format(cls=cls)
      with driver.session() as session:
        results = session.run(query).data()
        labels[cls] = [i['n']['name'] for i in results]
        columns += labels[cls]
    columns = np.array(columns)
    columns, list(labels.keys())
    transition_matrix = np.zeros((len(columns), len(columns)))
    with driver.session() as session:
      results = session.run("MATCH p=()-[:dominate]->() RETURN p").data()
      for vertex in results:
        i, j = np.where(columns==vertex['p'][0]['name'])[0][0], np.where(columns==vertex['p'][2]['name'])[0][0]
        transition_matrix[i][j] = 1
    sensitive_table = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)):
      state = [0 for _ in range(len(columns))]
      state[i] = 1
      for j in range(len(columns)):
        if i != j:
          temp_state = state.copy()
          P = transition_matrix.copy()
          P[j] = [0 for _ in range(len(columns))]
          P[j][j] = 1
          for z in range(len(columns)):
            if np.sum(P[z]) > 0:
              P[z] /= np.sum(P[z])
          for _ in range(1000):
            temp_state = np.dot(temp_state, P)
          sensitive_table[i][j] = round(temp_state[j], 2)
    sensitive_df = pd.DataFrame(sensitive_table)
    self.sensitive_df = sensitive_df.set_index(columns)
    self.sensitive_df.columns = columns
    self.labels = labels

  def fit(self, temp_df):
    position = {list(temp_df['Name'])[i]: list(temp_df['Suggest Position'])[i] for i in range(temp_df.shape[0])}
    field_sensitive = self.sensitive_df.loc[list(position.keys()), self.labels['field']]
    topic_sensitive = self.sensitive_df.loc[list(position.keys()), self.labels['topic']]
    product_sensitive = self.sensitive_df.loc[list(position.keys()), self.labels['product']]
    field = {i:np.dot(np.array(list(field_sensitive[i])), np.array(list(position.values()))) for i in field_sensitive}
    field = {i:field[i] / sum(field.values()) for i in field if field[i] > 0.01}
    topic = {i:np.dot(np.array(list(topic_sensitive[i])), np.array(list(position.values()))) for i in topic_sensitive}
    topic = {i:topic[i] / sum(topic.values()) for i in topic if topic[i] > 0.01}
    product = {i:np.dot(np.array(list(product_sensitive[i])), np.array(list(position.values()))) for i in product_sensitive}
    product = {i:product[i] / sum(product.values()) for i in product if product[i] > 0.01}
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(20, 6))
    fig, axs = plt.subplots(1, 3)
    axs[0].pie(list(field.values()), labels=list(field.keys()), autopct='%1.1f%%')
    axs[1].pie(list(topic.values()), labels=list(topic.keys()), autopct='%1.1f%%')
    axs[2].pie(list(product.values()), labels=list(product.keys()), autopct='%1.1f%%')
    axs[0].set_title('Field')
    axs[1].set_title('Market')
    axs[2].set_title('Product')
    f = io.BytesIO()
    plt.savefig(f, format='png', dpi = 100, bbox_inches='tight')
    plt.clf()
    plt.close()

    buffered = io.BytesIO()
    im = Image.open(f)
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str, {'field': field, 'topic': topic, 'product': product}