###########[  indexes  ]############
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

###########[  crawler  ]############
import yfinance as yf
import pandas as pd

def GetCodeIndexes(code, timeperiod = 14):
  temp_df = yf.Ticker(code).history(period='max').sort_index()
  temp_df[code] = temp_df.Close
  temp_df['Mean'] = SMA(temp_df.Close, timeperiod = timeperiod)
  temp_df['Std'] = STDDEV(temp_df.Close, timeperiod = timeperiod)
  temp_df['Ticks'] = (temp_df[code] / 200) / 50 * 50
  temp_df[code + '/Bias'] = (temp_df[code] - temp_df['Mean']) / temp_df['Std']
  temp_df[code + '/Trend'] = temp_df[code].diff(1) / temp_df['Std']
  temp_df[code + '/Pred'] = np.where(-temp_df[code].diff(-1) > temp_df['Ticks'], 1, 0) + np.where(-temp_df[code].diff(-1) < -temp_df['Ticks'], -1, 0)
  return temp_df.loc[:, [code, code + '/Bias', code + '/Trend', code + '/Pred']]

class GlobalMarket():
    def __init__(self):
        self.CODES = {"Forex": ['GC=F', 'USDCNY=X', 'USDJPY=X', 'USDEUR=X', 'USDHKD=X', 'USDKRW=X'],
                      "Debt": ['^TNX'],
                      "Futures": ['GC=F', 'CL=F'],
                      "Crypto": ['BTC-USD', 'ETH-USD'],
                      "Stock": ['^HSI', '^HSCE', '000001.SS', '399001.SZ',    # China
                                '^DJI', '^GSPC', '^IXIC',                     # UnitedStates
                                '^TWII',                                      # Taiwan
                                '^N225',                                      # Japan
                                '^KS11',                                      # Korea
                                ]}

    def summary(self, retry_time = 3):
        print(f'GlobalMarket.summary()')
        self.df = GetCodeIndexes("USDTWD=X").tz_convert('Asia/Taipei')
        self.df.index = self.df.index.date
        for market in self.CODES:
            for code in self.CODES[market]:
                for _ in range(retry_time):
                    self.add(code)
                    break
        return self.df

    def add(self, code):
      if code not in self.df.columns:
        temp_df = GetCodeIndexes(code).tz_convert('Asia/Taipei')
        temp_df.index = temp_df.index.date
        temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
        self.df = pd.concat([self.df, temp_df.loc[self.df.index[0]:, ]], axis=1).sort_index()
        print(' -', code, 'added to dataset.', self.df.shape)
      self.df = self.df.ffill().dropna()

      
###########[  modeling  ]############
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def Split_Dataset(df, window_size = 1, train_ratio = 0.8, delay_period = 5):
    df = df.copy().iloc[:-delay_period, :]

    X, y = np.array(df.loc[:, [i for i in df if 'State' in i or 'Bias' in i]]), np.array(df.loc[:, ['^TWII/Pred']])
    encoder = OneHotEncoder(handle_unknown='ignore')
    y = encoder.fit_transform(y).toarray()
    X = np.array([X[i - window_size: i, :] for i in range(window_size, df.shape[0])]).astype('float32')
    y = np.array([y[i] for i in range(window_size, df.shape[0])]).astype('float32')
    shuffled_indices = np.random.permutation(len(X))

    train_size = int(len(X) * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

class TwoLayerLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm_cell_1 = tf.keras.layers.LSTMCell(hidden_dim)
        self.lstm_cell_2 = tf.keras.layers.LSTMCell(hidden_dim)
        self.dense_layer = tf.keras.layers.Dense(output_dim)

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, patient = 5):
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        history = []
        for epoch in range(epochs):
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                with tf.GradientTape() as tape:
                    predictions = self.predict(X_batch)
                    loss = loss_fn(y_batch, predictions)
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            val_loss, accuracy = self.evaluate(X_test, y_test)
            history.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")
            if min(history) >= val_loss:
              opt_gradients = gradients
            if min(history[-patient:]) > min(history):
              break
        optimizer.apply_gradients(zip(opt_gradients, self.trainable_variables))
        y_pred_classes, y_true_classes = np.argmax(self.predict(X_test), axis=1), np.argmax(y_test, axis=1)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_classes, y_pred_classes))

    def predict(self, X):
        hidden_state_1 = tf.zeros([X.shape[0], self.hidden_dim])
        cell_state_1 = tf.zeros([X.shape[0], self.hidden_dim])
        hidden_state_2 = tf.zeros([X.shape[0], self.hidden_dim])
        cell_state_2 = tf.zeros([X.shape[0], self.hidden_dim])
        for t in range(X.shape[1]):
            output_1, (hidden_state_1, cell_state_1) = self.lstm_cell_1(X[:, t, :], (hidden_state_1, cell_state_1))
            output_2, (hidden_state_2, cell_state_2) = self.lstm_cell_2(output_1, (hidden_state_2, cell_state_2))
        output = self.dense_layer(output_2)
        return tf.nn.softmax(output)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(y_test, predictions)
        accuracy = tf.keras.metrics.CategoricalAccuracy()(y_test, predictions)
        print(f"Test Loss: {loss.numpy()}, Test Accuracy: {accuracy.numpy()}")
        return loss.numpy(), accuracy.numpy()

    @property
    def trainable_variables(self):
        return self.lstm_cell_1.trainable_variables + self.lstm_cell_2.trainable_variables + self.dense_layer.trainable_variables

###########[  optimize  ]############
def action_function(prices):
    actions = []
    for i in range(len(prices)):
      current_price = prices[i]
      future_price = prices[min([i + 5, len(prices) - 1])]
      if future_price > current_price:
        actions.append(2)
      elif future_price < current_price:
        actions.append(0)
      else:
        actions.append(1)
    return np.array(actions)

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV

def Optimizer(short_pred, long_pred, threadhold = 0.7):
  pca = PCA(n_components=6)
  X = pca.fit_transform(np.concatenate((short_pred, long_pred), axis = 1))
  y = action_function(list(df['^TWII'])[1:])

  grid_search = GridSearchCV(DecisionTreeClassifier(), {'max_depth': np.arange(3, 30)}, cv=10, scoring='accuracy') # 5-fold cross-validation, using accuracy
  grid_search.fit(X, y)
  print("Best max_depth:", grid_search.best_params_['max_depth'], "score:", grid_search.best_score_)
  clf = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'])
  clf.fit(X, y)
  tree_rules = export_text(clf, feature_names=[f"Feature_{i}" for i in range(X.shape[1])])
  print(stree_rules)
  return np.array([clf.predict_proba(np.array([x]))[0] for x in X])
