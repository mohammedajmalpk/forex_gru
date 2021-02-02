#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from datetime import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt


# In[2]:


df = pdr.DataReader("USD/INR", "av-forex-daily", start=datetime(2014, 11, 7),end=datetime(2021, 2, 1),api_key='ALPHAVANTAGE_API_KEY')


# In[3]:


df.tail(5)


# In[4]:


df.index = pd.to_datetime(df.index)


# In[5]:


df1 = df["close"]

#plotting dataset to visualize the pattern of prices over the years

df1.plot(kind='line',figsize=(15,7))


# In[6]:


df1.describe()


# In[7]:


df1.head()


# In[8]:


print(len(df1))


# In[9]:


date_splt = pd.Timestamp('2019-03-20')
train = df1.loc[:date_splt]
test = df1.loc[date_splt:]
print(test)


ax = train.plot(kind='line',figsize=(15,8))
ax = test.plot(kind='line',figsize=(15,8))
plt.legend(['train', 'test'])


# In[10]:


train = np.array(train).reshape(-1,1)
test = np.array(test).reshape(-1,1)


# In[11]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()


# In[12]:


train_std = stdsc.fit_transform(train.reshape(-1, 1))
test_std = stdsc.transform(test.reshape(-1,1))


# In[13]:


X_train = train_std[:-1]
y_train = train_std[1:]

X_test = test_std[:-1]
y_test = test_std[1:]


# In[14]:


print(y_train.shape)
print(X_train.shape)


# In[15]:


from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf 
import sklearn.metrics as metrics


# In[16]:


X_train_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_train_t = y_train.reshape(-1,1) 


# In[17]:


model = Sequential()
model.add(GRU(7, input_shape=(1, X_train.shape[1]), activation='linear', kernel_initializer='lecun_uniform', return_sequences=False))
model.add(Dense(1))
model.summary()


# In[18]:


model.compile(loss= tf.keras.metrics.mean_squared_error,
             metrics = [tf.keras.metrics.RootMeanSquaredError(name='rmse')],
             optimizer= 'adam')
history = model.fit(X_train_t, y_train_t, epochs=10, batch_size=1)


# In[19]:


plt.plot(np.arange(0,10), history.history['loss'], label="loss")


# In[20]:


def adjusted_r2(R2, n, p):
    ar2 = 1-(1-R2)*(n-1)/(n-p-1)
    return ar2


# In[21]:


## Train_data

y_pred = model.predict(X_test_t)
y_train_pred = model.predict(X_train_t)

train_mse = metrics.mean_squared_error(y_train,y_train_pred)
root_mean_squared_error = np.sqrt(train_mse)

r2_train = r2_score(y_train, y_train_pred)
mean_absolute_error_train = metrics.mean_absolute_error(y_train, y_train_pred)
Adjusted_r2_score = adjusted_r2(r2_train, X_train.shape[0], X_train.shape[1])

print(Adjusted_r2_score)


# In[22]:


## Test_data

test_mse = metrics.mean_squared_error(y_test,y_pred)
root_mse_test = np.sqrt(test_mse)

r2_test = r2_score(y_test, y_pred)
mean_absolute_error_test = metrics.mean_absolute_error(y_test, y_pred)
Adjusted_r2_score_test = adjusted_r2(r2_test, X_test.shape[0], X_test.shape[1])

print(mean_absolute_error_test)


# In[23]:


plt.figure(figsize=(12,8))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='GRU')
plt.title("GRU's_Prediction")
plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
plt.show()


# In[24]:


inv_ytest = stdsc.inverse_transform(y_test)
inv_yhat = stdsc.inverse_transform(y_pred)


# In[25]:


plt.plot(inv_ytest)
plt.plot(inv_yhat)


# In[26]:


mse_test = metrics.mean_squared_error(inv_ytest, inv_yhat)
print(mse_test)

rmse = np.sqrt(mse_test)
print(rmse)

r2_test_inv = r2_score(inv_ytest, inv_yhat)
print(r2_test_inv) # r2 score


# In[27]:


df1.head()


# In[33]:


col1 = pd.DataFrame(inv_ytest , columns=['close'])
col2 = pd.DataFrame(inv_yhat, columns=['Predition'])
#col3 = pd.DataFrame(history.history['rmse'], columns=['RMSE'])

Final = pd.concat([col1,col2], axis=1)


# In[38]:


Final.tail()


# In[39]:


test_data = pd.DataFrame(df1.loc[date_splt:])
test_data.head()


# In[40]:


test_data.reset_index(level=0, inplace=True)
test_data.head()


# In[42]:


output = test_data.merge(Final,how='inner', left_on='close', right_on='close')
output.head()


# In[44]:


output.drop_duplicates(subset='index', inplace=True)
output.tail()


# In[46]:


plt.figure(figsize=(12,8))
price_date = output['index']
price_true = output['close']
price_pred = output['Predition']
plt.plot_date(price_date, price_true,label='Actual Price')
plt.plot_date(price_date, price_pred,label='Predicted Price')
plt.legend()
plt.show()


# In[48]:


output.describe()

