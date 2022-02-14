#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[3]:


data = pd.read_csv('GOOG.csv', date_parser = True)
# data.head()
# data.tail()


# In[4]:


data_train = data[data['Date']<'2020-01-01'].copy()
data_train

data_test = data[data['Date']>='2020-01-01'].copy()
data_test


# In[5]:


training_data = data_train.drop(['Date','Adj Close'], axis = 1)
training_data.head()


# In[6]:


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data


# In[7]:


X_train = []
y_train = []


# In[8]:


training_data.shape[0]


# In[9]:


for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    y_train.append(training_data[i,0])


# In[10]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[11]:


X_train.shape


# In[ ]:


##Building LSTM


# In[12]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[37]:


regressor = Sequential()

regressor.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1],5)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


# In[38]:


regressor.summary()


# In[39]:


regressor.compile(optimizer ='adam', loss = 'mean_squared_error')


# In[40]:


regressor.fit(X_train, y_train, epochs = 10, batch_size= 32 )


# In[41]:


data_test.head()


# In[42]:


past_60_days = data_train.tail(60)
past_60_days


# In[43]:


df = past_60_days.append(data_test, ignore_index = True)
df


# In[44]:


df = df.drop(['Date','Adj Close'], axis = 1)


# In[45]:


inputs = scaler.transform(df)
inputs


# In[46]:


X_test = []
y_test = []


# In[47]:


for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i,0])


# In[48]:


X_test, y_test = np.array(X_test), np.array(y_test)


# In[49]:


X_test.shape
y_test.shape


# In[50]:


y_pred = regressor.predict(X_test)
y_pred


# In[51]:


scaler.scale_


# In[52]:


scale = 1/7.61069658e-04
scale


# In[53]:


y_test = y_test*scale
y_pred = y_pred*scale


# In[54]:


### Visualising the Result


# In[55]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






