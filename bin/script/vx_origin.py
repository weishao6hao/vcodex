#coding:utf-8
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
Author: vcodex@author
Date: 2020/08/07 20:23:23
"""
import tushare as ts
import pandas as pd


class StockData(object):
    def __init__(self):
        self.pro = ts.pro_api('32d57ff4cf328a6b7a2f06d638f882763130c06dfa3168a886f10f37')

    # 这里可以自定义开始到结束的时间，还有股票代码，使用的时候设置
    def get_data(self,code, start='19900101', end='20200722'):
        stock_code = self.tran_code(code)
        return self.pro.query('daily', ts_code=stock_code, start_date=start, end_date=end)

    def tran_code(self,code):
        if code[0:1] == '6':
            return code + '.SH'
        else:
            return code + '.SZ'
stock = StockData()
data_train = stock.get_data("600519", start = '20000501',end = '20191201')
data_test = stock.get_data("600519", start = '20200101',end = '20200301')
print(data_test.head())


# In[93]:


data_train.shape


# In[18]:


# data_train.shape
data_train.head(10)


# In[28]:


cols = ['open','high','low','close']
# data_train.iloc[:3]


# In[146]:


# n = data_train.shape[0]
X_train = []
y_train = []
for i in range(0,n-5):
    cur = data_train.iloc[i:i+5,[2,3,4,5]]
    temp = []
    for j in cur.values:
        temp+=list(j)
    X_train.append(temp)
    y_train.append(data_train.iloc[i+5,5])
    


# In[149]:


# X_train = pd.DataFrame(X_train)
# X_train = X_train.values
# import numpy as np
# c = np.array(X_train)[np.newaxis,:]
# y_train = np.array(y_train)
# y_train3 = y_train2[np.newaxis,:]
# y_train3.shape
# X_train2 = X_train2.reshape((464,1,20))
# X_train2.shape
# y_train[:,np.newaxis].shape
# y_train[:,np.newaxis][:,np.newaxis][:5]
# len(y_train)
# X_train2 = X_train2.reshape((2304,1,80))
# y_train = np.array(y_train)
# y_train.shape
X_train3 = X_train2.reshape((2319,5,4))
# X_train[:5]
# y_train[:5]
# X_train2.shape


# In[133]:


# data_train = data_train[::-1]
data_train.head()


# In[134]:


# len(X_train),len(y_train)
# y_train[:1]
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.layers import RepeatVector
import keras


# In[165]:


model = Sequential()

model.add(LSTM(6, input_shape=(X_train3.shape[1], X_train3.shape[2]), return_sequences=False))

# model.add(Dense(5,kernel_initializer="uniform",activation='relu'))        
model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

adam = keras.optimizers.Adam(learning_rate=0.1,decay=0.5)
model.compile(loss='mae', optimizer='adam')
model.summary()


# In[169]:


# model.get_weights()[3].shape
history = model.fit(X_train3, y_train, epochs=1000, verbose=2, shuffle=False)


# In[170]:


pred = model.predict(X_train3)
# len(y_train)
# X_train3.shape


# In[172]:


# import matplotlib.pyplot as plt
plt.figure()
plt.plot(pred)
plt.plot(y_train)
plt.show()


# In[9]:


from sklearn import preprocessing as process
scaler = process.StandardScaler()
scaler.fit(X)
X_scalerd = scaler.transform(X)
y = pd.DataFrame(X_scalerd)[3].values


# In[145]:


y_train[-6:]


# In[ ]:




 

