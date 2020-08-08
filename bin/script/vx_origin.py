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
        self.pro = ts.pro_api('xxxxxxxxxx')

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


n = data_train.shape[0]
#选取过去7天的'high','low','close'数据作为特征，并进行归一化
length = 7  
X_train,y_train = [],[]
#提取训练数据
for i in range(0,n-length):
    cur = data_train.iloc[i:i+length,[3,4,5]]
    temp = []
    for j in cur.values:
        temp+=list(j)
    X_train.append(list((np.array(temp)-min(temp))/(max(temp)-min(temp))))
    y_train.append((data_train.iloc[i+length,5]-min(temp))/(max(temp)-min(temp)))
    


# In[334]:


n = data_test.shape[0]
# length = 7
#提取测试数据
X_test,y_test = [],[]
for i in range(0,n-length):
    cur = data_test.iloc[i:i+length,[3,4,5]]
    temp = []
    for j in cur.values:
        temp+=list(j)
    X_test.append(list((np.array(temp)-min(temp))/(max(temp)-min(temp))))
    y_test.append((data_test.iloc[i+length,5]-min(temp))/(max(temp)-min(temp)))


# In[335]:


X_train = np.array(X_train)[np.newaxis,:]
X_train = X_train.reshape((X_train.shape[1],length,3))

X_test = np.array(X_test)[np.newaxis,:]
X_test = X_test.reshape((X_test.shape[1],length,3))


# In[271]:


# data_train = data_train[::-1]
# data_train.head()
# np.array(X_test).shape
y_test[:5]


# In[365]:


model = Sequential()

model.add(LSTM(6, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))

# model.add(Dense(5,kernel_initializer="uniform",activation='relu'))        
model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

adam = keras.optimizers.Adam(learning_rate=0.07,decay=0.1)
model.compile(loss='mae', optimizer='adam')
model.summary()


# In[366]:


# model.get_weights()[3].shape
history = model.fit(X_train, y_train, epochs=1000, verbose=2, shuffle=False)


# In[367]:


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(pred)
# plt.plot(y_train)
# plt.show()
pred = model.predict(X_train)
result = pd.DataFrame(X_train[:,-1,-1]<y_train,columns=['tlabel'])
result['plabel'] = list(X_train[:,-1,-1]<pred[:,0])
print('train acc:',result[result.tlabel==result.plabel].shape[0]/result.shape[0])

pred = model.predict(X_test)
result = pd.DataFrame(X_test[:,-1,-1]<y_test,columns=['tlabel'])
result['plabel'] = list(X_test[:,-1,-1]<pred[:,0])
print('test acc',result[result.tlabel==result.plabel].shape[0]/result.shape[0])
# result.head()
# X_train[:,-1,-1]
# result.plabel.value_counts()
