### Data Collection
import pandas_datareader as pdr
df=pdr.DataReader('TSLA', data_source='yahoo', start='2015-01-01', end='2021-01-23')
df.to_csv('TSLA.csv')
import pandas as pd
df=pd.read_csv('TSLA.csv')
df1=df.filter(['Close'])

import matplotlib.pyplot as plt
plt.plot(df1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))



##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
print(training_size, test_size)