from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
ps =pd.read_csv('train1.csv')
pt = pd.read_csv('validation.csv')
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
  
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
   
    return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)
x,y = to_xy(ps,"adview")
sc=StandardScaler()
x=sc.fit_transform(x)
x_test,y_test = to_xy(pt,"adview")
x_test = sc.fit_transform(x_test)       
model = Sequential()
model.add(Dense(60,input_dim=7,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.25))
model.add(Dense(50,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(BatchNormalization(axis=-1))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(100,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(80,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(70,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(35,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(26,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(26,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(40,activation='linear'))
model.add(LeakyReLU(alpha=0.08))
model.add(Dropout(0.5))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x,y,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)
pred = model.predict(x_test)
r2_score(y_test,pred)
rk = pd.read_csv('test.csv')
k = rk.drop('vidid',axis=1)
xtt = np.array(k)
y_pred = abs(model.predict(xtt))
rk['predicted_adview']= y_pred
rk.to_csv('prediction.csv')