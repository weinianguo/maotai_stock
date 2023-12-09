import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


maotai = pd.read_csv('./SH600519.csv')
training_set = maotai.iloc[0:5335 - 300, 2:3].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = maotai.iloc[5335 - 300:, 2:3].values  # 后300天的开盘价作为测试集

#归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set_scaled = sc.transform(test_set) 

#创建时序数据，用前时刻数据预测后面的数据
def create_dataset(dataset, time_step=60):  #适用于二维度的数据
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		x = dataset[i:(i+time_step)]  
		y = dataset[i:(i+time_step)]
		dataX.append(x)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

train_x, train_y = create_dataset(training_set_scaled)
test_x, test_y = create_dataset(test_set_scaled)

#模型搭建
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
#数据的拟合和训练
history = model.fit(train_x, train_y, batch_size=64, epochs=500, validation_data=(test_x, test_y), validation_freq=1,
                    callbacks=[cp_callback])
