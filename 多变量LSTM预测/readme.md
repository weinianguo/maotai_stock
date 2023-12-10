lstm模型的input_shape必须设置为(30,5)，
30为窗口的滑动距离，5为类别数量，也就是.shape[-2: ]应该就是这里会有点区别于1维度的预测；

数据create：dataX.append(dataset[i - n_past: i, 0:dataset.shape[1]])和dataY.append(data set[i,0])，此时的data的维度就变为了(数据量，seq_len, 维度)


trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)

#数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
#数据反归一化
# 将预测结果转回缩放之前（缩放之前是1011*5,但prediction是1011*1，所以要复制5列出来）
prediction_copies_array = np.repeat(prediction.reshape((-1,1)),5, axis=-1)
pred = scaler.inverse_transform(prediction_copies_array)[:,0]



#搜寻batch_size和ePochs的最佳参数
# grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
def build_model(optimizer='adam'):
    model = Sequential([
        LSTM(50,return_sequences=True,input_shape=(30,5)),
        LSTM(50),
        Dropout(0.2),
        Dense(1)  
    ])
    model.compile(loss = 'mse',optimizer = optimizer)
    return model
    
grid_model = KerasRegressor(build_model,verbose=1,optimizer='adam')

parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }
grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search.fit(trainX,trainY,validation_data=(testX,testY))
