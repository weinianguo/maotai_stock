#方法一：采用class来定义
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import tensorflow as tf
    from tensorflow.keras import Model,layers,Input
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    input_shape=(64,1)
    class softmax_lstm_Model(Model):
        def __init__(self, **kwargs):
            super(softmax_lstm_Model,self).__init__(**kwargs)
            self.lstm=tf.keras.layers.LSTM(100)
            self.Dense1=layers.Dense(20,activation='relu')
            self.Dense2=layers.Dense(2,activation='softmax')
        def call(self,x):
            y=self.lstm(x)
            y=self.Dense1(y)
            y=self.Dense2(y)
            return y
    model_softmax=softmax_lstm_Model()
    
    model_softmax.compile("adam", loss='sparse_categorical_crossentropy',metrics=['binary_accuracy'])
    model_softmax.build(input_shape=(1000,64,1))
    model_softmax.call(Input(shape=(64,1)))  #这句话一定要，不然summary出现的模型的output shape会未知
    model_softmax.summary()

#方法二：采用input的add来定义
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import tensorflow as tf
    from tensorflow.keras import Model,layers,Input
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    input_shape=(64,1)
    
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    
    input_layer=Input(shape=(64,1))
    lstm=layers.LSTM(100)(input_layer)
    dense1=layers.Dense(20,activation='relu')(lstm)
    dense2=layers.Dense(2,activation='softmax')(dense1)
    model = keras.Model(input_layer,dense2)
    
    model.compile("adam", loss='sparse_categorical_crossentropy',metrics=['binary_accuracy'])
    history = model.fit(x_train, y_train, epochs=2,batch_size=512)
    print(model.predict(x_train))
    model.summary()

#方法三：采用Sequential来定义
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import tensorflow as tf
    from tensorflow.keras import Model,layers,Input
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    input_shape=(64,1)
    
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    
    model=tf.keras.Sequential([
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(20,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])
    
    model.compile("adam", loss='sparse_categorical_crossentropy',metrics=['binary_accuracy'])
    history = model.fit(x_train, y_train, epochs=2,batch_size=512)
    print(model.predict(x_train))
    model.summary()
            
#方法四：采用Sequential来定义
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import tensorflow as tf
    from tensorflow.keras import Model,layers,Input
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    input_shape=(64,1)
    
    x_train = np.random.rand(1000,64,1).astype(np.float32) 
    y_train = np.random.randint(0,2,(1000,1))
    
    
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    
    model.compile("adam", loss='sparse_categorical_crossentropy',metrics=['binary_accuracy'])
    history = model.fit(x_train, y_train, epochs=2,batch_size=512)
    print(model.predict(x_train))
    model.summary()

