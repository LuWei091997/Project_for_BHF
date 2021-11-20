import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

optimizer = tf.keras.optimizers.RMSprop(0.001)


def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(125000,), activation='softplus'))
    model.add(Dense(256, activation='softplus'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(512, activation='softplus'))
    model.add(Dense(2500, name='output_layer', activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    model.summary()
    return model


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] ='0'

    pre_data = np.load(file='../output/final_data_green_0.npy')
    input_data = []
    lable = []
    for i in pre_data:
        input_data.append(i[:, 0:-1])
        lable.append(i[:, -1])
    input_data = np.array(input_data)
    lable = np.array(lable)
    print(input_data.shape)
    print(lable.shape)
    # 改变数据形状
    input_data = input_data.reshape(len(input_data), 125000)
    lable = lable.reshape(len(lable), 2500)
    print(input_data[0])
    print(lable[0])
    # 此时已经完成数据的输入输出分类，需要进行标准预处理
    X_train, X_test, y_train, y_test = train_test_split(input_data, lable, test_size=0.2, shuffle=True,
                                                        random_state=None)
    model = create_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=2)

    # 保存模型
    model_path = '../output/model/Dense_model_green_0.h5'
    model.save(model_path)

    # 图
    plt.plot()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    end_time = time.time()
    print('总用时%fs' % (end_time - start_time))

# 存储参数部分
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    np_loss = np.array(loss).reshape(1, len(loss))
    np_val_loss = np.array(val_loss).reshape(1, len(val_loss))
    mae = np.array(mae).reshape(1, len(mae))
    np_val_mae = np.array(val_mae).reshape(1, len(val_mae))
    # 输出位置 loss, mae, val_loss, val_mae
    loss_data = np.concatenate([np_loss, mae, np_val_loss, np_val_mae], axis=0)
    np.savetxt('../output/model/pred_result/loss_data.txt', loss_data)
    '''
    from playsound import playsound
    path = '../sound/8517.wav'
    playsound(path)
    '''
    print('finish')
