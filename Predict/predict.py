from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from normal_tools import read_data
from normal_tools import save_data
from sklearn.model_selection import train_test_split
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    pre_data = np.load(file='../output/final_data_green_1.npy')
    input_data = []
    lable = []
    for i in pre_data:
        input_data.append(i[:, 0:-1])
        lable.append(i[:, -1])
    input_data = np.array(input_data)
    lable = np.array(lable)
    # print(input_data[0])
    # print(lable[0])
    print('完成输出读取')

    # 将输入输出转为np.array
    new_data = np.array(input_data)
    output_data = np.array(lable)

    new_data = new_data.reshape(len(new_data), 125000)
    output_data = output_data.reshape(len(output_data), 2500)
    print(new_data.shape)
    print(output_data.shape)

    # 划分训练集以及测试集
    X_train, X_test, y_train, y_test = train_test_split(new_data, output_data, test_size=0.2, shuffle=True,
                                                        random_state=None)

    print(X_test.shape)
    print(y_test.shape)
    # 读取模型
    model = load_model('../output/model/Dense_model_green_1.h5')
    pred = model.predict(X_test)
    print(pred.shape)

    print(pred[0])
    print(y_test[0])
    save_data.save_data('../output/pred_result/pred_with_rate_green=1.csv', pred)
    save_data.save_data('../output/pred_result/origin_with_rate_green=1.csv', y_test)
