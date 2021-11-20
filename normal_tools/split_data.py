from sklearn.model_selection import train_test_split
import numpy as np

def split_data(data_list, y_list, ratio=0.20):
    data_list = np.array(data_list)
    y_list = np.array(y_list)
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio,shuffle=True,random_state=None)
    return X_train, X_test, y_train, y_test
