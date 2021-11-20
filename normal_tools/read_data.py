import pandas as pd
import numpy as np

def readfile(filename):
    data = pd.read_csv(filename, header=None,encoding="gbk")
    listx = data.values.tolist()
    dataset = []
    for i in range(len(listx)):
        a = listx[i]
        dataset.append([float(j) for j in a[0:len(a)]])
    dataset = np.array(dataset)
    return dataset
