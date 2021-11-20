import Sequential_Dense
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pre_data = np.load(file='../output/final_data_rates.npy')
input_data = []
lable = []
for i in pre_data:
    input_data.append(i[:, 0:-1])
    lable.append(i[:, -1])
input_data = np.array(input_data)
lable = np.array(lable)
# 改变数据形状
input_data = input_data.reshape(len(input_data), 320000)
lable = lable.reshape(len(lable), 6400)
print(input_data.shape)
print(lable.shape)
# 此时已经完成数据的输入输出分类，需要进行标准预处理
X_train, X_test, y_train, y_test = train_test_split(input_data, lable, test_size=0.2, shuffle=True,
                                                    random_state=None)
# creat model
model = KerasRegressor(build_fn=Sequential_Dense.create_model())
# define params
batch_size = [10, 20, 50, 80, 100]
epochs = [100, 200, 500]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
gird = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = gird.fit(X_train, y_train)

# output result
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
