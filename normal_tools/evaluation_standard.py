from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import math

def fun_mean_squared_error(y_true,y_pred):
    score = mean_squared_error(y_true,y_pred)
    return score

def fun_root_mean_square_error(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred)
    return math.sqrt(score)

def fun_mean_absolute_error(y_true, y_pred):
    score = mean_absolute_error(y_true, y_pred)
    return score

def fun_mean_absolute_percentage_error(y_true, y_pred):
    score = mean_absolute_percentage_error(y_true, y_pred)
    return score

def fun_r2_score(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    return score

