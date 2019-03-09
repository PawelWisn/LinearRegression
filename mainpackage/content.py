# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''
    desMat = design_matrix(x, w.shape[0] - 1)
    modelPred = desMat @ w
    Q = 0
    for i in range(y.shape[0]):
        Q += (y[i] - desMat[i]@w) ** 2
    mse = Q / x.shape[0]
    return np.array(*mse)


def design_matrix(x_train, M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''
    desMat = []
    for row in x_train:
        desMat.append([row ** i for i in range(M + 1)])
    desMat = np.reshape(desMat, (-1, M + 1))
    return desMat
    # pass


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    desMat = design_matrix(x_train, M)

    w = np.linalg.inv(desMat.transpose() @ desMat) @ desMat.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    desMat = design_matrix(x_train, M)

    w = np.linalg.inv(desMat.transpose() @ desMat + (regularization_lambda * np.identity(M+1)))\
        @ desMat.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''
    w_m_train_arr = []
    err_m_train_arr = []
    for m in M_values:
        w, err = least_squares(x_train, y_train, m)
        w_m_train_arr.append(w)
        err_m_train_arr.append(err)

    err_m_val_arr = []
    for m in M_values:
        err_m_val_arr.append(mean_squared_error(x_val, y_val, w_m_train_arr[m]))

    min_err_val_index = err_m_val_arr.index(min(err_m_val_arr))

    err_m_train = err_m_train_arr[min_err_val_index]
    chosen_w = w_m_train_arr[min_err_val_index]

    return (chosen_w, err_m_train, min(err_m_val_arr))

def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''
    pass
