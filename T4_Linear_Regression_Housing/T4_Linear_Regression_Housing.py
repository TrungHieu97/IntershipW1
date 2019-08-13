import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
random.seed(7)

def fit(X,y, Lambda):
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T.dot(X) + Lambda*I).dot(X.T.dot(y))

def predict(X,w):
    return X.dot(w)


def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data.drop(["id" , "date", "zipcode"], axis= 1)
    return data

def split_data(data, rate = 0.2):
    data_train , data_test = train_test_split(data, test_size= rate)
    data_train , data_valid = train_test_split(data_train, test_size= rate)
    return data_train, data_valid, data_test

def nomalized_data(data):
    x_min = np.min(data, axis = 0)
    x_range =np.max(data, axis = 0) - np.min(data,axis=0)
    x_normalized = (data - x_min)/(x_range + 1e-8)
    return x_normalized, x_min, x_range

def normalized_test(data, min, range):
    return (data - min)/(range + 1e-8)

def train_test_separate(data):
    X = data.drop('price',axis = 1)
    y = data['price']
    return X,y

def error_function(y_pred, y_real):
    return np.sum(np.square(y_real - y_pred))/(2*len(y_pred))


def update_W_by_GD(X, y, W0, iter = 100, learning_rate = 0.01):
    N = X.shape[0]
    for i in np.arange(iter):
        w_grad = (1/N)*X.T.dot((X.dot(W0) - y))
        W0 = W0 - w_grad*learning_rate
    return W0

def update_W_by_SGD(X, y, W0, iter = 100, learning_rate = 0.01):
    N = X.shape[0]
    for i in np.arange(N):
        w_grad = (1/N)*X[i].T.dot((X[i].dot(W0) - y[i]))
        W0 = W0 - w_grad*learning_rate
    return W0


if __name__ == '__main__':
    Lambda = 0.01
    data = load_data('kc_house_data.csv')
    # print (data.shape)
    data_train, data_valid, data_test = split_data(data)
    # print(data_train.shape)
    # print(data_valid.shape)
    # print(data_test.shape)

    X_train, y_train = train_test_separate(data_train)
    X_valid, y_valid = train_test_separate(data_valid)
    X_test, y_test = train_test_separate(data_test)

    X_train, min, range = nomalized_data(X_train)
    X_valid = normalized_test(X_valid, min, range)
    X_test = normalized_test(X_test, min, range)

    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    X_valid = np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

    w_1 = fit(X_train, y_train, Lambda)

    print("From function :")
    print("W1 :")
    print(w_1)

    print("Error 1 :")
    print(error_function(predict(X_test, w_1), y_test))

    # Find the best learning rate
    w_init = np.zeros(X_train.shape[1])
    list_leanrning_rate = [a/100 for a in np.arange(100)]
    w = []
    error = []
    for lr in list_leanrning_rate:
        w.append(update_W_by_GD(X_train,y_train,W0 = w_init, iter =100, learning_rate= lr))
        y_pred = predict(X_valid, w[-1])
        error.append(error_function(y_pred, y_valid))
    #
    best_learning_rate = list_leanrning_rate[error.index(np.min(error))]
    #
    print("From gradient :")
    w_2 = update_W_by_GD(X_train, y_train, W0=w_init, iter=100, learning_rate= 0.65)
    print("W2 :")
    print(w_2)
    print("Error 2 :")
    print(error_function(predict(X_test, w_2), y_test))


from sklearn.linear_model import LinearRegression
print("From library :")
reg = LinearRegression().fit(X_train,y_train)
w_3 = reg.coef_
w_3[0] = reg.intercept_
print("W3 :")
print(w_3)
print("Error 3 :")
print(error_function(predict(X_test, w_3), y_test))

