import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math
from dataloader import data_load
from time import time
import pickle
import os

if not os.path.exists('./ckpt'):
    os.makedirs(os.path.join('./ckpt'))

train = True
methods = ['Linear discriminant analysis', 'Quadratic discriminant analysis', 'Neural networks', 'Support vector machines', 'Decision tree', 'Random forest']

def hparam_lists(method):
    if method == methods[0]:
        # solver
        hparams_1 = ['svd', 'lsqr', 'eigen']
        hparams_2 = ['None']
        hparams_3 = ['None']
    if method == methods[1]:
        # reg_param
        hparams_1 = [0.01, 0.1, 0.0]
        hparams_2 = ['None']
        hparams_3 = ['None']
    if method == methods[2]:
        # hidden_layer_sizes
        hparams_1 = [(100), (100, 50), (100,50,20)]
        # alpha
        hparams_2 = [0.001, 0.0001, 0.00005]
        # learning_rate_init
        hparams_3 = [0.002, 0.001, 0.0005]
    if method == methods[3]:
        # C
        hparams_1 = [1.0, 0.5, 0.1]
        # kernel
        hparams_2 = ['rbf', 'linear', 'poly']
        # decision_function_shape
        hparams_3 = ['ovo', 'ovr']
    if method == methods[4]:
        # max_depth
        hparams_1 = [50, 200, None]
        # max_feature
        hparams_2 = ['sqrt', 'log2', None]
        # min_samples_split
        hparams_3 = [10, 5, 2]
    if method == methods[5]:
        # n_estimators
        hparams_1 = [10, 100, 1000]
        # max_depth
        hparams_2 = [50, 200, None]
        # max_feature
        hparams_3 = ['sqrt', 'log2', None]
    return hparams_1, hparams_2, hparams_3

def load_model_with_hparams(method, hparam_1, hparam_2, hparam_3):
    if method == methods[0]:
        model = LinearDiscriminantAnalysis(solver=hparam_1)
    if method == methods[1]:
        model = QuadraticDiscriminantAnalysis(reg_param=hparam_1)
    if method == methods[2]:
        model = MLPClassifier(hidden_layer_sizes=hparam_1, alpha=hparam_2, learning_rate_init=hparam_3)
    if method == methods[3]:
        model = SVC(C=hparam_1, kernel=hparam_2, decision_function_shape=hparam_3)
    if method == methods[4]:
        model = DecisionTreeClassifier(max_depth=hparam_1, max_features=hparam_2, min_samples_split=hparam_3)
    if method == methods[5]:
        model = RandomForestClassifier(n_estimators=hparam_1, max_depth=hparam_2, max_features=hparam_3)
    return model

def best_model_search_and_save(method, x_train, x_valid, x_test, y_train, y_valid, y_test, hparams_1, hparams_2, hparams_3):
    best_acc = 0
    for hparam_1 in hparams_1:
        for hparam_2 in hparams_2:
            for hparam_3 in hparams_3:
                t = time()
                model = load_model_with_hparams(method, hparam_1, hparam_2, hparam_3)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_valid)
                accuracy = accuracy_score(y_valid, y_pred)
                print('%s with %s, %s and %s, validation accuracy : %.03f, the time required : %.01f (sec)' % (method, hparam_1, hparam_2, hparam_3, accuracy, time() - t))
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_params = (hparam_1, hparam_2, hparam_3)
                    best_model = model

    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('==== Best model : %s with %s, %s and %s==, test accuracy : %.03f' % (method, *best_params, accuracy))
    print('==== Confusion matrix (test) : \n %s' % (cf_matrix))
    fname = os.path.join('ckpt',"best_%s.pkl" % method)
    pickle.dump(best_model, open(fname, 'wb'))

def model_load_and_test(method, x_test, y_test):
    fname = os.path.join('ckpt',"best_%s.pkl" % method)
    loaded_model = pickle.load(open(fname, 'rb'))
    y_pred = loaded_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('==== Loaded model : %s, test accuracy : %.03f' % (method, accuracy))
    print('==== Confusion matrix (test) : \n %s' % (cf_matrix))

# Data Load
x_train, y_train = data_load(split='train')
x_train = np.reshape(x_train, (x_train.shape[0], -1))
print("train data : ", x_train.shape)
print("train label : ", y_train.shape)

x_valid, y_valid = data_load(split='valid')
x_valid = np.reshape(x_valid, (x_valid.shape[0], -1))
print("valid label : ", x_valid.shape)
print("valid label : ", y_valid.shape)

x_test, y_test = data_load(split='test')
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print("test data : ", x_test.shape)
print("test label : ", y_test.shape)

# small data for debugging
# x_train, y_train = x_train[:2000], y_train[:2000]
# x_valid, y_valid = x_valid[:500], y_valid[:500]
# x_test, y_test = x_test[:500], y_test[:500]

# Experiments
if train:
    for method in methods:
        hparams_1, hparams_2, hparams_3 = hparam_lists(method)
        best_model_search_and_save(method, x_train, x_valid, x_test, y_train, y_valid, y_test, hparams_1, hparams_2, hparams_3)
else:
    for method in methods:
        model_load_and_test(method, x_test, y_test)
