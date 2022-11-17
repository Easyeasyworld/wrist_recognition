import sklearn.metrics
import torch
# import numpy as np
# print(sklearn.__version__)
#
# y_true=[1,2,3]
# y_pred=[1,1,3]
#
# f1 = f1_score( y_true, y_pred, average='macro' )
# p = precision_score(y_true, y_pred, average='macro')
# r = recall_score(y_true, y_pred, average='macro')
#
# print(f1, p, r)
# output: 0.555555555556 0.5 0.666666666667


def acc(y, pred_y):
    return sklearn.metrics.precision_score(y, pred_y, average='macro')

def f1_score(y, pred_y):
    return sklearn.metrics.f1_score(y, pred_y, average='weighted' )

def classification_report(y, pred_y):
    target_names = ['movement 0', 'movement 1', 'movement 2', 'movement 2', 'movement 3', 'movement 4', 'movement 5',
                    'movement 6', 'movement 7', 'movement 8', 'movement 9', 'movement 10', 'movement 11', 'movement 12',
                    'movement 13', 'movement 14', 'movement 15', 'movement 16']
    return sklearn.metrics.classification_report(y, pred_y, target_names=target_names)

def confusion_matrix(y, pred_y):
    return sklearn.metrics.confusion_matrix(y, pred_y)

def DBI(X, labels):
    return sklearn.metrics.davies_bouldin_score(X, labels)


