import sklearn.metrics
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
    return sklearn.metrics.f1_score(y, pred_y, average='macro' )