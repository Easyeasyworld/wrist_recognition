import matplotlib.pyplot as plt
import numpy as np
import torch
from myModel import get_net
from train import train

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 双斜杠表示除完后再向下取整
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, modelname, gpu, position, obj,  X, y, EPOCH, LR, WD, BATCH, DROP, adjustlr, optimizer):
    train_l_sum, test_l_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    train_acc_sum, test_acc_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    train_f1_sum, test_f1_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    for i in range(k):
        data = get_k_fold_data(k, i, X, y)  # 获取k折交叉验证的训练和验证数据
        net = get_net(modelname=modelname, batch=BATCH, drop=DROP, in_channels=X.shape[2], )
        print('Fold %d' % (i+1))
        train_ls, train_acc, train_f1, test_ls, test_acc, test_f1 = train(net, gpu, position, obj, i, *data,
                                                                          EPOCH, LR, WD, DROP, BATCH,
                                                                          adjustlr, optimizer)
        # list转成array
        train_ls = np.array(train_ls)
        train_acc = np.array(train_acc)
        train_f1 = np.array(train_f1)
        test_ls = np.array(test_ls)
        test_acc = np.array(test_acc)
        test_f1 = np.array(test_f1)
        train_l_sum += train_ls
        test_l_sum += test_ls
        train_acc_sum += train_acc
        test_acc_sum += test_acc
        train_f1_sum += train_f1
        test_f1_sum += test_f1

    return train_l_sum / k, test_l_sum / k, train_acc_sum / k, test_acc_sum/k, train_f1_sum / k, test_f1_sum / k
