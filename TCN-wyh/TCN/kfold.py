import matplotlib.pyplot as plt
import numpy as np
import torch
from Model import TCN
from train import train


# 对数据进行划分，以下代码为固定格式
def get_k_fold_data(k, i, X, y):  # k:划分的折数，i:第i块为验证集，
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 双斜杠表示除完后再向下取整,fold_size为每块的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # 在给定维度上对输入的张量序列进行连接操作；dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, EPOCH, LR, WD, BATCH, DROP, position, feature, obj):
    train_l_sum, test_l_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    train_acc_sum, test_acc_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    train_f1_sum, test_f1_sum = np.zeros(EPOCH), np.zeros(EPOCH)
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        net = TCN()  # 对数据进行隐藏层处理的结构
        print('Fold %d' % (i + 1))
        train_ls, train_acc, train_f1, test_ls, test_acc, test_f1 = train(i, net, *data, EPOCH, LR, WD, BATCH, DROP,
                                                                          position, feature, obj)
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

    # 生成图像
    plt.figure(1)
    plt.title('loss')
    plt.plot(train_l_sum / k)
    plt.plot(test_l_sum / k)
    plt.savefig('loss')
    # plt.show()
    plt.figure(2)
    plt.title('acc')
    plt.plot(train_acc_sum / k)
    plt.plot(test_acc_sum / k)
    plt.savefig('acc')
    # plt.show()
    plt.figure(3)
    plt.title('f1')
    plt.plot(train_f1_sum / k)
    plt.plot(test_f1_sum / k)
    plt.savefig('f1')
    # plt.show()
    return train_l_sum / k, test_l_sum / k, train_acc_sum / k, test_acc_sum / k, train_f1_sum / k, test_f1_sum / k
    # 返回均值
