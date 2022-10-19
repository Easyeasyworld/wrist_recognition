# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # argparse解析命令行参数和选项
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')  # 创建解析对象
    parser.add_argument('--epoch', type=int, default=15)  # 添加要关注的命令行参数和选项
    parser.add_argument('--lr', type=float, default=0.0002)  # 学习率
    # parser.add_argument('--lrstep', type=int, default=15)
    parser.add_argument('--drop', type=float, default=0.3)
    # parser.add_argument('--time_step', type=int, default=1)
    # parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--wd', type=float, default=0.0005)  # 权重衰减
    parser.add_argument('--obj', type=int, default=1)
    args = parser.parse_args()  # 调用parse_args()方法进行解析
    # Hyper Parameters
    EPOCH = args.epoch
    print('EPOCH:', EPOCH)
    LR = args.lr
    print('LR:', LR)
    # LRSTEP = args.lrstep
    # print('STEP:', LRSTEP)
    DROP = args.drop
    print('DROP:', DROP)
    # TIME_STEP = args.time_step
    # print('TIME_STEP:', TIME_STEP)
    # WINDOW = args.window
    # print('WINDOW:', WINDOW)
    BATCH = args.batch
    print('BATCH:', BATCH)
    WD = args.wd
    print('WD:', WD)
    obj = args.obj
    print('obj:', obj)
    feature = 'rms'
    position = 'wrist'

    # 读取数据
    from readmat import readmat
    X, y = readmat('F:\\Study\\data\\s' + str(obj) + '.mat',
                   'F:\\Study\\label\\s' + str(obj) + '.mat')
    # 交叉验证
    from kfold import k_fold

    train_l, test_l, train_acc, test_acc, train_f1, test_f1 = k_fold(7, X, y, EPOCH, LR, WD, BATCH, DROP, position,
                                                                     feature, obj)
    import numpy as np

    print('obj:', obj)
    print('trainloss:%.2f, testloss:%.2f' % (np.max(train_l), np.max(test_l)))
    print('trainacc:%.2f%%, testacc:%.2f%%' % (np.max(train_acc) * 100, np.max(test_acc) * 100))
    print('trainf1:%.2f, testf1:%.2f' % (np.max(train_f1), np.max(test_f1)))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
