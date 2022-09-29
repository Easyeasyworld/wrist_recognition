import torch


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('HuaXi')

    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--wd', type=float, default=0.)
    parser.add_argument('--winsize', type=int, default=409)
    parser.add_argument('--wininc', type=int, default=20)

    parser.add_argument('--day', type=int, default=1)
    parser.add_argument('--obj', type=int, default=1)
    parser.add_argument('--position', type=str, default='wrist')
    parser.add_argument('--feature', type=str, default='diff')
    parser.add_argument('--modelname', type=str, default='cnn')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--adjustlr', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=1)
    args = parser.parse_args()

    # Hyper Parameters
    EPOCH = args.epoch
    print('EPOCH:', EPOCH)
    LR = args.lr
    print('LR:', LR)
    DROP = args.drop
    print('DROP:', DROP)
    BATCH = args.batch
    print('BATCH:', BATCH)
    WD = args.wd
    print('WD:', WD)
    WINSIZE = args.winsize
    print('WINSIZE:', WINSIZE)
    WININC = args.wininc
    print('WINSIZE:', WININC)

    # Default configurations
    day = args.day
    print('day:', day)
    obj = args.obj
    print('obj:', obj)
    position = args.position
    print('position:', position)
    feature = args.feature
    print('feature:', feature)
    modelname = args.modelname
    print('modelname:', modelname)
    optimizer = args.optimizer
    print('optimizer:', optimizer)
    gpu = args.gpu
    print('gpu:', gpu)
    adjustlr = args.adjustlr
    print('adjstlr:', adjustlr)
    normalize = args.normalize
    print('normalize:', normalize)

    from readmat import readmat


    _, X, y = readmat('D:/EMG/Dataset/HuaXi/Session 1/DataExtracted' + str(obj) + '.mat', winsize=WINSIZE, wininc=WININC,
                    pos=position, normalize=normalize)
    # tmp = torch.mean(X[0], axis=0) # 验证数据是否去中心化了，看tmp是不是0: e^-20, 就是0了
    from kfold import k_fold
    train_l, test_l, train_acc, test_acc, train_f1, test_f1 = k_fold(k=7, modelname=modelname, gpu=gpu, position=position, obj=obj,
                                                                     X=X, y=y,
                                                                     EPOCH=EPOCH, LR=LR, WD=WD, BATCH=BATCH, DROP=DROP,
                                                                     adjustlr=adjustlr, optimizer=optimizer)
    import numpy as np

    print('trainloss:%.2f, testloss:%.2f' % (np.min(train_l), np.min(test_l)))
    print('trainacc:%.2f%%, testacc:%.2f%%' % (np.max(train_acc) * 100, np.max(test_acc) * 100))
    print('trainf1:%.2f, testf1:%.2f' % (np.max(train_f1), np.max(test_f1)))

    # plot train/test curve (avr results of k folds)
    from plotcurve import plotcurve

    plotcurve(modelname, day, obj, position, train_l, test_l, train_acc, test_acc, train_f1, test_f1)

    # write max acc and min loss into .txt (avr results of k folds)
    with open('./result/' + position + '_' + modelname + '_result.txt', "a") as f:
        f.write('\n')
        f.write('EPOCH:' + str(EPOCH) + ' LR:' + str(LR) + ' DROP: ' + str(DROP) + ' BATCH: ' + str(
            BATCH) + ' WEIGHT_DECAY: ' + str(WD) + '\n')
        import time

        t = time.localtime(time.time())
        execute_time = time.asctime(t)
        f.write(str(execute_time) + '\n')
        f.write('s' + str(obj) + ': trainacc ' + str(np.max(train_acc)) + '; testacc ' + str(
            np.max(test_acc)) + '\n')  # 自带文件关闭功能，不需要再写f.close()
        f.write('s' + str(obj) + ': trainloss ' + str(np.min(train_l)) + '; testloss ' + str(np.min(test_l)) + '\n')


