import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 程序可见的显卡ID
import torch

from myDataset import myDataset as myDataset  # 用于DataLoader？
from torch import nn
from Model import TCN
import numpy as np
from torch.utils.data import DataLoader  # 数据迭代器
from metric import f1_score
from tensorboardX import SummaryWriter
import time


def train(k, net, traindata, trainlabel, testdata, testlabel, EPOCH, LR, WD, BATCH, DROP, position, feature, obj):
    max_test_acc = 0
    train_ls, test_ls = [], []
    train_acc, test_acc = [], []
    train_f1, test_f1 = [], []

    # writer = SummaryWriter('runs/' + position + '/' + feature + '/s' + str(obj) + '/fold' + str(k+1) + '/' + str(time.strftime("%m-%d-%H-%M", time.localtime())))

    data_read_train = myDataset(traindata, testdata, trainlabel, testlabel, train=True)
    # 提供__getitem__,__len__给DataLoader
    loader_train = DataLoader(dataset=data_read_train, batch_size=BATCH, shuffle=True, drop_last=True)
    # 根据传入接口的参数将训练集分为若干个大小为batch size的batch以及其他一些细节上的操作
    data_read_test = myDataset(traindata, testdata, trainlabel, testlabel, train=False)
    loader_test = DataLoader(dataset=data_read_test, batch_size=BATCH, shuffle=False,
                             drop_last=True)  # drop_last在Batch比较大的时候容易把最后一个标签的样本都丢了

    model = net
    # print(model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=0.9)  # optimize all parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)  # 使用Adam优化模型参数？
    loss_func = nn.CrossEntropyLoss()  # 计算交叉熵损失

    if torch.cuda.is_available():
        print('available')
        model = model.cuda()

    for epoch in range(EPOCH):
        total_acc_train = 0  # 准确率
        total_f1_train = 0  # 精确率和召回率的调和均值
        total_loss_train = 0
        class_correct = list(0. for i in range(17))  # 定义一个存储每类中测试正确的个数的列表，初始化为0
        class_total = list(0. for i in range(17))   # 定义一个存储每类中测试总数的个数的列表，初始化为0

        model.train()   # 训练模型

        for step, (x, y) in enumerate(loader_train):  # 用DataLoader对象来不断地取batch数据
            # x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))  # x(b, 6, 200)-->x(b, 200, 6)
            x = torch.Tensor(np.array(x, dtype='float32'))  # 创建Tensor
            # x = torch.unsqueeze(x, 1)  # 对数据维度进行扩充   x(b,1,12,200)
            y = y.type(torch.LongTensor)
            y = torch.squeeze(y)    # 数据维度压缩

            output = model(x.cuda())
            pred_y = torch.max(output, 1)[1]    # 索引每行的最大值

            # total_acc_train += (torch.eq(pred_y, y.cuda()).sum()) / float(y.size(0))
            c = torch.eq(pred_y, y.data.cuda())    # 对两个张量Tensor进行逐元素的比较
            total_acc_train += (c.sum()) / float(y.size(0))
            for i in range(BATCH):
                label = y[i]
                label = label.int()
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total_f1_train += f1_score(y.cpu(), pred_y.cpu())

            loss_train = loss_func(output, y.cuda())
            total_loss_train += loss_train
            optimizer.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # 反向传播, 计算梯度
            optimizer.step()  # 应用梯度更新权重
        avr_acc_train = total_acc_train / (step + 1)
        avr_f1_train = total_f1_train / (step + 1)
        avr_loss_train = total_loss_train / (step + 1)
        print("Epoch: %d, trainAcc: %.2f %%" % (epoch + 1, avr_acc_train * 100))

        # for i in range(17):
        #     print('Train accuracy of %d : %.2f %%' % (i+1, (class_correct[i] / (class_total[i]))*100))
        # writer.add_scalar('train/Movement' + str(i + 1) + 'acc', class_correct[i] / (class_total[i]), epoch)
        # writer.add_scalar('train/Accurancy', avr_acc_train, epoch)
        train_acc.append(avr_acc_train.item())  # append函数会在数组后加上相应的元素
        # writer.add_scalar('train/F1', avr_f1_train, epoch)
        train_f1.append(avr_f1_train)
        # writer.add_scalar('train/Loss', avr_loss_train, epoch)
        train_ls.append(avr_loss_train.item())

        if epoch + 1 == EPOCH / 2:
            for p in optimizer.param_groups:
                p['lr'] = LR / 2
                LR = p['lr']

        # writer.add_scalar('Hyperpram/LR', LR, epoch)
        # writer.add_scalar('Hyperpram/DROP', DROP, epoch)
        # writer.add_scalar('Hyperpram/srccnn3_WD', WD, epoch)
        # writer.add_scalar('Hyperpram/BATCH', BATCH, epoch)

        total_acc_test = 0
        total_f1_test = 0
        total_loss_test = 0
        class_correct_test = list(0. for i in range(17))
        class_total_test = list(0. for i in range(17))

        model.eval()

        with torch.no_grad():
            for step, (x, y) in enumerate(loader_test):
                # x = torch.Tensor(np.transpose((np.array(x, dtype='float32')), (0, 2, 1)))
                x = torch.Tensor(np.array(x, dtype='float32'))
                # x = torch.unsqueeze(x, 1)
                y = y.type(torch.LongTensor)
                y = torch.squeeze(y)

                output_test = model(x.cuda())
                pred_y_test = torch.max(output_test, 1)[1]

                # total_acc_test += (torch.eq(pred_y_test, y.data.cuda()).sum()) / float(y.size(0))
                c_test = (torch.eq(pred_y_test, y.data.cuda()))
                total_acc_test += (c_test.sum()) / float(y.size(0))
                for i in range(BATCH):
                    label_test = y[i]
                    label_test = label_test.int()
                    class_correct_test[label_test] += c_test[i].item()
                    class_total_test[label_test] += 1
                total_f1_test += f1_score(y.cpu(), pred_y_test.cpu())

                loss_test = loss_func(output_test, y.cuda())
                total_loss_test += loss_test
            avr_acc_test = total_acc_test / (step + 1)
            avr_f1_test = total_f1_test / (step + 1)
            avr_loss_test = total_loss_test / (step + 1)
            print("Epoch: %d, testAcc: %.2f %%" % (epoch + 1, avr_acc_test * 100))

            # for i in range(17):
            #     print('Train accuracy of %d : %.2f %%' % (i+1, (class_correct_test[i] / (class_total_test[i]))*100))
            # writer.add_scalar('test/Movement' + str(i + 1) + 'acc', class_correct_test[i] / (class_total_test[i]),epoch)
            if epoch + 1 >= EPOCH / 2:
                if avr_acc_test > max_test_acc:
                    max_test_acc = avr_acc_test
                    print('max_test_acc:', max_test_acc, 'epoch:', epoch + 1)
                    # torch.save(model, 'save/myModel.pkl')
            # writer.add_scalar('test/Accurancy', avr_acc_test, epoch)
            test_acc.append(avr_acc_test.item())
            # writer.add_scalar('test/F1', avr_f1_test, epoch)
            test_f1.append(avr_f1_test)
            # writer.add_scalar('test/Loss', avr_loss_test, epoch)
            test_ls.append(avr_loss_test.item())
            # writer.add_scalar('test/Maxtestacc', max_test_acc, epoch)

    return train_ls, train_acc, train_f1, test_ls, test_acc, test_f1
