import os
import numpy as np
import pandas as pd
import json
import torch
from torch import nn


def read_split_data(root, sam_rate):
    np.random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 类名列表
    mo_class = [os.path.splitext(cla)[0] for cla in os.listdir(root) if os.path.isfile(os.path.join(root, cla))]
    mo_class.sort()
    # 类名字典
    class_indices = dict((k, v) for v, k in enumerate(mo_class))
    # 存入json
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=17)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_mo_data = []  # 存储训练集的所有数据
    train_mo_label = []  # 存储训练集的标签
    var_mo_data = []  # 存储验证的所有数据
    var_mo_label = []  # 存储训练集的标签
    rate = 0.80

    for cla_i in range(0, len(mo_class)):

        dic_name = os.listdir(root)[cla_i]  # 文件名.后缀
        cla_path = os.path.join(root, dic_name)  # 文件路径

        data = pd.read_csv(cla_path, header=None)  # 读取数据

        num_sample = len(data) // sam_rate # 178

        for i in range(0, num_sample): #0-178

            if i <= num_sample * rate:  # 取出80%用于训练0--142
                train_mo_data.append(np.array(data.loc[i * sam_rate:(i + 1) * sam_rate - 1], dtype=np.float32))
                train_mo_label.append(class_indices[mo_class[cla_i]])

            else:  # 10%用于验证 # 143--177
                var_mo_data.append(np.array(data.loc[i * sam_rate:(i + 1) * sam_rate - 1], dtype=np.float32))
                var_mo_label.append(class_indices[mo_class[cla_i]])

    return train_mo_data, train_mo_label, var_mo_data, var_mo_label

# train_mo_data, train_mo_label, var_mo_data, var_mo_label = read_split_data('D:\work\py\lr_code\\vit\data', 401)
# print(var_mo_label)
# print(train_mo_data[1].shape)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    for step, data_label in enumerate(data_loader):
        data, label = data_label
        data, label = data.to(device), label.to(device)

        sample_num += data.shape[0]  # batch
        pred = model(data)  # batch,classes

        pred_classs = torch.max(pred, dim=1)[1]  # batch,1
        accu_num += torch.eq(pred_classs, label).sum()

        loss = loss_function(pred, label)
        loss.backward()

        accu_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
    print(f'train epoch{epoch} ,loss:{accu_loss.item() / (step + 1)},acc: {accu_num.item() / sample_num}')
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)  # 累计准确数量
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    for step, data_label in enumerate(data_loader):
        data, label = data_label
        data, label = data.to(device), label.to(device)
        sample_num += data.shape[0]
        pred = model(data)

        pred_classss = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classss, label).sum()
        loss = loss_function(pred, label)
        accu_loss += loss

    # print(f'train epoch{epoch} ,loss:{accu_loss.item() / (step + 1)},acc: {accu_num.item() / sample_num}')
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
