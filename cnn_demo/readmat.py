import torch
import numpy as np
import scipy.io as sio
import math
from matplotlib import pyplot as plt

def readmat(datadir, winsize, wininc, pos='wrist', normalize=1):
    print('Reading data...')
    data = sio.loadmat(datadir)['DATA']
    print(datadir + ' has been loaded')
    print('Trying to get differential' + pos + ' signals...')
    if pos == 'forearm':
        for i in range(7):
            for j in range(17):
                data[i, j] = data[i, j][:, [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]]
                data[i, j] = data[i, j][:, 0:6] - data[i, j][:, 6:12]
        posdata = data
    else:
        for i in range(7):
            for j in range(17):
                data[i, j] = data[i, j][:, [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30]]
                data[i, j] = data[i, j][:, 0:6] - data[i, j][:, 6:12]
        posdata = data
    print('Completed!')
    numsamples_sigcell = math.floor((10240-winsize)/wininc) + 1
    num_trial = data.shape[0]
    num_mov = data.shape[1]
    num_ch = data[0, 0].shape[1]
    samples = np.zeros((num_mov*num_trial*numsamples_sigcell, winsize, num_ch))
    labels = np.zeros((num_mov*num_trial*numsamples_sigcell, 1))
    print('Spliting samples...')
    for i in range(7):
        for j in range(17):
            for index in range(numsamples_sigcell):
                # 去中心化，让每个样本的各通道均值为0
                samples[(j+i*17)*numsamples_sigcell+index] = data[i, j][index*wininc:index*wininc+winsize] - np.mean(data[i, j][index*wininc:index*wininc+winsize], axis=0)
                labels[(j+i*17)*numsamples_sigcell+index] = j
    print('Samples Gotten Already!')
    if normalize == 1:
        print('Normalizing data...')
        for i in range(samples.shape[0]):
            samples[i] = (samples[i] - samples[i].min(axis=0)) / (samples[i].max(axis=0) - samples[i].min(axis=0))
        print('Normalized Samples Gotten Already!')
    samples = torch.from_numpy(samples)
    labels = torch.from_numpy(labels)
    print("样本总数：" + str(len(labels)))
    return posdata, samples, labels

def verifysplit(posdata, samples, winsize):
    data = None
    for i in range(7):
        for j in range(17):
            if data is None:
                data = posdata[i, j]
            else:
                data = np.concatenate((data, posdata[i, j]), axis=0)
    plt.plot(data)
    plt.savefig("before.png")
    plt.close()

    numsamples_sigcell = math.floor((10240 - winsize) / winsize) + 1
    for k in range(len(samples)):
        sample = samples[k]
        x = np.linspace((math.floor(k / numsamples_sigcell) * (
                    numsamples_sigcell - 1) + k % numsamples_sigcell) * winsize + math.floor(k / numsamples_sigcell) * (
                                    10240 % winsize),
                        (math.floor(k / numsamples_sigcell) * (
                                    numsamples_sigcell - 1) + k % numsamples_sigcell) * winsize + math.floor(
                            k / numsamples_sigcell) * (10240 % winsize) + winsize - 1,
                        winsize)
        plt.plot(x, sample, linewidth=1.0)  # 只画第一个通道
    plt.savefig("after")
    plt.close()

    print("Split verified")

