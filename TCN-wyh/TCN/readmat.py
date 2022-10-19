import torch
import numpy as np
import h5py
import scipy.io as sio

def readmat(feat_dir, label_dir):
    # 读取.mat数据转换成tensor
    print('Reading data...')
    # feat = torch.from_numpy(sio.loadmat(feat_dir)['dataset'])
    # f = h5py.File(feat_dir)
    # a = f.get('dataset')
    # a = np.array(a)
    # a = np.transpose(a, (2,1,0))
    # p = a[432]
    feat = torch.from_numpy(np.transpose(np.array(h5py.File(feat_dir, mode='r').get('dataset')), (2, 1, 0)))
    print(feat_dir + ' has been loaded')
    # label = torch.from_numpy(sio.loadmat(label_dir)['labels'])
    label = torch.from_numpy(np.transpose(np.array(h5py.File(label_dir, mode='r').get('labels')), (1, 0)))
    print(label_dir + ' has been loaded')
    print('Data successfully loaded!')

    result = [feat, label]
    return result