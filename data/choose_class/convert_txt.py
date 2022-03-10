from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata
import os
import h5py
import torchaudio
import random

# _h5file = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/eval.h5'
# hf = h5py.File(_h5file, 'r')
#     # X = hf['mel_feature'][:].astype(np.float32)
# y = hf['label'][:].astype(np.float32)
# events = np.array([target_event.decode() for target_event in hf['events'][:]])
# filename = np.array([filename.decode() for filename in hf['filename'][:]])
# pre = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/'
# for i in range(filename.shape[0]):
#     save_name = pre + filename[i][:-4] + '.txt'
#     np.savetxt(save_name,hf['mel_feature'][i].astype(np.float32))

# read txt
# name = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/' + 'YYxlGt805lTA.txt'
# dt = np.loadtxt(name)
# print(dt.shape)

_h5file = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/eval_choose.h5'
hf = h5py.File(_h5file, 'r')
    # X = hf['mel_feature'][:].astype(np.float32)
y = hf['label'][:].astype(np.float32)
events = np.array([target_event.decode() for target_event in hf['events'][:]])
filename = np.array([filename.decode() for filename in hf['filename'][:]])

h5_train_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/'  + 'eval_no_choose.h5'
print(h5_train_name)
hf_train = h5py.File(h5_train_name, 'w')
num_ = 100
frames_num = 501
num_freq_bin = 64
hf_train.create_dataset(
    name='filename', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S80')
hf_train.create_dataset(
    name='events', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S900')
hf_train.create_dataset(
    name='label',
    shape=(0,num_,2),
    maxshape=(None,num_,2),
    dtype=np.float32)
n = 0 
for i in range(y.shape[0]):
    print('train ',filename[i],n)
    hf_train['filename'].resize((n+1,))
    hf_train['filename'][n] = filename[i].encode()
    
    hf_train['events'].resize((n+1,))
    hf_train['events'][n] = events[i].encode()
    hf_train['label'].resize((n+1,num_,2))
    hf_train['label'][n] = y[i].astype(np.float32)
    n += 1


