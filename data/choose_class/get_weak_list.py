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

_h5file = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/train_choose.h5'
hf = h5py.File(_h5file, 'r')
    # X = hf['mel_feature'][:].astype(np.float32)
# y = hf['label'][:].astype(np.float32)
# events = np.array([target_event.decode() for target_event in hf['events'][:]])
filename = np.array([filename.decode() for filename in hf['filename'][:]])
tar_filenames = []
for name in filename:
    if name not in tar_filenames:
        tar_filenames.append(name)

train_weak = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/weak_train_choose.tsv',sep='\t',usecols=[0,1])
weak_file_name = train_weak['filename']
weak_labels = train_weak['event_labels']
weak_file_name_ls = []
weak_labels_ls = []
for i in weak_file_name:
    weak_file_name_ls.append(i)
for i in weak_labels:
    weak_labels_ls.append(i)

weak_train_choose_filename = []
weak_train_choose_labels = []

weak_validate_choose_filename = []
weak_validate_choose_labels = []
for i in range(len(weak_file_name_ls)):
    if weak_file_name_ls[i] in tar_filenames:
        weak_train_choose_filename.append(weak_file_name_ls[i])
        weak_train_choose_labels.append(weak_labels_ls[i])
    else:
        weak_validate_choose_filename.append(weak_file_name_ls[i])
        weak_validate_choose_labels.append(weak_labels_ls[i])

print(len(weak_train_choose_filename))
print(len(weak_validate_choose_filename))
train_dict = {'filename': weak_train_choose_filename, 'event_labels': weak_train_choose_labels}
df = pd.DataFrame(train_dict)
df.to_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/filelist/weak_train_choose.tsv',index=False,sep='\t')


validate_dict = {'filename': weak_validate_choose_filename, 'event_labels': weak_validate_choose_labels}
df = pd.DataFrame(validate_dict)
df.to_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/filelist/weak_validate_choose.tsv',index=False,sep='\t')

