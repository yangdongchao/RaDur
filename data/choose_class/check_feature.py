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
# # y = hf['label'][:].astype(np.float32)
# events = np.array([target_event.decode() for target_event in hf['events'][:]])
# filename = np.array([filename.decode() for filename in hf['filename'][:]])
# y = hf['label'][:].astype(np.float32)
# for i,name in enumerate(filename):
#     if name == 'YH0e-Qi0yUQw.wav':
#         print(y[i])
#         print(events[i])

ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/code2/data/reference/txt')
print(len(ls))
