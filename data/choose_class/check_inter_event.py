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
import sys  # 导入sys模块
#sys.setrecursionlimit(6000)  # 将默认的递归深度修改为3000
train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/generate_tsd/strong_eval_add_2_neg.tsv',sep='\t',usecols=[0,1,2,3])
labels = train_tsv['event_label']
segment_ids = train_tsv['filename']
start_time_seconds = train_tsv['onset']
end_time_seconds = train_tsv['offset']
trans_labels = []
filename = []
onset = []
offset = []
print(labels)
for i in labels:
    #print(i
    trans_labels.append(i)
for i in segment_ids:
    filename.append(i)
for i in start_time_seconds:
    onset.append(i)
for i in end_time_seconds:
    offset.append(i)

# print('len ',len(filename))
# assert 1==2
new_filename = []
new_onset = []
new_offset = []
new_event = []
# k = len(filename)-1

def dfs(k,fn,on,off,lab):
    print(k)
    if k >= len(filename)-1:
        new_filename.append(fn)
        new_onset.append(on)
        new_offset.append(off)
        new_event.append(lab)
        return
    if filename[k] == fn and trans_labels[k] == lab:
        if onset[k] <= off:
            on = min(on,onset[k])
            off = max(off,offset[k])
            dfs(k+1,fn,on,off,lab)
        else:
            new_filename.append(fn)
            new_onset.append(on)
            new_offset.append(off)
            new_event.append(lab)
            dfs(k+1,filename[k],onset[k],offset[k],trans_labels[k])
    else:
        new_filename.append(fn)
        new_onset.append(on)
        new_offset.append(off)
        new_event.append(lab)
        dfs(k+1,filename[k],onset[k],offset[k],trans_labels[k])

def inter(pre_onset,pre_offset,onset,offset):
    if onset >= pre_onset and onset <= pre_offset:
        return True
    if offset >= pre_onset and offset <= pre_offset:
        return True
    return False
# dfs(1,filename[0],onset[0],offset[0],trans_labels[0])
k = 1
fn = filename[0] 
# on = onset[0] 
# off = offset[0] 
# lab = trans_labels[0]
ls = []
ls.append(0)
while k < len(filename):
    print(ls)
    while k < len(filename) and filename[k] == filename[ls[0]]:
        ls.append(k)
        k += 1
    dl = []
    for i in range(len(ls)):
        if ls[i] in dl:
            continue
        fn = filename[ls[i]] 
        on = onset[ls[i]] 
        off = offset[ls[i]] 
        lab = trans_labels[ls[i]]
        for j in range(i+1,len(ls)):
            if trans_labels[ls[j]] == lab and inter(on,off,onset[ls[j]],offset[ls[j]]) == 1:
                on = min(on,onset[ls[j]])
                off = max(off,offset[ls[j]])
                dl.append(ls[j])
        new_filename.append(fn)
        new_onset.append(on)
        new_offset.append(off)
        new_event.append(lab)
    if k >= len(filename):
        break
    ls = []
    fn = filename[k]
    ls.append(k)
    k += 1


dict = {'filename': new_filename, 'onset': new_onset, 'offset': new_offset, 'event_label': new_event}
df = pd.DataFrame(dict)
df.to_csv('strong_and_weakly_eval_psds_2_neg.tsv',index=False,sep='\t')
