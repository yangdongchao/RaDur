import argparse
import librosa
from tqdm import tqdm
import io
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import gzip
import h5py
train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/strong_train_choose.tsv',sep='\t',usecols=[0,1,2,3])
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

weak_filename = []
weak_labels = []
i=0
while i < len(filename):
    pre = filename[i]
    tmp_label = []
    tmp_label.append(trans_labels[i])
    while i < len(filename) and filename[i] == pre:
        if trans_labels[i] not in tmp_label:
            tmp_label.append(trans_labels[i])
        i = i+1
    weak_filename.append(pre)
    event_labels = tmp_label[0]
    for j in range(1,len(tmp_label)):
        event_labels = event_labels +','+tmp_label[j]
    weak_labels.append(event_labels)
    # print(weak_filename)
    # print(weak_labels)
    # print(i)
    # # assert 1==2
    # if i > 100:
    #     break

dict = {'filename': weak_filename, 'event_labels': weak_labels}
df = pd.DataFrame(dict)
df.to_csv('weak_train_choose.tsv',index=False,sep='\t')


