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

event_duration_dict = {}
event_num_dict = {}
for i in range(len(filename)):
    if trans_labels[i] not in event_duration_dict.keys():
        event_duration_dict[trans_labels[i]] = offset[i] - onset[i]
        event_num_dict[trans_labels[i]] = 1
    else:
        event_duration_dict[trans_labels[i]] += offset[i] - onset[i]
        event_num_dict[trans_labels[i]] += 1

weak_filename = []
weak_labels = []

# for key in event_duration_dict.keys():
#     event_duration_dict[key] = 1.0*event_duration_dict[key]/event_num_dict[key]
    # weak_filename.append(key)
    # weak_labels.append(event_duration_dict[key])

new_sys1 = sorted(event_duration_dict.items(), key=lambda d: d[0], reverse=False)
print(new_sys1)
for entity in new_sys1:
    weak_filename.append(entity[0])
    weak_labels.append(entity[1])
dict = {'event_label': weak_filename, 'time': weak_labels}
df = pd.DataFrame(dict)
df.to_csv('train_time_total.tsv',index=False,sep='\t')