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
import os
# delete these audio file, which not include our predifined class
train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/weak_eval_final.tsv',sep='\t',usecols=[0,1])
#eval_tsv = pd.read_csv('./strong_label/audioset_train_strong.tsv',sep='\t',usecols=[0,1,2,3])
labels = train_tsv['event_labels']
filename = train_tsv['filename']

labels_ls = []
filenames_ls = []
for i in labels:
    labels_ls.append(i)
for i in filename:
    filenames_ls.append(i)

fw = open('./reference.txt','r')
class_dict = []
for line in fw:
    line_ls = line.split('\t')
    class_name = line_ls[0]
    class_dict.append(class_name)
    # print(class_name)
    # assert 1==2
empty_file = []
for i in range(len(filenames_ls)):
    labels_split = labels_ls[i].split(',')
    # print(labels_split)
    # assert 1==2
    flag = 0
    for label in labels_split:
        if label in class_dict:
            flag=1
    if flag == 0:
        empty_file.append(filenames_ls[i])

print(len(empty_file))
print(empty_file)
strong_train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_eval_final.tsv',sep='\t',usecols=[0,1,2,3])
labels_strong = strong_train_tsv['event_label']
segment_ids = strong_train_tsv['filename']
start_time_seconds = strong_train_tsv['onset']
end_time_seconds = strong_train_tsv['offset']
trans_labels = []
filename_strong = []
onset = []
offset = []
# print(labels)
for i in labels_strong:
    #print(i
    trans_labels.append(i)
for i in segment_ids:
    filename_strong.append(i)
for i in start_time_seconds:
    onset.append(i)
for i in end_time_seconds:
    offset.append(i)

new_trans_labels = []
new_filename_strong = []
new_onset = []
new_offset = []
print(len(trans_labels))
for i in range(len(trans_labels)):
    if filename_strong[i] not in empty_file:
        new_filename_strong.append(filename_strong[i])
        new_trans_labels.append(trans_labels[i])
        new_onset.append(onset[i])
        new_offset.append(offset[i])

dict = {'filename': new_filename_strong, 'onset': new_onset, 'offset': new_offset, 'event_label': new_trans_labels}
df = pd.DataFrame(dict)
df.to_csv('strong_eval_now_class.tsv',index=False,sep='\t')
