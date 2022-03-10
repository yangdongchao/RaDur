#!/usr/bin/env python3
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
def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
def get_valid_class_dict():
    fw = open('/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/reference/reference.txt','r')
    class_dict = []
    for line in fw:
        line_ls = line.split('\t')
        class_name = line_ls[0]
        class_dict.append(class_name)
    return class_dict

what_type = 'train'
weak_csv = '/apdcephfs/share_1316500/donchaoyang/code2/data/filelist/weak_train_choose_split.tsv'
DF_weak = pd.read_csv(weak_csv, sep='\t',usecols=[0,1])  # only read first cols, allows to have messy csv

strong_csv = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/strong_train_choose.tsv'

print('weak_csv ',weak_csv)
print('strong_csv ',strong_csv)
DF_strong = pd.read_csv(strong_csv,sep='\t',usecols=[0,1,2,3])

strong_file_dict = {}
for i, filename in enumerate(DF_strong['filename']):
    if filename not in strong_file_dict.keys():
        strong_file_dict[filename] = [i]
    else:
        strong_file_dict[filename].append(i)

# frames_num = 501
# num_freq_bin = 64
h5_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/'  + 'train_no_choose.h5'
print(h5_name)
hf = h5py.File(h5_name, 'w')
num_ = 100
hf.create_dataset(
    name='filename', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S80')
hf.create_dataset(
    name='events', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S900')
hf.create_dataset(
    name='label',
    shape=(0,num_,2),
    maxshape=(None,num_,2),
    dtype=np.float32)
weak_filename = DF_weak['filename']
weak_label = DF_weak['event_labels']
n=0
# valida_class = get_valid_class_dict()
for i,filename in enumerate(weak_filename):
    basename = Path(filename).name
    print(i,basename)
    # strong_name = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_label/eval/'+basename # validate/
    # # print(strong_name)
    # fname, lms_feature = extract_feature(strong_name)
    # print(fname,lms_feature.shape)
    # print(i,filename)
    event_labels = weak_label[i]
    ls_event = event_labels.split(',')
    hf['filename'].resize((n+1,))
    hf['filename'][n] = basename.encode()
    time_label = []
    events = []
    for index in strong_file_dict[basename]:
        st = DF_strong['onset'][index]
        ed = DF_strong['offset'][index]
        st = float(st)
        ed = float(ed)
        tmp = [st,ed]
        time_label.append(tmp)
        events.append(DF_strong['event_label'][index])
    
    while len(time_label) < num_:
        time_label.append([-1,-1])
        # events.append('None')
    assert len(time_label) == num_
    save_events = events[0]
    for i in range(1,len(events)):
        save_events = save_events+','+events[i]
    # print(save_events)
    # print(time_label)
    # assert 1==2
    # print(basename)
    # print(save_events)
    # print(time_label)
    # assert 1==2
    hf['events'].resize((n+1,))
    hf['events'][n] = save_events.encode()
    hf['label'].resize((n+1,num_,2))
    time_label = np.array(time_label)
    hf['label'][n] = time_label
    n += 1

 

print(n)

