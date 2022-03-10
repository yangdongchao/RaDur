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
# you can set which class is you need in this file
exclude_class = ['Background_noise','Medium_engine_(mid_frequency)',
                 'Heavy_engine_(low_frequency)','Motor_vehicle_(road)',
                 'Generic_impact_sounds','Noise','Sound_effect',
                 'Synthetic_singing','Wind_noise_(microphone)', 'roadway_noise','speech_babble']
train_number = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/class_number_in_train.tsv',sep='\t',usecols=[0,1])
test_number = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/class_number_in_eval.tsv',sep='\t',usecols=[0,1])
class_names_train = train_number['class_names']
numbers_train = train_number['numbers']
numbers_test = test_number['numbers']
#print(labels)
final_class = []
class_names = []
train_num = []
test_num = []
for i in class_names_train:
    #print(i)
    class_names.append(i)
for i in numbers_train:
    train_num.append(i)

for i in numbers_test:
    test_num.append(i)

for i in range(len(class_names)):
    if train_num[i] > 500 and test_num[i] > 50 and class_names[i] not in exclude_class:
        final_class.append(class_names[i])
    
print(len(final_class)) # 192
print(final_class) # 192
# assert 1==2
# delete some not including our expect class
train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/reference/strong_eval_now_class.tsv',sep='\t',usecols=[0,1,2,3])
labels = train_tsv['event_label']
segment_ids = train_tsv['filename']
start_time_seconds = train_tsv['onset']
end_time_seconds = train_tsv['offset']
trans_labels = []
filename = []
onset = []
offset = []
#print(labels)
for i in labels:
    #print(i
    trans_labels.append(i)
for i in segment_ids:
    filename.append(i)
for i in start_time_seconds:
    onset.append(i)
for i in end_time_seconds:
    offset.append(i)

new_trans_labels = []
new_filename = []
new_onset = []
new_offset = []

for i in range(len(filename)):
    if trans_labels[i] not in final_class:
        continue
    new_filename.append(filename[i])
    new_trans_labels.append(trans_labels[i])
    new_onset.append(onset[i])
    new_offset.append(offset[i])

dict = {'filename': new_filename, 'onset': new_onset, 'offset': new_offset, 'event_label': new_trans_labels}
df = pd.DataFrame(dict)
df.to_csv('strong_eval_choose.tsv',index=False,sep='\t')

