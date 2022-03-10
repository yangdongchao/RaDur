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

train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/weak_eval_choose.tsv',sep='\t',usecols=[0,1])
# labels = train_tsv['event_label']
segment_ids = train_tsv['filename']
# start_time_seconds = train_tsv['onset']
# end_time_seconds = train_tsv['offset']
trans_labels = []
filename = []
onset = []
offset = []
for i in segment_ids:
    filename.append(i)

new_filename = []
new_duration = []
for i in range(len(filename)):
    new_filename.append(filename[i])
    new_duration.append(10.0)

dict = {'filename': new_filename, 'duration': new_duration}
df = pd.DataFrame(dict)
df.to_csv('mata_eval_choose.tsv',index=False,sep='\t')