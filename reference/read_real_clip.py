# import os
# filePath = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_label/clips/train'
# ls = os.listdir(filePath)
# for l in ls:
#     print(l)
#     assert 1==2
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
from pydub import AudioSegment
ps = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_label/clips/train.txt'
segment_ids_to_real_file = './segment_ids_to_real_file.txt'
f=open(ps,'r')
fw = open(segment_ids_to_real_file,'w')
seg_dict = {}
for line  in f:
    line_ls = line.split('\t')
    if line_ls[1] not in seg_dict.keys():
        seg_dict[line_ls[1]] = line_ls[2]

for a, b in seg_dict.items():
    fw.write(a+'\t'+b+'\n')
