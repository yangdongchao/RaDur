# we will get all the reference class, and it coressponding number, return a txt file
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

ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/reference/audio4')
reference_dict = {}
for name in ls:
    tmp_name = name[:-5]
    if tmp_name not in reference_dict.keys():
        reference_dict[tmp_name] = 1
    else:
        reference_dict[tmp_name] += 1

reference_txt = './reference.txt'
f=open(reference_txt,'w')
for key in  reference_dict.keys():
    f.write(key + '\t' + str(reference_dict[key])+'\n')
