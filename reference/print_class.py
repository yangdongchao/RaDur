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
segment_ids_to_real_file = './reference.txt'
f=open(segment_ids_to_real_file,'r')
class_ls = []
for line  in f:
    line_ls = line.split('\t')
    class_ls.append(line_ls[0])
print(class_ls)
