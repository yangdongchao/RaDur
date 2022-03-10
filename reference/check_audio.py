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
# 平均时长 8.1s
def get_wav_make(dataDir,save_name):
    sound = AudioSegment.from_wav(dataDir)
    duration = sound.duration_seconds * 1000
    if duration < 10000:
        if duration*2 >= 10000:
            sound = sound + sound[:10000-duration]
        elif duration*3 >= 10000:
            sound = sound +sound +sound[:10000-2*duration]
        elif duration*4 >= 10000:
            sound = sound +sound + sound + sound[:10000-3*duration]
        else:
            assert 'error'
    elif duration > 10000:
        sound = sound[:10000]
    # end = min(end,duration)
    cut_wav = sound[:10000]
    cut_wav.export(save_name,format='wav')
# check and padding
ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio3')
pre = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio3/'
save = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio4/'
reference_dict = {}
for audio in ls:
    # print(audio)
    tmp_name = audio[:-4]
    #score = int(tmp_name.split('_')[-1])
    tmp_index = len(tmp_name)-1
    while tmp_index >0 and tmp_name[tmp_index] != '_':
        tmp_index -= 1
    tmp_name = tmp_name[:tmp_index]
    if tmp_name not in reference_dict.keys():
        reference_dict[tmp_name] = 1
    else:
        reference_dict[tmp_name] += 1
    dataDir_ = pre+audio
    save_name_ = save + tmp_name + str(reference_dict[tmp_name]) + '.wav'
    # print(save_name_)
    # assert 1==2
    get_wav_make(dataDir_,save_name_)

# dict_audio = {}
# tot_duration = 0
# for name in ls:
#     tmp_name = name[:-4]
#     score = int(tmp_name.split('_')[-1])
#     tot_duration += score
#     index_i = len(name) - 1
#     while index_i > 0 and name[index_i] != '_':
#         index_i -=1
#     tmp_name = name[:index_i]
#     if tmp_name not in dict_audio.keys():
#         dict_audio[tmp_name] = score
#     else:
#         dict_audio[tmp_name] = max(score,dict_audio[tmp_name])
    # print(tmp_name)
    # assert 1==2
# for key,value in dict_audio.items():
#     tot_duration += value
# print(tot_duration/len(dict_audio.keys()))
# print(len(dict_audio.keys()))
# print(dict_audio)
