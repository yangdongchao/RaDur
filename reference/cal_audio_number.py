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

# we will reshape this audio, if a audio duration less than 5s, we will cat it with others
def merge_audio(path_ls,save_name):
    sound = AudioSegment.from_wav(path_ls[0])
    #duration = sound.duration_seconds * 1000
    for i in range(1,len(path_ls)):
        sound_tmp = AudioSegment.from_wav(path_ls[i])
        sound += sound_tmp
    sound.export(save_name,format='wav')


ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio2')
dict_audio = {}
for name in ls:
    tmp_name = name[:-4]
    score = int(tmp_name.split('_')[-1])
    tmp_index = len(tmp_name)-1
    while tmp_index >0 and tmp_name[tmp_index] != '_':
        tmp_index -= 1
    tmp_name = tmp_name[:tmp_index]
    if tmp_name not in dict_audio.keys():
        dict_audio[tmp_name] = [name]
    else:
        dict_audio[tmp_name].append(name)


for key in dict_audio.keys():
    score_dict = {}
    for name in dict_audio[key]:
        tmp_name = name[:-4]
        score = int(tmp_name.split('_')[-1])
        # print(score)
        if score < 5000:
            score_dict[name] = score
        else:
            save_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio3/' + name
            pre_audio_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio2/' + name
            merge_audio([pre_audio_path],save_name)
    if len(score_dict.keys()) ==0:
        continue
    # print(score_dict)
    new_sys = sorted(score_dict.items(),  key=lambda d: d[1], reverse=False)
    # print(new_sys)
    # assert 1==2
    pre = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio2/'
    i = 0 
    while i < len(new_sys):
        # print('i ',i)
        path_ls = []
        s_tmp = new_sys[i][1]
        path_ls.append(pre+new_sys[i][0])
        k = i+1
        while (s_tmp < 10000) and (k < len(new_sys)):
            s_tmp += new_sys[k][1]
            path_ls.append(pre+new_sys[k][0])
            k += 1
        # print('k ',k)
        i = k
        # print(path_ls)
        # assert 1==2
        if s_tmp < 1000:
            continue
        save_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio3/' + key + '_' + str(int(s_tmp))+'.wav'
        merge_audio(path_ls,save_name)
    #assert 1==2

        




        

# print(score_dict)

        




# Max = 0
# Min = 1000000
# Min_class = None
# Max_class = None
# avg = 0
# for name,score in dict_audio.items():
#     if score < Min:
#         Min = score
#         Min_class = name
#     if score > Max:
#         Max = score
#         Max_class = name
#     avg += score
# print(Max_class,Max)
# print(Min_class,Min)
# print(avg/len(dict_audio.keys()))
    # print(tmp_name)
    # assert 1==2