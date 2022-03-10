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
event_labels = ['Alarm', 'Alarm_clock', 'Animal', 'Applause', 'Arrow', 'Artillery_fire', 
                'Babbling', 'Baby_laughter', 'Bark', 'Basketball_bounce', 'Battle_cry', 
                'Bell', 'Bird', 'Bleat', 'Bouncing', 'Breathing', 'Buzz', 'Camera', 
                'Cap_gun', 'Car', 'Car_alarm', 'Cat', 'Caw', 'Cheering', 'Child_singing', 
                'Choir', 'Chop', 'Chopping_(food)', 'Clapping', 'Clickety-clack', 'Clicking', 
                'Clip-clop', 'Cluck', 'Coin_(dropping)', 'Computer_keyboard', 'Conversation', 
                'Coo', 'Cough', 'Cowbell', 'Creak', 'Cricket', 'Croak', 'Crow', 'Crowd', 'DTMF', 
                'Dog', 'Door', 'Drill', 'Drip', 'Engine', 'Engine_starting', 'Explosion', 'Fart', 
                'Female_singing', 'Filing_(rasp)', 'Finger_snapping', 'Fire', 'Fire_alarm', 'Firecracker', 
                'Fireworks', 'Frog', 'Gasp', 'Gears', 'Giggle', 'Glass', 'Glass_shatter', 'Gobble', 'Groan', 
                'Growling', 'Hammer', 'Hands', 'Hiccup', 'Honk', 'Hoot', 'Howl', 'Human_sounds', 'Human_voice', 
                'Insect', 'Laughter', 'Liquid', 'Machine_gun', 'Male_singing', 'Mechanisms', 'Meow', 'Moo', 
                'Motorcycle', 'Mouse', 'Music', 'Oink', 'Owl', 'Pant', 'Pant_(dog)', 'Patter', 'Pig', 'Plop',
                'Pour', 'Power_tool', 'Purr', 'Quack', 'Radio', 'Rain_on_surface', 'Rapping', 'Rattle', 
                'Reversing_beeps', 'Ringtone', 'Roar', 'Run', 'Rustle', 'Scissors', 'Scrape', 'Scratch', 
                'Screaming', 'Sewing_machine', 'Shout', 'Shuffle', 'Shuffling_cards', 'Singing', 
                'Single-lens_reflex_camera', 'Siren', 'Skateboard', 'Sniff', 'Snoring', 'Speech', 
                'Speech_synthesizer', 'Spray', 'Squeak', 'Squeal', 'Steam', 'Stir', 'Surface_contact', 
                'Tap', 'Tap_dance', 'Telephone_bell_ringing', 'Television', 'Tick', 'Tick-tock', 'Tools', 
                'Train', 'Train_horn', 'Train_wheels_squealing', 'Truck', 'Turkey', 'Typewriter', 'Typing', 
                'Vehicle', 'Video_game_sound', 'Water', 'Whimper_(dog)', 'Whip', 'Whispering', 'Whistle', 
                'Whistling', 'Whoop', 'Wind', 'Writing', 'Yip', 'and_pans', 'bird_song', 'bleep', 'clink', 
                'cock-a-doodle-doo', 'crinkling', 'dove', 'dribble', 'eructation', 'faucet', 'flapping_wings', 
                'footsteps', 'gunfire', 'heartbeat', 'infant_cry', 'kid_speaking', 'man_speaking', 'mastication', 
                'mice', 'river', 'rooster', 'silverware', 'skidding', 'smack', 'sobbing', 'speedboat', 'splatter',
                'surf', 'thud', 'thwack', 'toot', 'truck_horn', 'tweet', 'vroom', 'waterfowl', 'woman_speaking']

train_strong = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/strong_train_final.tsv',sep='\t',usecols=[0,1,2,3])
labels = train_strong['event_label']
segment_ids = train_strong['filename']
start_time_seconds = train_strong['onset']
end_time_seconds = train_strong['offset']
trans_labels = []
filename = []
onset = []
offset = []
for i in labels:
    #print(i
    trans_labels.append(i)
for i in segment_ids:
    filename.append(i)
for i in start_time_seconds:
    onset.append(i)
for i in end_time_seconds:
    offset.append(i)

filename_dict = {} # we can lookup the sequence_id according to filename
for i, f_name in enumerate(filename):
    if f_name not in filename_dict.keys():
        filename_dict[f_name] = [i]
    else:
        filename_dict[f_name].append(i)

def get_wav_make(dataDir,begin,end,save_name):
    sound = AudioSegment.from_wav(dataDir)
    duration = sound.duration_seconds * 1000
    end = min(end,duration)
    cut_wav = sound[begin:end]
    cut_wav.export(save_name,format='wav')

def get_wav_make_ls(dataDir,begins,ends,save_name):
    sound = AudioSegment.from_wav(dataDir)
    duration = sound.duration_seconds * 1000
    start = max(begins[0],0)
    end = min(ends[0],duration)
    cut_wav = sound[start:end] 
    for i in range(1,len(begins)):
        start = max(begins[i],0)
        end = min(ends[i],duration)
        cut_wav += sound[start:end]
    cut_wav.export(save_name,format='wav')

def check(st,ed,ls_t,ls_e):
    flag = -1
    index_i = -1
    for i in range(len(ls_t)):
        if st >= ls_t[i] and st <= ls_e[i] and ed >= ls_t[i] and ed <= ls_e[i]:
            # 完全在里面
            flag = 0
            break
        elif st >= ls_t[i] and st <= ls_e[i] and ed > ls_e[i]:
            # 部分包括
            flag = 1
            index_i = i
            break
        elif ed >= ls_t[i] and ed <= ls_e[i] and st < ls_t[i]:
            # 部分包括
            flag = 2
            index_i = i
            break
        elif st < ls_t[i] and ed > ls_e[i]:
            # 外包括
            flag = 3
            index_i = i
            break
        else:
            #不包括
            pass
    if flag == 1:
        ls_e[index_i] = ed
    elif flag == 2:
        ls_t[index_i] = st
    elif flag == 3:
        ls_t[index_i] = st
        ls_e[index_i] = ed
    return ls_t,ls_e

def check_overlap():
    pass
# ps_train = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_label/clips/train.txt'
# segment_ids_to_real_file = './segment_ids_to_real_file.txt'
# f=open(ps_train,'r')
# fw = open(segment_ids_to_real_file,'w')
# seg_dict = {}
# for line  in f:
#     line_ls = line.split('\t')
#     if line_ls[1] not in seg_dict.keys():
#         seg_dict[line_ls[1]] = line_ls[2]
# la,lb = check(1,4,[2,9],[8,10])
# print(la,lb)
# assert 1==2
txt_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/txt'
txt_ls = os.listdir(txt_path)
for ps in txt_ls:
    target_class = ps[:-4]
    # print(target_class)
    ps = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/txt/' + ps
    # print(ps)
    # assert 1==2
    f=open(ps,'r')
    txt=[]
    for line in f:
        txt.append(line.strip())
    for t in txt:
        sequence_ids = filename_dict[t] # 该文件所处的位置
        st_times = []
        ed_times = []
        tot_time = 0
        for seq_id in sequence_ids:
            if trans_labels[seq_id] == target_class:
                on_t = onset[seq_id]*1000
                off_t = offset[seq_id]*1000
                flag = 0
                for s_id in sequence_ids: # check overlap
                    if s_id == seq_id:
                        continue
                    if trans_labels[s_id] != target_class:
                        tmp_on = onset[s_id]*1000
                        tmp_off = offset[s_id]*1000
                        if (tmp_on >= on_t and tmp_on <= off_t) or (tmp_off >= on_t and tmp_off <= off_t):
                            flag += 1
                if flag == 0: # on overlap with others
                    if len(st_times) == 0:
                        st_times.append(on_t)
                        ed_times.append(off_t)
                        tot_time += off_t-on_t
                    else:
                        st_times,ed_times = check(on_t,off_t,st_times,ed_times)

                # else:
                #     pass
                    # save_name = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/reference/audio2/' + target_class+'_'+str(int(off_t-on_t))+'_'+str(flag)+'.wav'
                    # pre_audio_path = '/apdcephfs/share_1316500/donchaoyang/code2/target_sound_event_detection/data/strong_label/train/' + t
                    # get_wav_make(pre_audio_path,on_t,off_t,save_name)
        # flag = 0
        #提取出的audio命名方式 原名_类别_时长_重叠声音数量
        if len(st_times) == 0:
            print('t ',t,target_class)
            print('search again')
            # search again
            for seq_id in sequence_ids:
                if trans_labels[seq_id] == target_class:
                    on_t = onset[seq_id]*1000
                    off_t = offset[seq_id]*1000
                    flag = 0
                    for s_id in sequence_ids: # check overlap
                        if s_id == seq_id:
                            continue
                        if trans_labels[s_id] != target_class:
                            tmp_on = onset[s_id]*1000
                            tmp_off = offset[s_id]*1000
                            if (tmp_on >= on_t and tmp_on <= off_t) or (tmp_off >= on_t and tmp_off <= off_t):
                                if trans_labels[s_id] in event_labels: # we releax the limit
                                    flag += 1
                    if flag == 0: # on overlap with others
                        if len(st_times) == 0:
                            st_times.append(on_t)
                            ed_times.append(off_t)
                            tot_time += off_t-on_t
                        else:
                            st_times,ed_times = check(on_t,off_t,st_times,ed_times)
        # else:
        #     if tot_time > 10000:
        #         print(t)
        #         print('sequence_ids ',sequence_ids)
        #         assert 1==2
        if len(st_times) == 0:
            print('this class is hard to get ',t,target_class)
        else:    
            save_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio2/' + target_class+'_'+str(int(tot_time))+'.wav' # '_'+str(flag)
            pre_audio_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/strong_label/train/' + t
            get_wav_make_ls(pre_audio_path,st_times,ed_times,save_name)
        #assert 1==2