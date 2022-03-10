# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang 
# @File    : train.py
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata
import os
import h5py
import torchaudio
import random
import math
import pickle
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

event_to_id = {label : i for i, label in enumerate(event_labels)}
id_to_event = {i: label for i,label in enumerate(event_labels)}
event_to_time ={}
event_time_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/train_time.tsv',sep='\t',usecols=[0,1])
labels = event_time_tsv['event_label']
times = event_time_tsv['time']
trans_labels = []
times_ls = []
for i in labels:
    trans_labels.append(i)
for i in times:
    times_ls.append(i)
for i in range(len(trans_labels)):
    if trans_labels[i] not in event_to_time.keys():
        event_to_time[trans_labels[i]] = times_ls[i]

def read_spk_emb_file(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_lab_dict = {}
    spk_emb_dict = {}
    with open(spk_emb_file_path, 'r') as file:
        for line in file:
            temp_line = line.strip().split('\t')
            file_id = os.path.basename(temp_line[0]) # get filename
            emb = np.array(temp_line[1].split(' ')).astype(np.float) # embedding
            spk_lab = file_id[:-5] # 
            # spk_id = event_to_id[spk_id]
            spk_emb_dict[file_id] = emb
            if spk_lab in spk_lab_dict:
                spk_lab_dict[spk_lab].append(file_id)
            else:
                spk_lab_dict[spk_lab] = [file_id]
        # print(len(spk_lab_dict.keys()))
        # assert 1==2
    return spk_emb_dict, spk_lab_dict

def read_spk_emb_file_by_h5(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_id_dict = {}
    spk_emb_dict = {}
    mel_mfcc = h5py.File(spk_emb_file_path, 'r') # libver='latest', swmr=True
    file_name = np.array([filename.decode() for filename in mel_mfcc['filename'][:]])
    file_path = np.array([file_path.decode() for file_path in mel_mfcc['file_path'][:]])
    for i in range(file_name.shape[0]):
        file_id = file_name[i]
        emb = file_path[i]
        spk_id = int(file_id.split('-')[1])
        spk_id_label = id_to_event[spk_id]
        spk_emb_dict[file_id] = emb
        if spk_id_label in spk_id_dict:
            spk_id_dict[spk_id_label].append(file_id)
        else:
            spk_id_dict[spk_id_label] = [file_id]
    
    return spk_emb_dict,spk_id_dict
def time_to_frame(tim,time_resolution):
    radio = 10.0/time_resolution
    return int(tim/radio)  # 10/125
class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        #print(fname)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/' + fname[:-4]+'.txt'
        data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(125) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event

class HDF5Dataset_32000(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None,time_resolution=125):
        super(HDF5Dataset_32000, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.time_resolution = time_resolution
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(self.time_resolution) # 501 --> pooling 一次 到 250
        # assert 1==2
        slack_time = np.zeros(self.time_resolution)
        # if event_to_time[target_event] < 100:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/10.0))
        # elif event_to_time[target_event] > 100 and event_to_time[target_event] < 1000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/100.0))/9.0
        # elif event_to_time[target_event] > 1000 and event_to_time[target_event] < 10000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/1000.0))/81.0
        # elif event_to_time[target_event] > 10000 and event_to_time[target_event] < 100000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/10000.0))/(81.0*9)
        # else:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/100000.0))/(81.0*9*9)
        slack_time[0:self.time_resolution] = float(event_to_time[target_event]) # mean time
        # slack_time[0:self.time_resolution] = 0
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            # print(time[i,0],time[i,1])
            start = time_to_frame(time[i,0],self.time_resolution)
            end = min(self.time_resolution,time_to_frame(time[i,1],self.time_resolution))
            # print(start,end)
            # assert 1==2
            frame_level_label[start:end] = 1
        
        # print('frame_level_time ',frame_level_time)
        # print('frame_level_label ',frame_level_label)
        # assert 1==2
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        slack_time = torch.as_tensor(slack_time).float()
        embedding = torch.as_tensor(embedding).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, slack_time, time, embedding, fname, target_event


class HDF5Dataset_32000_pkl(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None,time_resolution=125):
        super(HDF5Dataset_32000_pkl, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.time_resolution = time_resolution
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # print(data)
        # data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(self.time_resolution) # 501 --> pooling 一次 到 250
        # assert 1==2
        slack_time = np.zeros(self.time_resolution)
        # if event_to_time[target_event] < 100:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/10.0))
        # elif event_to_time[target_event] > 100 and event_to_time[target_event] < 1000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/100.0))/9.0
        # elif event_to_time[target_event] > 1000 and event_to_time[target_event] < 10000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/1000.0))/81.0
        # elif event_to_time[target_event] > 10000 and event_to_time[target_event] < 100000:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/10000.0))/(81.0*9)
        # else:
        #     slack_time[0:125] = math.exp(-0.25*(event_to_time[target_event]/100000.0))/(81.0*9*9)
        slack_time[0:self.time_resolution] = float(event_to_time[target_event]) # mean time
        # slack_time[0:self.time_resolution] = 0
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            # print(time[i,0],time[i,1])
            start = time_to_frame(time[i,0],self.time_resolution)
            end = min(self.time_resolution,time_to_frame(time[i,1],self.time_resolution))
            # print(start,end)
            # assert 1==2
            frame_level_label[start:end] = 1
        
        # print('frame_level_time ',frame_level_time)
        # print('frame_level_label ',frame_level_label)
        # assert 1==2
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        slack_time = torch.as_tensor(slack_time).float()
        embedding = torch.as_tensor(embedding).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, slack_time, time, embedding, fname, target_event


class HDF5Dataset_strong_sed(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 transform=None):
        super(HDF5Dataset_strong_sed, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with h5py.File(self._h5file, 'r') as hf:
            self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['label'][:].astype(np.float32)
            self.events = np.array([target_event.decode() for target_event in hf['events'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        data = self.X[index]
        fname = self.filename[index]
        time = self.y[index]
        event = self.events[index]
        event_ls = event.split(',')
        if len(event_ls) == 0:
            event_ls.append(event) 
        # print(len(event_ls))
        # print(time.shape)
        # print(time[len(event_ls)])
        frame_level_label = np.zeros((125,len(event_labels))) # 501 --> pooling 2次 到 125
        for i in range(len(event_ls)):
            if event_ls[i] not in event_labels:
                continue
            class_id = event_to_id[event_ls[i]]
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end,class_id] = 1
        
        # print(time)
        # print(event)
        # assert time[len(event_ls),0] == -1 and time[len(event_ls),1] == -1

        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, fname

class HDF5Dataset_strong_sed_txt(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 transform=None):
        super(HDF5Dataset_strong_sed_txt, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['label'][:].astype(np.float32)
            self.events = np.array([target_event.decode() for target_event in hf['events'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/' + fname[:-4]+'.txt'
        data = np.loadtxt(mel_path)
        time = self.y[index]
        event = self.events[index]
        event_ls = event.split(',')
        if len(event_ls) == 0:
            event_ls.append(event) 
        # print(len(event_ls))
        # print(time.shape)
        # print(time[len(event_ls)])
        frame_level_label = np.zeros((125,len(event_labels))) # 501 --> pooling 2次 到 125
        for i in range(len(event_ls)):
            if event_ls[i] not in event_labels:
                continue
            class_id = event_to_id[event_ls[i]]
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end,class_id] = 1
        
        # print(time)
        # print(event)
        # assert time[len(event_ls),0] == -1 and time[len(event_ls),1] == -1

        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, fname


class HDF5Dataset_join(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """ 
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None,time_resolution=None):
        super(HDF5Dataset_join, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.time_resolution = time_resolution
        self.spk_emb_dict, self.spk_id_dict = read_spk_emb_file(self._embedfile) # 
        self.embed_mel_file = self._embedfile[:-4]  + '.h5'
        self.embeddings_mel = h5py.File(self.embed_mel_file,'r',libver='latest', swmr=True)
        #self.X = h5py.File('/apdcephfs/share_1316500/donchaoyang/code2/data/feature_32000/merge.h5','r',libver='latest', swmr=True)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len
    
    def get_fea(self,path_):
        waveform, sr = torchaudio.load(path_)
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        output = audio_mono.numpy()
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        new_feat = new_feat.permute(0, 2, 1).squeeze()
        return new_feat

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        data = np.loadtxt(mel_path)
        #data = self.X[fname][()]
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_id_dict[target_event], 1)
        embedding = np.loadtxt('/apdcephfs/share_1316500/donchaoyang/code2/data/embeddings/mel_txt2/'+embed_file_list[0][:-4]+'.txt')
        # embedding = self.spk_emb_dict[embed_file_list[0]]
        embed_label = np.zeros(10)
        # embed_label_index = int(embed_file_list[0].split('-')[1])
        # print('embedding ',embedding.shape)
        # embed_label[embed_label_index] = 1
        # frame_level_label = np.zeros((125,len(event_labels))) # 501 --> pooling 2次 到 125
        # for i in range(len(event_ls)):
        #     if event_ls[i] not in event_labels:
        #         continue
        #     class_id = event_to_id[event_ls[i]]
        #     start = time_to_frame(time[i,0])
        #     end = min(125,time_to_frame(time[i,1]))
        #     frame_level_label[start:end,class_id] = 1
        
        frame_level_label = np.zeros(self.time_resolution) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0],self.time_resolution)
            end = min(125,time_to_frame(time[i,1],self.time_resolution))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        embed_label = torch.as_tensor(embed_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding,embed_label, fname, target_event

class HDF5Dataset_join_pkl(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """ 
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None,time_resolution=None):
        super(HDF5Dataset_join_pkl, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.time_resolution = time_resolution
        self.spk_emb_dict, self.spk_id_dict = read_spk_emb_file(self._embedfile) # 
        self.embed_mel_file = self._embedfile[:-4]  + '.h5'
        self.embeddings_mel = h5py.File(self.embed_mel_file,'r',libver='latest', swmr=True)
        #self.X = h5py.File('/apdcephfs/share_1316500/donchaoyang/code2/data/feature_32000/merge.h5','r',libver='latest', swmr=True)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len
    
    def get_fea(self,path_):
        waveform, sr = torchaudio.load(path_)
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        output = audio_mono.numpy()
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        new_feat = new_feat.permute(0, 2, 1).squeeze()
        return new_feat

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        #data = self.X[fname][()]
        time_ = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_id_dict[target_event], 1)
        embed_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/embeddings/mel_pkl2/'+embed_file_list[0][:-4]+'.pkl'
        f_e = open(embed_path,'rb')
        embedding = pickle.load(f_e)
        f_e.close()
        # embedding = self.spk_emb_dict[embed_file_list[0]]
        slack_time = np.zeros(self.time_resolution)
        slack_time[0:self.time_resolution] = float(event_to_time[target_event]) # mean time
        embed_label = np.zeros(10)
        frame_level_label = np.zeros(self.time_resolution) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time_[i,0] == -1:
                break
            if time_[0,0]== 0.0 and time_[0,1] == 0.0 and time_[1,0] == -1:
                break
            start = time_to_frame(time_[i,0], self.time_resolution)
            end = min(125,time_to_frame(time_[i,1], self.time_resolution))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        embed_label = torch.as_tensor(embed_label).float()
        slack_time = torch.as_tensor(slack_time).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, slack_time, time_, embedding, embed_label, fname, target_event


class MinimumOccupancySampler(tdata.Sampler):
    """
        docstring for MinimumOccupancySampler
        samples at least one instance from each class sequentially
    """
    def __init__(self, labels, sampling_mode='same', random_state=None):
        self.labels = labels
        data_samples, n_labels = labels.shape # get number of label ,and the dim of label
        label_to_idx_list, label_to_length = [], []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in range(n_labels): # look for all class
            label_selection = labels[:, lb_idx] # select special class on all labels
            if scipy.sparse.issparse(label_selection):
                label_selection = label_selection.toarray()
            label_indexes = np.where(label_selection == 1)[0] # find all audio, where include the special class
            self.random_state.shuffle(label_indexes) # shuffle these index
            label_to_length.append(len(label_indexes))
            label_to_idx_list.append(label_indexes)

        self.longest_seq = max(label_to_length) # find the longest class
        self.data_source = np.zeros((self.longest_seq, len(label_to_length)),
                                    dtype=np.uint32) # build a matrix 
        # Each column represents one "single instance per class" data piece
        for ix, leng in enumerate(label_to_length):
            # Fill first only "real" samples
            self.data_source[:leng, ix] = label_to_idx_list[ix]

        self.label_to_idx_list = label_to_idx_list
        self.label_to_length = label_to_length

        if sampling_mode == 'same':
            self.data_length = data_samples
        elif sampling_mode == 'over':  # Sample all items
            self.data_length = np.prod(self.data_source.shape)

    def _reshuffle(self):
        # Reshuffle
        for ix, leng in enumerate(self.label_to_length):
            leftover = self.longest_seq - leng
            random_idxs = self.random_state.randint(leng, size=leftover)
            self.data_source[leng:,
                             ix] = self.label_to_idx_list[ix][random_idxs]

    def __iter__(self):
        # Before each epoch, reshuffle random indicies
        self._reshuffle()
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(
            self.data_source[random_indices])[:self.data_length]
        return iter(data)

    def __len__(self):
        return self.data_length


def getdataloader(data_file,embedding_file, transform=None,time_resolution=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000(data_file, embedding_file, transform=transform,time_resolution=time_resolution)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_pkl(data_file,embedding_file, transform=None,time_resolution=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_pkl(data_file, embedding_file, transform=transform,time_resolution=time_resolution)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

# def getdataloader_balance(data_file,embedding_file, transform=None, batch_size=None,time_resolution=None, **dataloader_kwargs):
#     with h5py.File(data_file, 'r') as hf:
#         target_events = np.array([event_to_id[target_event.decode()] for target_event in hf['target_event'][:]])
#     num_batches_tr = len(target_events)//(batch_size*32)
#     samplr_train = EpisodicBatchSampler(target_events,num_batches_tr,32,batch_size)
#     dset = HDF5Dataset_32000(data_file, embedding_file, transform=transform,time_resolution=time_resolution)
#     return tdata.DataLoader(dset,batch_sampler=samplr_train,
#                             collate_fn=sequential_collate,num_workers=0,
#                             pin_memory=True,shuffle=False)
    #return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)


def getdataloader_strong_sed(data_file, transform=None, **dataloader_kwargs): # aims at strong sed
    dset = HDF5Dataset_strong_sed_txt(data_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_join(data_file,embedding_file, transform=None,time_resolution=None, **dataloader_kwargs):
    dset = HDF5Dataset_join(data_file, embedding_file, transform=transform,time_resolution=time_resolution)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_join_pkl(data_file,embedding_file, transform=None,time_resolution=None, **dataloader_kwargs):
    dset = HDF5Dataset_join_pkl(data_file, embedding_file, transform=transform,time_resolution=time_resolution)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist, batch_first=batch_first, padding_value=padding_value)
    return padded_seq

def sequential_collate(batches):
    seqs = []
    for data_seq in zip(*batches):
        # print('data_seq[0] ',data_seq[0].shape)
        if isinstance(data_seq[0],
                      (torch.Tensor)):  # is tensor, then pad
            data_seq = pad(data_seq)
        elif type(data_seq[0]) is list or type(
                data_seq[0]) is tuple:  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs

class EpisodicBatchSampler(tdata.Sampler):
    def __init__(self, labels, n_episodes, n_way, n_samples):
        '''
        Sampler that yields batches per n_episodes without replacement.
        Batch format: (c_i_1, c_j_1, ..., c_n_way_1, c_i_2, c_j_2, ... , c_n_way_2, ..., c_n_way_n_samples)
        Args:
            label: List of sample labels (in dataloader loading order)
            n_episodes: Number of episodes or equivalently batch size
            n_way: Number of classes to sample
            n_samples: Number of samples per episode (Usually n_query + n_support)
        '''
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples
        labels = np.array(labels)
        self.samples_indices = []
        for i in range(max(labels) + 1): # 0-9
            ind = np.argwhere(labels == (i)).reshape(-1)
            #print('i ',i)
            # print('ind ',ind)
            ind = torch.from_numpy(ind)
            #print('ind ',ind.shape)
            self.samples_indices.append(ind)

        if self.n_way > len(self.samples_indices):
            raise ValueError('Error: "n_way" parameter is higher than the unique number of classes')

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for batch in range(self.n_episodes): # 采样多少次
            batch = []
            classes = torch.randperm(len(self.samples_indices))[:self.n_way] # torch.randperm(n)返回一个0到n-1的数组
            for c in classes:
                l = self.samples_indices[c]
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1) # c*n_samples
            yield batch
