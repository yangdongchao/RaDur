# We will recombine the h5 file, for the reason that some audio file is useless.
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
def choose_validation_set():
    ans = []
    train_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/weak_train_choose.tsv',sep='\t',usecols=[0,1])
    labels = train_tsv['event_labels']
    segment_ids = train_tsv['filename']
    trans_labels = []
    filename = []
    class_num_dict = {}
    number_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/class_number_in_train.tsv',sep='\t',usecols=[0,1])
    c_name = number_tsv['class_names']
    c_num = number_tsv['numbers']
    c_name_ls = []
    c_num_ls = []
    for i in c_name:
        c_name_ls.append(i)
    for i in c_num:
        c_num_ls.append(i)
    real_num_dict = {}
    for i in range(len(c_name_ls)):
        real_num_dict[c_name_ls[i]] = int(c_num_ls[i])
    
    for cl in event_labels:
        class_num_dict[cl] = 0 
    #print(labels)
    for i in labels:
        #print(i
        trans_labels.append(i)
    for i in segment_ids:
        filename.append(i)
    print('len(filename) ',len(filename))
    for i in range(len(filename)):
        labels_ls = trans_labels[i].split(',')
        flag = 0
        for lab in labels_ls:
            if class_num_dict[lab] > int(real_num_dict[lab]*0.035):
                flag = 1
                break
        if flag == 0:
            for lab in labels_ls:
                class_num_dict[lab] += 1
            ans.append(filename[i])
    print(class_num_dict)
    print(len(ans))
    return ans

        

_h5file = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/train.h5'



# print(X.shape)
validation_list = choose_validation_set()
train_weak = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/weak_train_choose.tsv',sep='\t',usecols=[0,1])
weak_file_name = train_weak['filename']
weak_labels = train_weak['event_labels']
weak_file_name_ls = []
weak_labels_ls = []
for i in weak_file_name:
    weak_file_name_ls.append(i)
for i in weak_labels:
    weak_labels_ls.append(i)


validate_filename = []
validate_names = []
train_filename = []
train_names = []
for i in range(len(weak_file_name_ls)):
    if weak_file_name_ls[i] in validation_list:
        validate_filename.append(weak_file_name_ls[i])
        validate_names.append(weak_labels_ls[i])
    else:
        train_filename.append(weak_file_name_ls[i])
        train_names.append(weak_labels_ls[i])

train_dict = {'filename': train_filename, 'event_labels': train_names}
df = pd.DataFrame(train_dict)
df.to_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/filelist/weak_train_choose_split.tsv',index=False,sep='\t')

validate_dict = {'filename': validate_filename, 'event_labels': validate_names}
df = pd.DataFrame(validate_dict)
df.to_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/filelist/weak_validate_choose_split.tsv',index=False,sep='\t')


# h5_validate_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/'  + 'validate_no_choose.h5'
# print(h5_validate_name)
# hf_validate = h5py.File(h5_validate_name, 'w')
# num_ = 100
# frames_num = 501
# num_freq_bin = 64
# hf_validate.create_dataset(
#     name='filename', 
#     shape=(0,), 
#     maxshape=(None,),
#     dtype='S80')
# hf_validate.create_dataset(
#     name='events', 
#     shape=(0,), 
#     maxshape=(None,),
#     dtype='S900')
# # hf_validate.create_dataset(
# #     name='mel_feature', 
# #     shape=(0, frames_num, num_freq_bin), 
# #     maxshape=(None, frames_num, num_freq_bin), 
# #     dtype=np.float32)
# hf_validate.create_dataset(
#     name='label',
#     shape=(0,num_,2),
#     maxshape=(None,num_,2),
#     dtype=np.float32
# )
# hf = h5py.File(_h5file, 'r')
#     # X = hf['mel_feature'][:].astype(np.float32)
# y = hf['label'][:].astype(np.float32)
# events = np.array([target_event.decode() for target_event in hf['events'][:]])
# filename = np.array([filename.decode() for filename in hf['filename'][:]])

# n = 0
# for i in range(y.shape[0]): # validation set
#     print('valide ',filename[i],n)
#     if filename[i] in validation_list:
#         # hf_validate['mel_feature'].resize((n + 1, frames_num, num_freq_bin))
#         # hf_validate['mel_feature'][n] = hf['mel_feature'][i].astype(np.float32)
#         hf_validate['filename'].resize((n+1,))
#         hf_validate['filename'][n] = filename[i].encode()
        
#         hf_validate['events'].resize((n+1,))
#         hf_validate['events'][n] = events[i].encode()
#         hf_validate['label'].resize((n+1,num_,2))
#         hf_validate['label'][n] = y[i].astype(np.float32)
#         n += 1

# # save train set
# h5_train_name = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature/'  + 'train_no_choose.h5'
# print(h5_train_name)
# hf_train = h5py.File(h5_train_name, 'w')
# num_ = 100
# frames_num = 501
# num_freq_bin = 64
# hf_train.create_dataset(
#     name='filename', 
#     shape=(0,), 
#     maxshape=(None,),
#     dtype='S80')
# hf_train.create_dataset(
#     name='events', 
#     shape=(0,), 
#     maxshape=(None,),
#     dtype='S900')
# # hf_train.create_dataset(
# #     name='mel_feature', 
# #     shape=(0, frames_num, num_freq_bin), 
# #     maxshape=(None, frames_num, num_freq_bin), 
# #     dtype=np.float32)
# hf_train.create_dataset(
#     name='label',
#     shape=(0,num_,2),
#     maxshape=(None,num_,2),
#     dtype=np.float32
# )
# n = 0 
# for i in range(y.shape[0]):
#     print('valide ',filename[i],n)
#     if filename[i] not in validation_list and filename[i] in weak_file_name_ls:
#         # hf_train['mel_feature'].resize((n + 1, frames_num, num_freq_bin))
#         # hf_train['mel_feature'][n] = hf['mel_feature'][i].astype(np.float32)
#         hf_train['filename'].resize((n+1,))
#         hf_train['filename'][n] = filename[i].encode()
        
#         hf_train['events'].resize((n+1,))
#         hf_train['events'][n] = events[i].encode()
#         hf_train['label'].resize((n+1,num_,2))
#         hf_train['label'][n] = y[i].astype(np.float32)
#         n += 1