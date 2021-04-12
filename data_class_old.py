import pandas as pd
import sklearn, sklearn.utils
import torch
import matplotlib.pyplot as plt

from scipy.io import wavfile
from python_speech_features import mfcc
    
import numpy as np
# import mmap

from tqdm import tqdm


import librosa
import librosa.display
import json

import functools
import operator

import os




            
            
class fsd_dataset(object):
    
    def __init__(self, csv_file, path, train = True):

        self.train = train
      
        self.dataset_frame = pd.read_csv(csv_file).dropna(axis=0, how='all')
        
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        #self.dataset_frame = self.dataset_frame.drop(['index'], axis=1)
      
        #self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame.label.unique()

        self.all_samples = []
        
        self.metadata_array = []
        self.spectrogram_array = []
        

      
        
        self.path = path


        #self.time_window = 140
        
        self.hop_length = 256
        self.n_fft = 1103

        self.samplerate = 44100

        self.window_length = int(self.samplerate/1) # depending on sample length
        self.overlap_length = 5000

        self.n_mels = 42
        self.n_mfcc = 26
        
        self.sample_count = 0

        self.samples = []
        
        self.decoded_dict = {}

        self.y_by_label = {}
        self.y_counts = {}
        
        
        for file in tqdm(self.dataset_frame['fname']):
            


            # S, sr = librosa.load(self.path + file, sr=self.samplerate, mono=True)
            #raw = librosa.resample(raw, sr, self.samplerate)
            rate, S2 = wavfile.read(self.path + file)
           
            
            label = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label'].values[0]
            
            
            
            # split_points = librosa.effects.split(raw, top_db=80, frame_length=self.n_fft, hop_length=self.hop_length)
            
            # S_cleaned = []
                        
            # for piece in split_points:
             
            #     S_cleaned.append(raw[piece[0]:piece[1]])
         
            # S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
           
            # for i in range(0,1):
            #     S = np.concatenate((S,S),axis=0)
                
                




            for idx in range(0, S2.shape[0]-self.window_length, self.overlap_length):
                
                # sample = S[idx : idx+self.window_length]
                sample2 = S2[idx : idx+self.window_length]
                
                # x_mel = librosa.feature.melspectrogram(sample, 
                #                         sr=self.samplerate,
                #                         n_fft=self.n_fft,
                #                         hop_length=self.hop_length
                                 
                #                       )
                
                ##
                
                # x = librosa.feature.mfcc(
                #     sample,
                #     sr=self.samplerate,
                #     n_mfcc=self.n_mels,
                #     n_fft=self.n_fft,
                #     hop_length=self.hop_length
                # )


                x2 = mfcc(sample2, rate, numcep=self.n_mfcc, nfilt=self.n_mels, nfft = self.n_fft).T                
                

                
                x2 = self.normalize_stuff(x2)
                

                
                self.sample_count += len(x2)
                
                      

                if label not in self.y_by_label:
                   
                    self.y_by_label[label] = len(self.y_by_label)

                y = self.y_by_label[label]

                if y not in self.y_counts:
                    self.y_counts[y] = 0
                
                
                self.samples.append((x2, y))
                self.decoded_dict.update(self.y_by_label)
                self.y_counts[y] += 1
                
                # if self.y_counts[y] < 300:
                    
                #     self.samples.append((x2, y))
                #     self.y_counts[y] += 1
                    
                # if ((self.y_counts[y] % 1000) == 0):
                    
                #     # plt.imshow(x)
                #     # plt.title(str(y))
                #     # plt.show()
                    
                #     plt.imshow(x2)
                #     plt.title(str(y) + '_1')
                #     plt.show()
                            
        print(f'len(self.samples): {len(self.samples)}')
        print('self.y_counts:')
        for key, y in self.y_by_label.items():
            print(f'{key}: {self.y_counts[y]}')

        self.y_weights = torch.zeros((len(self.y_counts.keys()),), dtype=torch.float)
        
        for key in self.y_counts.keys():
            # self.y_weights[key] = 1  -  (self.y_counts[key] / len(self.samples))
            self.y_weights[key] = (self.y_counts[key] / len(self.samples)) 
        
        
    def normalize_stuff(self, stuff):
        
        x_max = stuff.max()
        x_min = stuff.min()
        stuff -= x_min
        stuff /= (x_max - x_min) # 0..1
        stuff -= 0.5
        stuff *= 2.0 # -1 .. 1
        
        return stuff


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return np.expand_dims(x, axis=0).astype(np.float32), y








