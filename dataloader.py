import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np
import math
import os


class LIRIS_Sec2SecAV(Dataset):
    def __init__(self, emotion, sec2sec_feat_path, rating_path):
        self.sec2sec_feat_path = sec2sec_feat_path
        if emotion == 'valence':
            self.rating_data = json.load(open(rating_path + 'valence_binary_class.json', 'r'))
        if emotion == 'arousal':
            self.rating_data = json.load(open(rating_path + 'arousal_binary_class.json', 'r'))

    def __getitem__(self, index):
        feat_path = self.sec2sec_feat_path + str(index) + '/' 
        audio_feat = torch.from_numpy(np.load(feat_path + 'mfcc_feat.npy')).float()
        vision_feat = torch.from_numpy(np.load(feat_path + 'vision.npy')).float()

        rating = self.rating_data[str(index)]
        return audio_feat, vision_feat, torch.tensor(rating).float()
        
    def __len__(self):
        return len(self.rating_data)

