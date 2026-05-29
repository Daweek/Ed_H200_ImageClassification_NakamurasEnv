import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os
import glob
# from torchvision.io import read_image
from PIL import Image
from torchvision import transforms



class ExFractal_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        data_ls = []
        label_ls=[]
        for dir_path in glob.glob(f"{img_dir}/*"):
            view_ls = []
            base_path, dir_name =dir_path.rsplit('/', 1)
            print(dir_name)
            for data_path in glob.glob(f"{self.img_dir}/{dir_name}/0*_00000_*.png"):
                #add list of multi view point imagelist to data
                img = Image.open(data_path)
                img = img.convert('RGB')
                view_ls.append(img)
            #add list of multi view point imagelist to datalist
            data_ls.append(view_ls)
            #add label to list
            label_ls.append(int(dir_name))
            
        self.data = data_ls
        self.label = label_ls
        
        self.len = len(label_ls)
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        #select data
        data = self.data[index]
        #select view point
        dice = np.random.randint(0,len(data),1)
        image = data[dice[0]]
        #select label
        label = self.label[index]
        
        
        #transform data
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label).long()
            
        return image, label