#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the PyTorch Dataset class. 
@author: vpeterson
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
#%%
# define costum dataset
class SeizureDataset(Dataset):
    """ Seizure dataset"""
    def __init__(self, file, root_dir, transform=None, target_transform=None):
        """
        Args:
            file (data frame): data frame with file information.

            root_dir (string): Directory with all the numpy files.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        # Read the csv file
        self.seizure_frame = file
        # read spectrogram path
        self.root_dir = root_dir
        # save transform
        self.transform = transform
        self.target_transform = target_transform
        # second column contains the image paths
        self.image_arr = np.asarray(self.seizure_frame.iloc[:, 1])
        # third column is the labels
        self.label_arr = np.asarray(self.seizure_frame.iloc[:, 2])
        # fourth column is for an operation indicator
        self.time_arr = np.asarray(self.seizure_frame.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.seizure_frame.index)

    def __len__(self):
        return len(self.seizure_frame)
    

    def __getitem__(self, idx):
        # Get image name from the pandas df
        file_name = self.root_dir + self.image_arr[idx] + '.npy'

        data = np.load(file_name, allow_pickle=True)
        # to transform to tensor
        data = np.asarray(data.astype(np.float32))
        data = torch.from_numpy(data)

        label = self.label_arr[idx]
        time = self.time_arr[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label, time
    
class SeizureDatasetLabelTime(Dataset):
    """ Seizure dataset"""
    def __init__(self, file, root_dir, transform=None, target_transform=None):
        """
        Args:
            file (data frame): data frame with file information.

            root_dir (string): Directory with all the numpy files.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        # Read the csv file
        self.seizure_frame = file
        # read spectrogram path
        self.root_dir = root_dir
        # save transform
        self.transform = transform
        self.target_transform = target_transform
        # second column contains the image paths
        self.image_arr = np.asarray(self.seizure_frame.iloc[:, 1])
        # third column is the labels
        self.label_arr = np.asarray(self.seizure_frame.iloc[:, 2])
        # fourth column is for an operation indicator
        self.time_arr = np.asarray(self.seizure_frame.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.seizure_frame.index)

    def __len__(self):
        return len(self.seizure_frame)
    

    def __getitem__(self, idx):
        # Get image name from the pandas df
        file_name = self.root_dir + self.image_arr[idx] + '.npy'

        dic = np.load(file_name, allow_pickle=True)
        data = dic.item().get('spectrogram')
        # to transform to tensor
        data = np.asarray(data.astype(np.float32))
        data = torch.from_numpy(data)

        label = dic.item().get('label')

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

class SeizureDatasetv2(Dataset):
    """ Seizure dataset"""
    def __init__(self, file, root_dir, transform=None, target_transform=None):
        """
        Args:
            file (data frame): data frame with file information.

            root_dir (string): Directory with all the numpy files.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        # Read the csv file
        self.seizure_frame = file
        # read spectrogram path
        self.root_dir = root_dir
        # save transform
        self.transform = transform
        self.target_transform = target_transform
        # second column contains the image paths
        self.image_arr = np.asarray(self.seizure_frame.iloc[:, 1])
        # third column is the labels
        self.label_arr = np.asarray(self.seizure_frame.iloc[:, 2])
        # fourth column is for an operation indicator
        self.time_arr = np.asarray(self.seizure_frame.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.seizure_frame.index)

    def __len__(self):
        return len(self.seizure_frame)
    

    def __getitem__(self, idx):
        # Get image name from the pandas df
        file_name = self.root_dir + self.image_arr[idx] + '.npy'

        dic = np.load(file_name, allow_pickle=True)
        data = dic.item().get('spectrogram')
        # to transform to tensor
        data = np.asarray(data.astype(np.float32))
        data = torch.from_numpy(data)

        label = dic.item().get('label')
        label = max(label)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

class StatsRecorder:
    def __init__(self, red_dims=(0,1,3)):
        """Accumulates normalization statistics across mini-batches.
        
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions).
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n
            
class normalize_spec(object):
    def __init__(self, mean, std):
        """ read stats"""
        self.mean=mean
        self.std=std
    def __call__(self, data):
        """"apply normalization stats."""
        return torch.squeeze((data-self.mean) / self.std)
    
class zscore_spec(object):
    def __call__(self, data):
        # print(data.size())
        """"apply zscore norm to specs."""
        mu = torch.mean(data,dim=(1,2),keepdim=True)
        sd = torch.std(data,dim=(1,2),keepdim=True)
        
        return torch.squeeze((data-mu) / sd)
    
class scale_spec(object):
    def __call__(self, data):
        """"apply random scaling."""
        data*=torch.rand(1)  
        return data
    
class permute_spec(object):
    def __call__(self, data):
        """"apply random channel permute different from 1234."""
        import random
        S = [i for i in range(0,4)]
        out = True
        while out:
            idx = random.sample(S,4)
            if idx != [0, 1, 2, 3]:
                out=False        
        data = data[idx]  
        return data

class smoothing_label(object):
    def __call__(self, label, n=5, std=2.5):
        """"apply label smoothing."""
        label_time_smooth = label.copy()
        # for idx in range(len(label)):
        leng = len(label)
        idx_t = np.where(label==1)[0]
        
        if len(idx_t)!=0:
            if leng  - idx_t < n:
                n = leng  - idx_t
                aux = np.arange(idx_t-n,idx_t+n,1)
            elif idx_t - n < 0 : # in the very begining
                n = n + (idx_t - n)
                aux = np.arange(idx_t-n,idx_t+n,1)
            else:
                aux = np.arange(idx_t-n,idx_t+n,1)
                
            if aux.size!=0:                
                gaus =np.exp(-np.power(aux - idx_t, 2.) / (2 * np.power(std, 2.)))
                label_time_smooth[aux] = gaus
            else:
                label_time_smooth[idx_t] = 1
        return label_time_smooth

class smoothing_alllabels(object):
    def __call__(self, label, n=5, std=2.5):
        """"apply label smoothing."""
        label_time_smooth = label.copy()
        # for idx in range(len(label)):
        leng = len(label)
        idx_t = np.where(label==1)[0]

        if len(idx_t)!=0:
            if leng  - idx_t < n:
                n = leng  - idx_t
                aux = np.arange(idx_t-n,idx_t+n,1)
            elif idx_t -5 < 0 : # in the very begining
                n = 5 + (idx_t -5)
                aux = np.arange(idx_t-n,idx_t+n,1)
            else:
                aux = np.arange(idx_t-n,idx_t+n,1)
                
            if aux.size!=0:                
                gaus =np.exp(-np.power(aux - idx_t, 2.) / (2 * np.power(1.5, 2.)))
                label_time_smooth[aux] = gaus
            else:
                label_time_smooth[idx_t] = 1
        else:
            aux = np.arange(0,2*n,1)
            gaus = - np.exp(-np.power(aux - n-1, 2.) / (2 * np.power(std, 2.)))
            label_time_smooth[aux] = gaus
            
        return label_time_smooth
    
class square_label(object):
    def __call__(self, label, n=2):
        """"apply label smoothing."""
        label_time_smooth = label.copy()
        # for idx in range(len(label)):
        leng = len(label)
        idx_t = np.where(label==1)[0]
        
        
        if len(idx_t)!=0:
            aux  = np.arange(idx_t, idx_t+n,1)

            rect = np.ones((n,))
            label_time_smooth[aux] = rect
        return label_time_smooth

class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img

class append_transforms(RandomOrder):
    def __init__(self, FREQ_MASK_PARAM=None,
                       TIME_MASK_PARAN=None,
                       scale=True):

        self.transforms = []
        if FREQ_MASK_PARAM:
            self.transforms.append(T.FrequencyMasking(FREQ_MASK_PARAM))
        if TIME_MASK_PARAN:
            self.transforms.append(T.TimeMasking(TIME_MASK_PARAN))
        if scale:
            self.transforms.append(scale_spec())
