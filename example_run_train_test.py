#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is provided with the purpose of showing how training and testing was
performed. Do not run the code, it will fail since the complete training data 
is not provided.
@author: vpeterson
"""
import os
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import sys
sys.path.insert(1, './utilities')
from Generator import SeizureDatasetLabelTime, scale_spec, permute_spec, smoothing_label
from Model import iESPnet
from TrainEval import train_model_opt, test_model, train_model, get_thr_output, get_performance_indices
import IO
import torchaudio.transforms as T
import pandas as pd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# set the seed for reproducibility
torch.manual_seed(0)
#%%
def get_class_weights(train_df, classes):
    class_sample_count = np.zeros(len(classes,), dtype=int)
    for n, cl in enumerate(classes):
        class_sample_count[n] = sum(train_df.label==cl)
    class_weights = class_sample_count / sum(class_sample_count)             
    return class_weights

def make_weights_for_balanced_classes(train_df, classes, n_concat=2):
    class_sample_count = np.zeros(len(classes,), dtype=int)
    for n, cl in enumerate(classes):
        class_sample_count[n] = sum(train_df.label==cl)
       
    weights = (1 / class_sample_count)
    target = train_df.label.to_numpy()
    samples_weight = weights[target]
    
    for i in range(n_concat):
        if i == 0:
            sampler = samples_weight
        else:
            sampler = np.hstack((sampler, samples_weight))
    
    return torch.tensor(sampler , dtype=torch.float)

#%%
SPE_DIR = './Data/'
# get metadata file
#this is a csv file this the rns_id, file name (data), label and time infor
meta_data_file = './Data/Metadata/allfiles_metadata.csv'
df = pd.read_csv(meta_data_file) 

FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES = 1
learning_rate = 1e-3
batch_size = 128
epochs = 20
num_workers = 4
save_path = '/youroutputpath/'
df_subjects = pd.read_csv('./Data/Metadata/subjects_info.csv')

RNSIDS=df_subjects.rns_deid_id

#%%
hparams = {
         "n_cnn_layers": 3,
        "n_rnn_layers": 3,
        "rnn_dim": [150, 100, 50],
        "n_class": N_CLASSES,
        "out_ch": [8,8,16],
        "dropout": 0.3,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs
        }


def main():
    for s in range (len(RNSIDS)):
        
        model = iESPnet(hparams['n_cnn_layers'],
                       hparams['n_rnn_layers'],
                       hparams['rnn_dim'],
                       hparams['n_class'],
                       hparams['out_ch'],
                       hparams['dropout'],
                       )
        
        save_models = save_path + RNSIDS[s] +'/models/'
        save_predictions = save_path + RNSIDS[s]+'/results/'
        
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        
            
        if not os.path.exists(save_models):
            os.makedirs(save_models)
            
        if not os.path.exists(save_predictions):
            os.makedirs(save_predictions)
            
      
    
        print('Running training for subject ' + RNSIDS[s] + ' [s]: ' + str(s))
       
        
        train_df = df.copy()
        # define train, val and test from df
        test_df = df[df.rns_id==RNSIDS[s]]
        test_df.reset_index(drop=True, inplace=True)
    
        train_df.drop(train_df[train_df['rns_id'] == RNSIDS[s]].index, inplace = True)
            
        # DATA LOADERS
        train_data_ori = SeizureDatasetLabelTime(file=train_df,
                                    root_dir=SPE_DIR,
                                    transform=None, 
                                    target_transform=smoothing_label(),
                                    )
       
    
        transform_train1 = transforms.Compose([T.FrequencyMasking(FREQ_MASK_PARAM),
                                           T.TimeMasking(TIME_MASK_PARAN), 
                                           permute_spec()
                                           
                                           
                                          ])
        
        # data augmentation only in train data
        train_data_trf1 = SeizureDatasetLabelTime(file=train_df,
                                                root_dir=SPE_DIR,
                                                transform=transform_train1, 
                                                target_transform=smoothing_label() 
                                                )
        
        train_data = torch.utils.data.ConcatDataset([train_data_ori, train_data_trf1])
        # testing data should not be balanced
        test_data = SeizureDatasetLabelTime(file=test_df,
                                root_dir=SPE_DIR,
                                transform=None,
                                target_transform=smoothing_label()  
                                )
    
        # weights for classes
        weights = make_weights_for_balanced_classes(train_df, [0,1], n_concat=2)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        
        if len(weights) != len(train_data):
            AssertionError('sampler should be equal to train data shape')
        #%% train
        outputfile = save_models + 'model'
        avg_train_losses, avg_train_f1= train_model_opt(model, hparams, epochs, train_data, sampler, outputfile)
                                                            
        #%% eval        
        best_thr = 0.2
        best_path = save_models + 'model_opt.pth'
                
        # in testing
        outputs_test=test_model(model, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
                
        # in training
        outputs_train=test_model(model, hparams, best_path, train_data_ori)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)
                
        predict_ = { 
                    "train_losses": avg_train_losses,
                    "train_acupr": avg_train_f1,
                    "prediction_te": prediction_te,
                    "prediction_tr": prediction_tr, 
                    "hparams": hparams, 
                    "threshold": 0.2, 
                    "train_size": len(train_data_ori)/len(df)

                    }
        np.save(save_predictions+ RNSIDS[s]+ 'results.npy', predict_)
                
        
        del train_data, test_data

if __name__=='__main__':
    main()
