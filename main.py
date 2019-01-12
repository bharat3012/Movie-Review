#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:01:51 2018

@author: bharat
"""

import os
#import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets


#default parameters 

batch_size =64


shuffle = False

max_norm = 3.0
embed_dim = 128
kernel_num =100
save_dir = 'snapshot'
static = False
device =-1 # for CPU training
cuda = False
snapshot =None
predict = None
test = False
#Preprocessing
# load MR dataset
def mr(text_field, label_field, **kwargs):
    train_data, val_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, val_data)
    label_field.build_vocab(train_data, val_data)
    train_iter, val_iter = data.Iterator.splits(
                                (train_data, val_data), 
                                batch_sizes=(batch_size, len(val_data)),
                                **kwargs)# kwargs - Passed to the constructor of dataset.Datasets (here TarDataset Class)
    return train_iter, val_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True) #Text Field contains information on how we want datapreprocessed- lowercase--- Field for sentences
label_field = data.Field(sequential=False)# label Field contains information on how datapreprocessed without tokenization--Field for label data
train_iter, val_iter = mr(text_field, label_field)



# update parameters and print
embed_num = len(text_field.vocab)
class_num = len(label_field.vocab) - 1
cuda = (not cuda) and torch.cuda.is_available(); del cuda

save_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))#making a Directory in snapshot named after time and date"



# model- load from model.py 
cnn = model.CNN_Text(embed_num, embed_dim, class_num, kernel_num, static)
if snapshot is not None:
    print('\nLoading model from {}...'.format(snapshot))
    cnn.load_state_dict(torch.load(snapshot))


    
    
# train or predict
if predict is not None:
    label = train.predict(predict, cnn, text_field, label_field)
    print('\n[Text]  {}\n[Label] {}\n'.format(predict, label))
elif test:
    try:
        train.eval(test_iter, cnn) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, val_iter, cnn, save_dir)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

    
    