#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 01:50:39 2019

@author: bharat
"""

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

log_interval =1
test_interval=100
lr =0.001
epochs =256

save_interval = 500
early_stop = 1000
save_best = True 


def train(train_iter, val_iter, model, save_dir):

    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    losses =[]
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, epochs+1):
        total_loss =0
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
        

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            
            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % test_interval == 0:
                val_acc = eval(val_iter, model)
                if val_acc > best_acc:
                    best_acc = val_acc
                    last_step = steps
                    if save_best:
                        save(model, save_dir, 'best', steps)
                else:
                    if steps - last_step >= early_stop:
                        print('early stop by {} steps.'.format(early_stop))
            elif steps % save_interval == 0:
                save(model, save_dir, 'snapshot', steps)  
                
                
                
def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
    

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)

    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)                