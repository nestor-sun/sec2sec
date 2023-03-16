import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torchvision.transforms as transforms

from layer import Sec2SecCoattention
from dataloader import LIRIS_Sec2SecAV
from utils import get_train_val_test_size, set_seed

from datetime import datetime
import argparse
import os
import json



parser = argparse.ArgumentParser()
parser.add_argument('-emotion', type=str, default='arousal')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-num_of_epochs', type=int, default=250)
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-video_length', type=int, default=15)
parser.add_argument('-d_input', type=int, default=1024)
parser.add_argument('-num_of_dense_layers', type=int, default=1)
parser.add_argument('-dense_hidden_dim', type=int, default=1024)
parser.add_argument('-lstm_hidden_dim', type=int, default=1024)
parser.add_argument('-num_of_heads', type=int, default=64)
parser.add_argument('-seed_number', type=int, default=3407)
parser.add_argument('-lr', type=float, default=1e-7)
parser.add_argument('-lstm_layer', type=int, default=1)
parser.add_argument('-norm', type=str, default='layer')
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-max_grad_norm', type=int, default=1)

parser.add_argument('-rating_path', type=str)
parser.add_argument('-sec2sec_feature_path', type=str)
parser.add_argument('-output_path', type=str)

opt = parser.parse_args()

set_seed(opt.seed_number)


data = LIRIS_Sec2SecAV(opt.emotion, opt.sec2sec_feature_path, opt.rating_path)

train_size, val_size, test_size = get_train_val_test_size(len(data), (7,1,2))
train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=opt.batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=opt.batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=opt.batch_size)

model = Sec2SecCoattention(opt.d_input, opt.num_of_heads, opt.lstm_hidden_dim, opt.lstm_layer, opt.num_of_dense_layers, \
        opt.dense_hidden_dim, opt.device, opt.norm, opt.dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

loss_fn = nn.BCELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

model.to(opt.device)
loss_fn.to(opt.device)

training_loss_list = []
validation_loss_list = []

best_val_accuracy = 0
best_model = model

prev_time = datetime.now()
early_stop_count = 0


def metrics(preds, targets):
    TP, FP, TN, FN = 0, 0, 0, 0
    for pred, target in zip(preds, targets):
        if (pred == 1) and (target == 1):
            TP += 1
        elif (pred == 1) and (target == 0):
            FP += 1
        elif (pred == 0) and (target == 0):
            TN += 1
        else:
            FN += 1
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    f1_score = (2*precision*recall)/(precision+recall)
    return {'accuracy': accuracy, 'precision': precision, 'recall' :recall, 'f1_score': f1_score}


for e in range(opt.num_of_epochs):
    
    training_loss_per_epoch = 0
    val_loss_per_epoch = 0
    model.train()
    
    print('Epoch', e, 'Start training')
    for step, batch in enumerate(train_dataloader):
        audio = batch[0].to(opt.device)
        visual = batch[1].to(opt.device)
        rating = batch[2].to(opt.device)
        output = model(audio, visual)
        loss = loss_fn(output.squeeze(), rating.float())
        loss.backward()
        clip_grad_norm_(model.parameters(), opt.max_grad_norm) # clipping gradient for avoiding exploding gradients
        optimizer.step()

        training_loss_per_epoch += float(loss)

    scheduler.step()
    training_loss_per_epoch = training_loss_per_epoch/train_size
    training_loss_list.append(training_loss_per_epoch)
    
    print('Start validating')
    model.eval()
    val_pred_list = []
    val_label_list = []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            audio = batch[0].to(opt.device)
            visual = batch[1].to(opt.device)
            rating = batch[2].to(opt.device)
            output = model(audio, visual)
 
            loss = loss_fn(output.squeeze(), rating.float()) 
            preds = output.reshape(-1).detach().round().tolist()
            labels = rating.reshape(-1).tolist()
            
            val_pred_list += preds
            val_label_list += labels
            val_loss_per_epoch += float(loss)
    
    val_loss_per_epoch = val_loss_per_epoch/val_size
    validation_loss_list.append(val_loss_per_epoch)
    
    val_accuracy = metrics(val_pred_list, val_label_list)['accuracy']
    if val_accuracy > best_val_accuracy:
        best_model = model
        best_val_accuracy = val_accuracy

    if (e > 20) and (val_loss_per_epoch > validation_loss_list[-2]):
        early_stop_count += 1
        if early_stop_count == 5:
            break
    else:
        early_stop_count = 0
        
    current_time = datetime.now()
    print('Current time at', current_time, 'Epoch took', current_time - prev_time, \
            'Training loss', training_loss_per_epoch, 'Validation loss', val_loss_per_epoch, 
            'Validation Accuracy', val_accuracy, 'Best val accuracy', best_val_accuracy)
    prev_time = current_time


test_pred_list = []
test_label_list = []

model.eval()
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        audio = batch[0].to(opt.device)
        visual = batch[1].to(opt.device)
        rating = batch[2].to(opt.device)
        output = model(audio, visual)
 
        preds = output.reshape(-1).detach().round().tolist()
        labels = rating.reshape(-1).tolist()

        test_pred_list += preds
        test_label_list += labels

report = metrics(test_pred_list, test_label_list)

output_path = opt.output_path + opt.emotion + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


output_foldername = '_'.join([str(opt.batch_size), 'batchsize', str(opt.num_of_heads), \
                            'heads', str(opt.lstm_layer), 'lstmlayernum', opt.norm + 'norm', \
                            str(opt.num_of_dense_layers), 'denselayernum', str(opt.dense_hidden_dim),\
                            'densehiddim', str(opt.lstm_hidden_dim), 'lstmhiddim', str(opt.seed_number)])

output_path += output_foldername + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


torch.save(best_model.state_dict(), output_path + 'best_model.pth')
json.dump(training_loss_list, open(output_path + 'training_loss_list.json', 'w'))
json.dump(validation_loss_list, open(output_path + 'val_loss_list.json', 'w'))
json.dump(report, open(output_path + 'testing_report.json', 'w'))

preds_and_labels = {'preds': test_pred_list, 'labels': test_label_list}
json.dump(preds_and_labels, open(output_path + 'preds_labels.json', 'w'))

with open(output_path + 'readme.txt', 'w') as f:
    f.write('Stopping epoch number ' + str(e + 1) + '\n')
    f.write('Best validation accuracy' + str(best_val_accuracy) + '\n')






