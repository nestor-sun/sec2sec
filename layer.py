import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import random
from torch.autograd import Variable


class Pretrained3DResNet(nn.Module):
    def __init__(self, out_dim):
        super(Pretrained3DResNet, self).__init__()
        # ResNet (2+1)D
        self.model = models.video.r2plus1d_18(pretrained=True)
        #Add one fully connected layer
        self.model.fc = nn.Sequential(nn.Dropout(0.0),
                                        nn.Linear(in_features=self.model.fc.in_features, out_features=out_dim))
    def forward(self, video):
        return self.model(video)


class PretrainedResNet(nn.Module):
    def __init__(self, out_dim, pretrained=True):
        super(PretrainedResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                nn.Linear(in_features=self.resnet.fc.in_features, out_features=out_dim))
        self.out_dim = out_dim
    def forward(self, audio):
        out = self.resnet(audio)
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1, norm_method='layer'):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if norm_method == 'layer':
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x_old, x_new):
        return self.norm(self.dropout(x_new) + x_old)


def attention_matmul(q,k,v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)/math.sqrt(d_k))
    #print('QK dimension', scores.shape)
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_of_heads == 0
        self.d_k = int(d_model/num_of_heads)
        self.num_of_heads = num_of_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

    def forward(self, query, key, value):
        num_of_batches = query.size(0)
        query, key, value = [linear(x).view(num_of_batches, -1, self.num_of_heads, self.d_k).transpose(1,2)\
                                    for linear, x in zip(self.linears, (query, key, value))]
        x = attention_matmul(query, key, value)
        x = x.transpose(1, 2).contiguous().view(num_of_batches, self.num_of_heads*self.d_k)
        return x


class Sec2SecCoattention(nn.Module):
    def __init__(self, d, num_of_heads, lstm_hidden_dim, num_of_lstm_layers, num_of_dense_layers,\
                        dense_hidden_dim, device, norm_method, dropout=0.1):
        super(Sec2SecCoattention, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_of_lstm_layers = num_of_lstm_layers
        self.dense_hidden_dim = dense_hidden_dim
        self.device = device
        self.d = d
        self.num_of_heads = num_of_heads
        
        self.visual = PretrainedResNet(d)
        self.audio = PretrainedResNet(d)
        
        self.norm_visual = AddNorm(d, dropout, norm_method)
        self.norm_audio = AddNorm(d, dropout, norm_method)

        self.linear_visual = nn.Linear(d, d)
        self.linear_audio = nn.Linear(d, d)

        self.visual_layer = MultiHeadAttention(num_of_heads, d)
        self.audio_layer = MultiHeadAttention(num_of_heads, d)

        self.coattention_visual = MultiHeadAttention(num_of_heads, d)
        self.coattention_audio = MultiHeadAttention(num_of_heads, d)

        self.coattention_norm_visual = AddNorm(d, dropout, norm_method)
        self.coattention_norm_audio = AddNorm(d, dropout, norm_method)

        self.lstm = nn.LSTM(d+d, lstm_hidden_dim, num_of_lstm_layers, bias=True, batch_first=True)

        fc_layers = []
        prev_dense_dim = lstm_hidden_dim
        for i in range(num_of_dense_layers):
            linear = nn.Linear(prev_dense_dim, dense_hidden_dim)
            linear.weight.data.normal_(mean=0.0, std=1)
            linear.bias.data.zero_()
            fc_layers.append(linear)
            prev_hidden_dim = dense_hidden_dim
        fc_layers.append(nn.Linear(prev_hidden_dim, 1))
        self.fc =  nn.Sequential(*fc_layers)
    
    def get_init_hidden_state(self, batch_size):
        h0 = Variable(torch.zeros(self.num_of_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device)
        c0 = Variable(torch.zeros(self.num_of_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device)
        return h0, c0

    def forward(self, visual_batch, audio_batch):
        visual_shape = visual_batch.shape
        audio_shape = audio_batch.shape
        #visual_shape: batch_size, audio_length, 3_channel, feat_dim1, feat_dim2
        #audio_shape: batch_size, audio_length, 3_channel, feat_dim1, feat_dim2
        x_visual = visual_batch.view(visual_shape[0] * visual_shape[1], visual_shape[2], visual_shape[3], visual_shape[4])
        x_audio = audio_batch.view(audio_shape[0] * audio_shape[1], audio_shape[2], audio_shape[3], audio_shape[4])
        x_visual = self.visual(x_visual)
        x_audio = self.audio(x_audio)
        
        x_visual = self.linear_visual(self.norm_visual(x_visual,self.visual_layer(x_visual, x_visual, x_visual)))
        x_audio = self.linear_audio(self.norm_audio(x_audio, self.audio_layer(x_audio, x_audio, x_audio)))
        
        x_coattention_visual = self.coattention_norm_visual(x_visual, self.coattention_visual(x_visual, x_audio, x_audio))
        x_coattention_audio = self.coattention_norm_audio(x_audio, self.coattention_audio(x_audio, x_visual, x_visual))
        
        x_coattention = torch.cat((x_coattention_visual, x_coattention_audio), dim=1)

        x_coattention = x_coattention.view(visual_shape[0], visual_shape[1], self.d + self.d)
        lstm_output, _ = self.lstm(x_coattention, self.get_init_hidden_state(visual_shape[0]))
        
        lstm_last_step_output = lstm_output[: , visual_shape[1] - 1, :]
        out = self.fc(lstm_last_step_output)
        return torch.sigmoid(out)


