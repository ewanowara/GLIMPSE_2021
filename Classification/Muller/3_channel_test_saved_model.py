from Shraman_loading_data_scene_kinds_150_Seg_Channels_argparse import MultiPartitioningClassifier, cuda_base, device_ids, num_epochs
import yaml
from argparse import Namespace
import torch
import argparse




with open('../config/baseM_Shraman_150_Channels.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()

# Choose the first n_steps batches with batch_size samples in each batch
n_steps = 4000

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary
import random
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))


num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]

multi_hop_dim_1 = 80
multi_hop_dim_2 = 30


class GeoClassification(nn.Module):

    def __init__(self):
        super(GeoClassification, self).__init__()

        self.rgb_model = models.resnet34(pretrained=True)  ## load pre-trained weights ##
        self.rgb_features = nn.Sequential(*list(self.rgb_model.children())[:-2])
        
        self.seg_model = models.resnet18(pretrained=False)  ## load pre-trained weights ##
        self.seg_features = nn.Sequential(*list(self.seg_model.children())[:-2])
        
        self.W_s1 = nn.Linear(512, multi_hop_dim_1)
        self.W_s2 = nn.Linear(multi_hop_dim_1, multi_hop_dim_2)
        
        self.rgb_linear_1 = nn.Linear(multi_hop_dim_2 * 512, 6000)
        
        self.seg_linear_1 = nn.Linear(multi_hop_dim_2 * 512, 6000)
        
        self.atmf_1 = nn.Linear(6000,80)
        self.atmf_2 = nn.Linear(80,30)
        self.atmf_3 = nn.Linear(30,1)
        
        self.fc_concat1 = nn.Linear(12000,6000)
        self.fc_concat2 = nn.Linear(6000, num_classes_coarse)
        
        self.dropout = nn.Dropout(p=0.20)
        self.relu = torch.nn.LeakyReLU()
        
    
    def attention_net(self, features):

        attn_weight_matrix = self.W_s2(self.dropout(self.relu(self.W_s1(features))))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix


    def forward(self, rgb_image, seg_image):
        
        seg_image = torch.permute(seg_image, (0,3,1,2))
        
        rgb_features = self.rgb_features(rgb_image).reshape(-1, 512, 49).permute(0,2,1)
        seg_features = self.seg_features(seg_image).reshape(-1, 512, 49).permute(0,2,1)
        
        rgb_attention_weights = self.attention_net(rgb_features)
        seg_attention_weights = self.attention_net(seg_features)

        att_rgb_features = torch.bmm(rgb_attention_weights, rgb_features)
        att_rgb_features = att_rgb_features.view(-1, att_rgb_features.size()[1] * att_rgb_features.size()[2])

        att_seg_features = torch.bmm(seg_attention_weights, seg_features)
        att_seg_features = att_seg_features.view(-1, att_seg_features.size()[1] * att_seg_features.size()[2])
        
        att_rgb_features = self.dropout(self.relu(self.rgb_linear_1(att_rgb_features)))
        att_seg_features = self.dropout(self.relu(self.seg_linear_1(att_seg_features)))
        
        
        s_rgb = self.atmf_3(self.dropout(self.relu(self.atmf_2(self.dropout(self.relu(self.atmf_1(att_rgb_features)))))))
        s_seg = self.atmf_3(self.dropout(self.relu(self.atmf_2(self.dropout(self.relu(self.atmf_1(att_seg_features)))))))

        s_comb = torch.cat((s_rgb, s_seg), 0)
        s_comb = F.softmax(s_comb, dim=0) + 1
        att_rgb_features = torch.mul(att_rgb_features, s_comb[0].item())
        att_seg_features = torch.mul(att_seg_features, s_comb[1].item())

        concat_embed = torch.cat((att_rgb_features, att_seg_features),1)
        
        concat_embed = self.fc_concat1(concat_embed)
        concat_embed = self.dropout(concat_embed)
        concat_embed = self.relu(concat_embed)


        output = self.fc_concat2(concat_embed)
        
        return (output)

device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
model = GeoClassification()
model = model.to(device)
model = nn.DataParallel(model, device_ids = device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = config["momentum"], weight_decay = config["weight_decay"])


print("======================================")
print("Evaluating the test set using the saved model")
print("======================================")


#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
saved_model = torch.load('./saved_models/natural_3_channel_segNPT.tar')
model.load_state_dict(saved_model['Model_state_dict'])
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

model.to(device)
        

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []

with torch.no_grad():
        
    n_correct = 0
    n_samples = 0

    for i, (rgb_image, seg_image, label, _, _, _) in enumerate(val_data_loader):
            
        rgb_image = rgb_image.type(torch.float32).to(device)
        seg_image = seg_image.type(torch.float32).to(device)

        label = label[0].to(device)


        # Forward pass
        model.eval()
        outputs = model(rgb_image, seg_image)

        
        if i == 0:
            # save a batch of data to write the triplet sampling code
            outputs1 = outputs
            outputs1.cpu()
            outputs1.numpy()
            print('saving outputs')
            np.save('outputs_Muller.npy', outputs1)

            # rgb_image1 = rgb_image1
            label1 = label
            label1.cpu()
            label1.numpy()
            print('saving labels')
            np.save('labels_Muller.npy', label1)

            # print('saving RGB images')
            # np.save('labels_Muller.npy', label1)


        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        #print(label)
        #print(predicted)
        n_samples += label.size(0)
        n_correct += (predicted == label).sum().item()

        target_total_test.append(label)
        predicted_total_test.append(predicted)
        model_outputs_total_test.append(outputs)

        target_inter = [t.cpu().numpy() for t in target_total_test]
        predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
        outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]
        #print(target_inter[-1].shape)
        #print(predicted_inter[-1].shape)
        #print(outputs_inter[-1].shape)
        target_inter =  np.stack(target_inter, axis=0).ravel()
        predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
        outputs_inter = np.concatenate(outputs_inter, axis=0)


    print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
    print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
    print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
    print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
    print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
    print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
    print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
    print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
    print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
