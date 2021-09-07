from Shraman_loading_data_scene_kinds_3_Seg_Channels import MultiPartitioningClassifier
import yaml
from argparse import Namespace
import torch

with open('../config/baseM_Shraman.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()

# Choose the first n_steps batches with 64 samples in each batch
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
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))


num_epochs = config["max_epochs"]
num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]

class GeoClassification(nn.Module):

    def __init__(self):
        super(GeoClassification, self).__init__()

        self.rgb_model = models.resnet34(pretrained=True)  ## load pre-trained weights ##
        self.rgb_features = nn.Sequential(*list(self.rgb_model.children())[:-2])
    
        self.rgb_linear_1 = nn.Linear(49*512, 6000)  ## Change this to (49*512, 6000) in case of ResNet34 or ResNet18
        self.rgb_linear_2 = nn.Linear(6000, num_classes_coarse)
        
        self.dropout = nn.Dropout(p=0.20)
        self.relu = torch.nn.LeakyReLU()
        
    
    def forward(self, rgb_image):
        
        
        rgb_features = self.rgb_features(rgb_image).reshape(-1, 512*49) ## Change this to (-1, 512*49) in case of ResNet34 or ResNet18
        #print(rgb_features.shape)
        
        rgb_features = self.rgb_linear_1(rgb_features)
        rgb_features = self.dropout(rgb_features)
        rgb_features = self.relu(rgb_features)

        output = self.rgb_linear_2(rgb_features)
        return (output)
    

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = GeoClassification()     
model = model.to(device)
model = nn.DataParallel(model, device_ids=[2, 3])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = config["momentum"], weight_decay = config["weight_decay"])
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma= config["gamma"])

print(summary(model, (3, 224, 224)))



import warnings
warnings.filterwarnings("ignore")

n_total_steps = len(train_data_loader)

batch_wise_loss = []
batch_wise_micro_f1 = []
batch_wise_macro_f1 = []
epoch_wise_top_50_accuracy = []
epoch_wise_top_10_accuracy = []

for epoch in range(num_epochs):
    for i, (rgb_image, _, label, _, _, _) in enumerate(train_data_loader):
        rgb_image = rgb_image.type(torch.float32).to(device)

        label = label[0].to(device)
        
         # Forward pass
        model.train()
        outputs = model(rgb_image)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr_scheduler.step()
        
        
        if (i+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        if (i+1) == n_steps:
            break
    
    #step_lr_scheduler.step()

    target_total_test = []
    predicted_total_test = []
    model_outputs_total_test = []

    with torch.no_grad():
        
        n_correct = 0
        n_samples = 0

        for i, (rgb_image, _, label, _, _, _) in enumerate(val_data_loader):
            
            rgb_image = rgb_image.type(torch.float32).to(device)

            label = label[0].to(device)

            # Forward pass
            model.eval()
            outputs = model(rgb_image)
            #print(outputs)
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
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            outputs_inter = np.concatenate(outputs_inter, axis=0)


        current_top_10_accuracy = topk_accuracy(target_inter, outputs_inter, k=10)
        epoch_wise_top_10_accuracy.append(current_top_10_accuracy)
        current_top_50_accuracy = topk_accuracy(target_inter, outputs_inter, k=50)
        epoch_wise_top_50_accuracy.append(current_top_50_accuracy)
        
        acc = 100.0 * n_correct / n_samples
        print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Micro F1 on the testing: {f1_score(target_inter, predicted_inter, average="micro")}')
        print(f' Macro F1 on the testing: {f1_score(target_inter, predicted_inter, average="macro")}')
        print(f' Precision on the testing: {precision_score(target_inter, predicted_inter, average="macro")}')
        print(f' Recall on the testing: {recall_score(target_inter, predicted_inter, average = "macro")}')
        print(f' Best Top_10_accuracy on test set till this epoch: {max(epoch_wise_top_10_accuracy)} Found in Epoch No: {epoch_wise_top_10_accuracy.index(max(epoch_wise_top_10_accuracy))+1}')
        print(f' Best Top_50_accuracy on test set till this epoch: {max(epoch_wise_top_50_accuracy)} Found in Epoch No: {epoch_wise_top_50_accuracy.index(max(epoch_wise_top_50_accuracy))+1}')
        print(f' Top_10_accuracy: {epoch_wise_top_10_accuracy}')
        print(f' Top_50_accuracy: {epoch_wise_top_50_accuracy}')


