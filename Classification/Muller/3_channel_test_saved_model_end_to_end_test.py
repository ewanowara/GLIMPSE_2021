from Shraman_loading_data_scene_kinds_150_Seg_Channels_argparse import MultiPartitioningClassifier, cuda_base, device_ids, num_epochs
import yaml
from argparse import Namespace
import torch
import argparse
import torch.nn.functional as F

with open('../config/baseM_Shraman_150_Channels.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()

# load the json file to find images from the same geo-cell class as the query image 
import json
import random
# TODO: change to trainig set
with open(model_params['val_label_mapping'], "r") as f: # /cis/home/enowara/Muller/GeoEstimation-master/resources/yfcc_25600_places365_mapping_h3.json
    target_mapping_tmp = json.load(f) # lenght: 25600 # TODO: load this once outside this loop and outside the batch loop too, just once per dataset

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

def process_sample(file_name): 
# load the positive or negative sample image using the file name
# process it and apply all transforms
# pass through the pre-trained network to get the embedding and predictions

    # load that RGB image and segmentation map  
    # TODO: for now paths are hardcoded because the RGB image path here needs mp16_rgb_images to be added since we are reading in a single jpg file instead of the whole msg file
    # rgb_image = Image.open(model_params['msgpack_train_dir2'] + file_name)
    # seg_image = Image.open(model_params['msgpack_train_seg_dir'] + file_name.replace('.jpg', '.png'))
    rgb_image = Image.open(model_params['msgpack_val_dir2'] + file_name)
    seg_image = Image.open(model_params['msgpack_val_seg_dir'] + file_name.replace('.jpg', '.png'))
    
    # process RGB image
    if rgb_image.mode != "RGB":
        rgb_image = rgb_image.convert("RGB")

    if rgb_image.width > 320 and rgb_image.height > 320:
        rgb_image = torchvision.transforms.Resize(320)(rgb_image)
    
    tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(224), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )

    rgb_image = tfm(rgb_image)

    # process segmented image 
    if seg_image.mode != "RGB":
        seg_image = seg_image.convert("RGB")

    if seg_image.width > 320 and seg_image.height > 320:
        seg_image = torchvision.transforms.Resize(320)(seg_image)    

    seg_image = torch.from_numpy(np.array(seg_image))
    
    # to let the model work on a single image, add batch dimension
    
    rgb_image = rgb_image[None, :,:,:].float()
    seg_image = seg_image[None, :,:,:].float()
    
    # get its embedding - pass through model again 
    (emb, out) = model(rgb_image, seg_image) 
    return (emb, out) 

def compute_triplet_loss(q_emb, p_emb, n_emb, triplet_loss_type):
    # compute triplet loss using q_emb, p_emb, n_emb
     
    if triplet_loss_type == 'L2': # L2 loss
        qp_distances = (q_emb - p_emb).pow(2).sum(1)  # .pow(.5) 
        qn_distances = (q_emb - n_emb).pow(2).sum(1)  # .pow(.5)
        margin = 1 # TODO: set margin based on the data
        triplet_loss = F.relu(qp_distances - qn_distances + margin)#.cpu().numpy()  

    elif triplet_loss_type == 'L1': # L1 loss
        qp_distances = abs(q_emb - p_emb).sum(1)  # .pow(.5) 
        qn_distances = abs(q_emb - n_emb).sum(1)  # .pow(.5)
        margin = 1 # TODO: set margin based on the data
        triplet_loss = F.relu(qp_distances - qn_distances + margin)#.cpu().numpy()  

    elif triplet_loss_type == 'IBM': # IBM loss
        qp_distances = abs(q_emb - p_emb).sum(1)  # L1 distance, can try the same loss formulation with L2 distance too
        qn_distances = abs(q_emb - n_emb).sum(1) # L1 distance
        probs = F.softmax([qp_distances, qn_distances])
        triplet_loss = abs(probs[0]) + abs(1 - probs[1])

    return triplet_loss
    
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
        
        concat_embed_fc = self.fc_concat1(concat_embed)
        concat_embed_fc = self.dropout(concat_embed_fc)
        concat_embed_fc = self.relu(concat_embed_fc)

        output = self.fc_concat2(concat_embed_fc)

        return (concat_embed, output) # returns embeddings of input images (concat_embed) and predicted logit values (output) 

device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
model = GeoClassification()
model = model.to(device)
model = nn.DataParallel(model, device_ids = device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = config["momentum"], weight_decay = config["weight_decay"])
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma= config["gamma"])

print("======================================")
print("Evaluating the test set using the saved model")
print("======================================")

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# saved_model = torch.load('./saved_models/natural_3_channel_segNPT.tar')
saved_model = torch.load('./saved_models/natural_3_channel_segNPT.tar', map_location = 'cpu')
model.load_state_dict(saved_model['Model_state_dict'])
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

model.to(device)

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []

S2_cell_fineness = 0 # 0 - corase, 1 - middle, 2 - fine

for epoch in range(num_epochs):    
    for i, (rgb_image, seg_image, label, _, _, _, img_name) in enumerate(val_data_loader):#train_data_loader):
            
        rgb_image = rgb_image.type(torch.float32).to(device)
        seg_image = seg_image.type(torch.float32).to(device)
        
        label = label[S2_cell_fineness].to(device)

        # Forward pass
        model.train() 
        (concat_embed, outputs) = model(rgb_image, seg_image)

        # sample a triplet, given the current batch
        triplet_loss = np.array([]) 

        outputs_tmp = outputs
        outputs_tmp = outputs_tmp.detach()
        outputs_tmp = outputs_tmp.cpu()
        outputs_tmp = outputs_tmp.numpy()
        label_tmp = label.cpu().numpy() 

        for qq in range(2):#rgb_image.shape[0]): # query image index 
            # query image
            q_emb = concat_embed[qq,:] # embedding
            
            q_out = np.argsort(outputs_tmp[qq,:]) # sorts predicted logits in increasing order, min value (lowest cross entropy loss) output first
            q_GT = label_tmp[qq]# ground truth geo-cell class             
            q_file_name = img_name[qq] # name of file corresponding to the input (output by the data loader)

            # positive image
            possible_positives = []
            for key in target_mapping_tmp: # for all image names, get 3D arrays of labels 
                # key is the image name
                value = target_mapping_tmp[key]
                if value[S2_cell_fineness] == q_GT: 
                    if key != q_file_name: # make sure the file is not the same as the query image file name q_file_name
                        possible_positives.append(key)
                    
            # random sampling      
            if len(possible_positives) != 0:
                p_file_name = random.choice(possible_positives) # randomly selected file from the query geo-cell class

                # check if file exist                
                # rgb_image_name = model_params['msgpack_train_dir2'] + p_file_name
                # seg_image_name = model_params['msgpack_train_seg_dir'] + p_file_name.replace('.jpg', '.png')
                rgb_image_name = model_params['msgpack_val_dir2'] + p_file_name
                seg_image_name = model_params['msgpack_val_seg_dir'] + p_file_name.replace('.jpg', '.png')
    
                #  = ('/cis/home/enowara/Muller/GeoEstimation-master/resources/images/mp16/mp16_rgb_images/' + p_file_name)
                #  = ('/cis/home/enowara/Muller/GeoEstimation-master/resources/images/mp16/mp16_seg_images_PNG/' + p_file_name.replace('.jpg', '.png'))
                if os.path.exists(rgb_image_name) and os.path.exists(seg_image_name):
                    (p_emb, p_out) = process_sample(p_file_name)

                    # negative image 
                    # if there is a positive sample in the dataset (more than one image in the same cell)
                    k = np.array(np.where(q_out == q_GT))[0][0]
                    q_out = np.array(q_out)
                    q_out_k_minus_1 = q_out[0:k] 
                
                    # random sampling: select a negative class from the k-1 classes, and then select an image from that class as a negative example
                    # try also:
                    #       closest in kilometers to the query geo-cell class - 
                    #       closest in embedding space
                    #       most similar segmentation map to the query image's segmentation
                    
                    possible_negatives = []
                    while len(possible_negatives) == 0: 
                        n_class = random.choice(q_out_k_minus_1)
                        for key in target_mapping_tmp: # for all image names, get 3D arrays of labels 
                            # key is the image name
                            value = target_mapping_tmp[key]
                            if value[S2_cell_fineness] == n_class: 
                                possible_negatives.append(key)

                    n_file_name = random.choice(possible_negatives)

                    # check if file exist
                    # rgb_image_name = model_params['msgpack_train_dir2'] + n_file_name
                    # seg_image_name = model_params['msgpack_train_seg_dir'] + n_file_name.replace('.jpg', '.png')
                    rgb_image_name = model_params['msgpack_val_dir2'] + n_file_name
                    seg_image_name = model_params['msgpack_val_seg_dir'] + n_file_name.replace('.jpg', '.png')
                    
                    if os.path.exists(rgb_image_name) and os.path.exists(seg_image_name):
                        (n_emb, n_out) = process_sample(n_file_name)

                        # only compute the triplet loss if a triplet exists
                        triplet_loss_type = 'L2'
                        triplet_loss_tmp = compute_triplet_loss(q_emb, p_emb, n_emb, triplet_loss_type)
                        triplet_loss_tmp = triplet_loss_tmp.detach()
                        triplet_loss_tmp = triplet_loss_tmp.cpu()
                        triplet_loss_tmp = triplet_loss_tmp.numpy()
                        triplet_loss = np.append(triplet_loss, triplet_loss_tmp) # save mean loss for all samples in a batch 
        
        print('len(triplet_loss)')
        print(len(triplet_loss))
        
        if len(triplet_loss) > 0:
            tripet_loss_mean = triplet_loss.mean() # TODO: compare the value of this loss to that of CE loss
        else:
            tripet_loss_mean = 0
        
        print('tripet_loss_mean')
        print(tripet_loss_mean)
        
        loss_class = criterion(outputs, label) # compute cross-entropy classification loss
        triplet_weight = 0.001 # TODO: how to set this weight ? Now triplet loss is at 1000 X larger scale than classification loss
        loss = loss_class + triplet_weight * tripet_loss_mean # total loss is from the classification and triplet 

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        print('COMPUTED loss.backward()')
        optimizer.step()
        step_lr_scheduler.step()

        # Evaluate performance

        with torch.no_grad():
        
            n_correct = 0
            n_samples = 0
        
            for i, (rgb_image, seg_image, label, _, _, _, _) in enumerate(val_data_loader):
            
                rgb_image = rgb_image.type(torch.float32).to(device)
                seg_image = seg_image.type(torch.float32).to(device)

                label = label[0].to(device)

                # Forward pass
                model.eval()
                _, outputs = model(rgb_image, seg_image)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
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

    print('End to end training epoch:' + str(num_epochs))
    print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
    print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
    print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
    print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
    print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
    print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
    print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
    print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
    print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
