import os
import re
from typing import Dict, Union
from io import BytesIO
import random
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision
import torch
import msgpack
from argparse import Namespace
import json
import yaml

import numpy as np
import math
class MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str,
        target_mapping: Dict[str, int],
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        meta_path=None,
        cache_size=6 * 4096,
        lat_key="LAT",
        lon_key="LON",
        scene_key="S3_Label",
    ):

        super(MsgPackIterableDatasetMultiTargetWithDynLabels, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.meta_path = meta_path
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.meta = self.meta.astype({lat_key: "float32", lon_key: "float32", scene_key: "float32"})
            self.lat_key = lat_key
            self.lon_key = lon_key
            self.scene_key = scene_key # indoor - 0, natural - 1, urban - 2

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary

        img = Image.open(BytesIO(x[self.key_img_encoded]))

        if img.mode != "RGB":
            img = img.convert("RGB")

        # apply all user specified image transformations
        img = self.transformation(img)        

        if self.meta_path is None:
            return img, x["target"]
        else:
            _id = x[self.key_img_id].decode("utf-8")
            
            # load semantically segmented image
            # img_segmented = Image.open(semantic_seg_path '/' + _id)
        meta = self.meta.loc[_id]

        return img, _id, x["target"], meta[self.lat_key], meta[self.lon_key], meta[self.scene_key]
        # return img, img_segmented, x["target"], meta[self.lat_key], meta[self.lon_key], meta[self.scene_key]

    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    _id = x[self.key_img_id].decode("utf-8") # name of the image file
                    try:
                        # set target value dynamically
                        

                        if len(self.target_mapping[_id]) == 1:
                            x["target"] = self.target_mapping[_id][0]
                        else:
                            x["target"] = self.target_mapping[_id]
                    except KeyError:
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length

class MultiPartitioningClassifier():
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

    def train_dataloader(self):

        with open(self.hparams.train_label_mapping, "r") as f:
            target_mapping = json.load(f) # dictionary - name of the image and 
                                          # three labels: S2 class_label for 
                                          # 50_5000, 50_2000, 50_1000 - set based on latitude and longitude
            
        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        # dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
        #     path=self.hparams.msgpack_train_dir,
        #     target_mapping=target_mapping,
        #     key_img_id=self.hparams.key_img_id,
        #     key_img_encoded=self.hparams.key_img_encoded,
        #     shuffle=True,
        #     transformation=tfm,
        # )

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_train_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=True,
            transformation=tfm,
            meta_path=self.hparams.train_meta_path,
            cache_size=1024,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )
        return dataset, dataloader

    def val_dataloader(self):

        with open(self.hparams.val_label_mapping, "r") as f:
            target_mapping = json.load(f) # length 25600

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # mean , std
                ),
            ]
        )
        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_val_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=False,
            transformation=tfm,
            meta_path=self.hparams.val_meta_path,
            cache_size=1024,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )
        
        return dataset, dataloader

def main():
    with open('config/baseM_Ewa.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
   
    tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

    train_dataset, train_data_loader = tmp_model.train_dataloader() #  4,654,532 images

    val_dataset, val_data_loader = tmp_model.val_dataloader() # val_dataset: 25,600 images but val_data_loader: 200 images

    save_folder = '/export/r16/data/enowara/GLIMPSE_datasets/yfcc25600/yfcc25600_jpg_images/'
    # save_folder = '/export/r16/data/enowara/GLIMPSE_datasets/mp16/mp16_jpg_images/'

    # iterate through the dataset
    # this will give it only 200 samples, how can I make val_data_loader and val_dataset same length for now

    # it = iter(train_data_loader)
    # for k in range(math.ceil(len(train_dataset) / model_params['batch_size'])):
    #     print('processing batch ' + str(k) + ' out of ' + str(math.ceil(len(train_dataset) / model_params['batch_size'])))

    it = iter(val_data_loader)
    for k in range(math.ceil(len(val_dataset) / model_params['batch_size'])):
        print('processing batch ' + str(k) + ' out of ' + str(math.ceil(len(val_dataset) / model_params['batch_size'])))
            
        first_batch = next(it)

        # print(first_batch)
        image_it = first_batch[0] # image
        image_name_it = first_batch[1]

        # S2_labels_it = first_batch[1] # label
        # lat_it = first_batch[2]
        # lon_it = first_batch[3]
        # scene_it = first_batch[4]

        ############### get the image name ################  

        ############### save image ################  
        for batch_index in range(image_it.shape[0]):
            # if image not saved yet
            image_name = image_name_it[batch_index]
            # print('save_folder')
            # print(save_folder)
            # print('image_name')
            # print(image_name)
            if os.path.exists(save_folder + image_name) == 0:
                print('processing: ' + save_folder + image_name)
                # batch_index = 0
                img1 = image_it[batch_index, :,:,:]
                img1 = img1.numpy()  
                img1_0 = np.moveaxis(img1, 0, -1) # reshape from 3 x 224 x 224 to 224 x 224 x 3

                # undo normalization transform:
                # add the mean back and multiply by std
                # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # mean , std
                m1 = 0.485
                s1 = 0.229
                m2 = 0.456 
                s2 = 0.224
                m3 = 0.406
                s3 = 0.225

                # img1_0 = img1 
                img1_0[:,:,0] = img1_0[:,:,0] + (m1 / s1)
                img1_0[:,:,1] = img1_0[:,:,1] + (m2 / s2)
                img1_0[:,:,2] = img1_0[:,:,2] + (m3 / s3)
                # multiply by std
                img1_0[:,:,0] = img1_0[:,:,0] * s1
                img1_0[:,:,1] = img1_0[:,:,1] * s2
                img1_0[:,:,2] = img1_0[:,:,2] * s3

                PIL_image1 = Image.fromarray(np.uint8(img1_0 * 255)).convert('RGB')

                image_path = os.path.dirname(image_name)

                if os.path.exists(save_folder + image_path) == 0:
                    os.makedirs(save_folder + image_path)

                PIL_image1.save(save_folder + image_name)
            # else:
                # print('image exists: yfcc25600_jpg_images/' + image_name)

            ############### print label ################  

            # print('len(first1)') # list of 3 x 128 - labels for each of 128 images: 
            # print(first1)

            # first1_0 = first1[0]
            # first1_1 = first1[1]
            # first1_2 = first1[2]

            # print('class label geo cells_50_5000') 
            # print(first1_0[1])
            # print('class label geo cells_50_2000') 
            # print(first1_1[1])
            # print('class label geo cells_50_1000') 
            # print(first1_2[1])

    # # ms_class = MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset)
    # # ms_class._process_sample()

if __name__ == "__main__":
    main()