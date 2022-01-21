'''
Dataset for training
Written by Whalechen
Revised by ZhiyuanCai
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from torchvision import transforms
import cv2
from PIL import Image
from volumentations import *

def get_augmentation(patch_size):
    return Compose([
        # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        # RandomCropFromBorders(crop_value=0.1, p=0.5),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        # RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
    #     RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)


class Octa500Dataset3D(Dataset):
    def __init__(self, root_dir, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f] #以空格读取数据地址和对应标签
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            label = ith_info[1]
            # print(img_name)
            aug = get_augmentation((self.input_D, self.input_H, self.input_W))
            assert os.path.isfile(img_name)
            # assert os.path.isfile(label)
            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            assert label is not None
            img = img.get_data()
            img = np.array(img)
            data = {'image': img}
            aug_data1 = aug(**data)
            aug_data2 = aug(**data)
            img1 = aug_data1['image']
            img2 = aug_data2['image']
            # data processing
            img_array1 = self.__training_data_process__(img1)
            img_array2 = self.__training_data_process__(img2)

            # 2 tensor array
            img_array1 = self.__nii2tensorarray__(img_array1)
            img_array2 = self.__nii2tensorarray__(img_array2)

            return img_array1, img_array2, label
        
        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            print(img_name)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None
            img = np.array(img)
            print('**************************************************')
            # data processing
            # img_array = self.__testing_data_process__(img)

            # 2 tensor array
            # img_array = self.__nii2tensorarray__(img_array)

            return img
            

    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label>0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())
        
        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))
       
        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])
 
        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)
        
        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data = self.__random_center_crop__ (data)
        
        return data

    def __training_data_process__(self, data): 
        # crop data according net input size
        # data = data.get_data()
        
        # drop out the invalid range
        data = self.__drop_invalid_range__(data)
        
        # crop data
        # data = self.__crop_data__(data) 

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


    def __testing_data_process__(self, data): 
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


class Octa500Dataset2D(Dataset):
    def __init__(self, root_dir, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f] #以空格读取数据地址和对应标签
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.phase = sets.phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            label = int(ith_info[1])
            # print(img_name)
            assert os.path.isfile(img_name)
            # assert os.path.isfile(label)
            img = cv2.imread(img_name)[:, :, ::-1]# We have transposed the data from WHD format to DHW
            # img = img.transpose(2, 0, 1)
            T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            ])
            img = img.copy()
            # print(img.shape)
            # img = Image.fromarray(img)
            img1 = T(img)
            img2 = T(img)
            # print(torch.max(img))
            # img = img /255.
            assert img is not None
            assert label is not None
            
            return img1, img2, label
        
        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            # print(img_name)
            assert os.path.isfile(img_name)
            img = cv2.imread(img_name)[:, :, ::-1]
            img = img.copy()
            # img = img.transpose(2, 0, 1)
            img = transform(img)
            # img = img /255.
            assert img is not None
            # img = img.copy()
            return img