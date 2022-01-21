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
import argparse
import torch
import torch.nn as nn
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
import timm
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tensorboardX import SummaryWriter
from octa500 import Octa500Dataset3D
from model import UniformerS1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.encoder = UniformerS1()
        state = torch.load('/mnt/caizy/MICCAI2022/trails/OCTA500_3M_OCTA/swin_epoch_96_batch_0.pth.tar')
        self.encoder.load_state_dict(state['state_dicts'])
        self.clshead = nn.Linear(in_features=768, out_features=2)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.clshead(x)
        return x

class Octa500_2DDataset(Dataset):
    def __init__(self, root_dir, img_list, phase):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f] #以空格读取数据地址和对应标签
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.phase = phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
        ])
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
            img = img.copy()
            img = transform(img)
            # print(torch.max(img))
            # img = img /255.
            assert img is not None
            assert label is not None
            
            return img, label
        
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


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval, writer):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_imgs = data[0]
            labels = data[1]
            # print(type(labels))
            fundus_imgs = fundus_imgs.to(0)
            labels = labels.to(0)
            optimizer.zero_grad()
            logits = model(fundus_imgs)
            loss = criterion(logits, labels)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().detach().numpy()):
                avg_kappa_list.append([p, l])
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            avg_loss_list.append(loss.cpu().detach().numpy())
            
            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1])
                avg_loss_list = []
                avg_kappa_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))

            if iter % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                writer.add_scalar("/mnt/caizy/OCTA500_2D_cls/log/resnet50pre1e-4/", avg_kappa, iter)
                print("[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))
                if avg_kappa > best_kappa:
                    best_kappa = avg_kappa
                    os.mkdir("3Dbest_model_{:.4f}".format(best_kappa))
                    torch.save(model.state_dict(),
                            os.path.join("3Dbest_model_{:.4f}".format(best_kappa), 'model.pdparams'))
                model.train()
    writer.close()

def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        for data in val_dataloader:
            fundus_imgs = data[0]
            labels = data[1]
            fundus_imgs = fundus_imgs.to(0)
            labels = labels.to(0)
            logits = model(fundus_imgs)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().detach().numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            avg_loss_list.append(loss.cpu().detach().numpy())
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1])
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, kappa

writer = SummaryWriter("/mnt/caizy/OCTA500_2D_cls/log")
batchsize = 16
# image_size = 256
iters = 10000 # For demonstration purposes only, far from reaching convergence
val_ratio = 0.2
# trainset_root = "./OCT(FULL)"
trainset_root = '/mnt/caizy/OCTA500_2D_cls/OCT(FULL)'
num_workers = 0
init_lr = 1e-5
optimizer_type = "adam"

seed = 10
setup_seed(seed)
count = 0
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--input_D", default=128, type=int)
parser.add_argument("--input_H", default=112, type=int)
parser.add_argument("--input_W", default=112, type=int)
parser.add_argument("--phase", default='train', type=str)
args = parser.parse_args()


torch.cuda.set_device(args.local_rank)
print("Successfully cuda initilize")

i = 0
train_dataset = Octa500_2DDataset(root_dir=trainset_root,
                                  img_list='/mnt/caizy/OCTA500_2D_cls/train.txt',
                                  phase='train')
val_dataset = Octa500_2DDataset(root_dir=trainset_root,
                                  img_list='/mnt/caizy/OCTA500_2D_cls/val.txt',
                                  phase='train')

# train_dataset = Octa500Dataset3D(root_dir=trainset_root,
#                                   img_list='/mnt/caizy/MedicalNet_classification/data/OCTA500_3M/train.txt',
#                                   sets=args)
# val_dataset = Octa500Dataset3D(root_dir=trainset_root,
#                                   img_list='/mnt/caizy/MedicalNet_classification/data/OCTA500_3M/val.txt',
#                                   sets=args)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    shuffle = True,
    num_workers = num_workers
    # sampler = train_sampler
    )
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle = False,
    num_workers = num_workers
    # sampler = val_sampler
    )

model = mymodel()
# model.patch_embed = PatchEmbed3D()
# print(model)
# model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
# model = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=2)

model = convert_syncbn_model(model)
model = model.to(0)

print('Successfully initialize model')
if optimizer_type == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr
    )
print('op')
criterion = nn.CrossEntropyLoss().to(0)
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100, writer=writer)
# train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100, writer=None)
