import torch
import torch.nn as nn
import torch.nn.functional as func
import timm
import numpy as np
import random, os
from octa500 import Octa500Dataset3D, Octa500Dataset2D
from settings import parse_opts
from improve_loss import MedianTripletHead, unifyloss, HCLloss
from model import UniformerS1
import time
from torch import optim
import logging
from torchvision import transforms
from PIL import Image
from functools import partial
import pvt_v2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(data_loader, model1, model2, optimizer1, scheduler1, optimizer2, scheduler2, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
    log = logging.getLogger()
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss = HCLloss() #用自己的tripet loss

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss = loss.cuda()
    
        
    model1.train()
    model2.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        log.info('lr1 = {}'.format(scheduler1.get_last_lr()))
        log.info('lr2 = {}'.format(scheduler2.get_last_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volume1, volume2, _ = batch_data
            if len(volume1.shape)==4:
                batch_size = sets.batchsize2
            else:
                batch_size = sets.batchsize3
            volume1, volume2 = volume1.cuda(), volume2.cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            logits1 = model1(volume1)
            logits2 = model2(volume2)

            # calculating loss
            loss_value = HCLloss(logits1, logits2, tau_plus=0.1, batch_size=batch_size, beta=1, estimator='easy')
            loss_v = loss_value
            loss_value.backward()                
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss_v.item(), avg_batch_time))
          
            # save model
            if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                torch.save({
                            'ecpoch': epoch,
                            'batch_id': batch_id,
                            'state_dictt': model1.state_dict(),
                            'optimizert': optimizer1.state_dict(),
                            'state_dicts': model2.state_dict(),
                            'optimizers': optimizer2.state_dict()},
                            model_save_path)
                            
    print('Finished training')            



if __name__ == '__main__':
    setup_seed(10)
    sets = parse_opts()

    torch.cuda.set_device(sets.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    train_dataset2 = Octa500Dataset2D(root_dir=sets.root2D,
                                      img_list=sets.train2D,
                                      sets=sets)
    val_dataset2 = Octa500Dataset2D(root_dir=sets.root2D,
                                      img_list=sets.val2D,
                                      sets=sets)
    train_dataset3 = Octa500Dataset3D(root_dir=sets.root3D,
                                    img_list=sets.train3D,
                                    sets=sets)
    val_dataset3 = Octa500Dataset3D(root_dir=sets.root3D,
                                    img_list=sets.val3D,
                                    sets=sets)
    train_loader2 = torch.utils.data.DataLoader(
        dataset=train_dataset2,
        batch_size=sets.batchsize2,
        shuffle = True,
        num_workers = sets.numworkers
        )
    val_loader2 = torch.utils.data.DataLoader(
            val_dataset2,
            batch_size=sets.batchsize2,
            shuffle = False,
            num_workers = sets.numworkers
        )
    train_loader3 = torch.utils.data.DataLoader(
        dataset=train_dataset3,
        batch_size=sets.batchsize3,
        shuffle = True,
        num_workers = sets.numworkers
        )
    val_loader3 = torch.utils.data.DataLoader(
            val_dataset3,
            batch_size=sets.batchsize3,
            shuffle = False,
            num_workers = sets.numworkers
        )

    # model1, model2 = UniformerS1(), UniformerS1()
    model1, model2 = pvt_v2.pvt_v2_b1_23D(num_classes=0), pvt_v2.pvt_v2_b1_23D(num_classes=0)
    save_model = torch.load('/mnt/caizy/MICCAI2022/checkpoint/pvt_v2_b1.pth')
    model_dict = model1.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model1.load_state_dict(model_dict)
    model2.load_state_dict(model_dict)
    # model1 = convert_syncbn_model(model1)
    # model2 = convert_syncbn_model(model2)
    model1, model2 = model1.cuda(), model2.cuda()
    model1 = torch.nn.parallel.DistributedDataParallel(model1, find_unused_parameters=True)
    model2 = torch.nn.parallel.DistributedDataParallel(model2, find_unused_parameters=True)
    optimizer1 = torch.optim.SGD(model1.parameters(), momentum=0.9, weight_decay=1e-3, lr=sets.init_lr) 
    optimizer2 = torch.optim.SGD(model2.parameters(), momentum=0.9, weight_decay=1e-3, lr=sets.init_lr)   
    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.99)
    sets.phase = 'train'
    # train(train_loader2, model1, model2, optimizer1, scheduler1, optimizer2, scheduler2, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)
    train(train_loader3, model1, model2, optimizer1, scheduler1, optimizer2, scheduler2, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)