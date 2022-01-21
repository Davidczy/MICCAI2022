'''
Configs for training & testing
Written by Caizhiyuan
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int)
    parser.add_argument(
        '--root2D',
        default='/mnt/caizy/OCTA500_2D_cls/OCT(FULL)',
        type=str)
    parser.add_argument(
        '--root3D',
        default='/mnt/caizy/MedicalNet_classification/data',
        type=str)
    parser.add_argument(
        '--train2D',
        default='/mnt/caizy/OCTA500_2D_cls/train.txt',
        type=str)
    parser.add_argument(
        '--val2D',
        default='/mnt/caizy/OCTA500_2D_cls/val.txt',
        type=str)
    parser.add_argument(
        '--train3D',
        default='/mnt/caizy/MedicalNet_classification/data/OCTA500_3M/train.txt',
        type=str)
    parser.add_argument(
        '--val3D',
        default='/mnt/caizy/MedicalNet_classification/data/OCTA500_3M/val.txt',
        type=str)
    parser.add_argument(
        '--batchsize2',
        default=8,
        type=int)
    parser.add_argument(
        '--batchsize3',
        default=1,
        type=int)
    parser.add_argument(
        '--numworkers',
        default=4,
        type=int)
    parser.add_argument(
        '--phase',
        default='train',
        type=str)
    parser.add_argument(
        '--init_lr',
        default=0.01,
        type=float)  
    parser.add_argument(
        '--n_epochs',
        default=50,
        type=int)
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int)
    parser.add_argument(
        '--save_folder',
        default=' ',
        type=str)
    parser.add_argument(
        '--model',
        default='swin',
        type=str)
    parser.add_argument(
        '--no_cuda',
        default='False',
        type=bool)
    parser.add_argument(
        '--input_D',
    default=128,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=112,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=112,
        type=int,
        help='Input size of width')
    args = parser.parse_args()
    args.save_folder = "./trails/HCLloss/{}".format(args.model)
    
    return args
