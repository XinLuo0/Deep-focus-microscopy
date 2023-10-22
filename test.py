import argparse
import os
from PIL import Image
import time
import numpy as np
import torch
from torchvision import transforms


from utils.utils import *
from Models.DFnet import *


def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=" ")
    parser.add_argument("--input_channel", type=int, default=169, help="input channel")
    parser.add_argument("--output_channel", type=int, default=1, help="output channel")
    parser.add_argument('--savefolder', default="./") 
    parser.add_argument('--model_path', default='./Models/pretrained_models_our_dataset/epoch-last.pth.tar')
    parser.add_argument('--resolution', default='1,1989,1989')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    
    files = ['G:/xianwei/Paper/Code/Data/LF/LF.tif']

   
    net = UNet(args.input_channel, args.output_channel)
    net.to(args.device)
    if os.path.isfile(args.model_path):
        model = torch.load(args.model_path, map_location={'cuda:0': args.device})
        net.load_state_dict(model['model_state_dict'])
    else:
        print("=> no model found at '{}'".format(args.model_path))


    for file in files:
        print(file)
        args.input = file
        lfstack = torch.tensor(read_files(args.input), dtype=torch.float32)
        lfstack = MacPI2SAI(lfstack, 13)
        inp_all = lfstack.unsqueeze(0)
        inp_all = inp_all / inp_all.max()
    
        t0 = time.time()
        
        with torch.no_grad():
            print(inp_all.shape)
            pred = net( ((inp_all - 0) / 1).to('cuda:0'))
            
        pred[pred<0] = 0
        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0
        pred = pred.squeeze(0).cpu()

        print(time.time()-t0)
        print(pred.shape)
       
        imwrite3dtiff(pred *2000, args.savefolder + args.input[-args.input[-1::-1].find('/'):][0:-4]+'_mip.tif')
      
        