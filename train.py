
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torch import optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision as tv
import random
import math
import time
from datetime import datetime
import os
import argparse
import numpy as np

from utils.utils import *
from Models.DFnet import *

    
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--model_name', type=str, default='MIP')

parser.add_argument("--input_channel", type=int, default=169, help="input channel")
parser.add_argument("--output_channel", type=int, default=1, help="output channel")
parser.add_argument('--LF_shape', type=int, default=list([169,153,153]))
parser.add_argument('--LF_path', default='./Datasets/train_LF/')
parser.add_argument('--HR_path', default='./Datasets/train_HR/')

parser.add_argument('--index_range', type=int, default=list([0,30]))
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--get_full_imgs', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=5)

parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--model_path', default='')
parser.add_argument('--outputPath', type=str, default='./runs/')
args =  parser.parse_args()
 
# Create output folder
today = datetime.now()
comment = today.strftime('%Y_%m_%d__%H:%M:%S') + "_"+ str(args.model_name) +"_" + \
    str(args.batch_size)+"BS_" + 'realdata' + "_commit__"
save_folder = args.outputPath +"/" + comment
print(save_folder)
writer = SummaryWriter(log_dir=save_folder)

# load datasets
dataset_train = Dataset(args.LF_shape, args.index_range, args.image_size, args.get_full_imgs, args.LF_path, args.HR_path)
train_loader = torch.utils.data.DataLoader(dataset_train, num_workers=8, batch_size=args.batch_size, shuffle=True)
net = UNet(args.input_channel, args.output_channel)
net.to(args.device)
optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
epoch_state = 0
scheduler._step_count = epoch_state

if args.load_pretrain:
    if os.path.isfile(args.model_path):
        model = torch.load(args.model_path, map_location={'cuda:1': args.device})
        net.load_state_dict(model['model_state_dict'], strict=False)
        optimizer.load_state_dict(model['optimizer_state_dict'])
        epoch_state = model["epoch"]
        print("load pre-train at epoch {}".format(epoch_state))
    else:
        print("=> no model found at '{}'".format(args.load_model))
        
# parameter
criterion_Loss = nn.MSELoss().to(args.device)

# training
for idx_epoch in range(epoch_state, args.n_epochs):
    loss_epoch = []
    psnr_epoch = []
    ssim_epoch = []

    print('Training')
    for idx_iter, (image, mip_gt) in enumerate(train_loader):
    
        image, mip_gt = Variable(image).to(args.device), Variable(mip_gt).to(args.device)
        mip_pre = net(image)
       
        loss = criterion_Loss(mip_gt, mip_pre) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute psnr and ssim
        lossMSE = nn.functional.mse_loss(mip_gt.detach(), mip_pre.detach())
        PSNR = 10 * math.log10(1 / lossMSE.item())
        
        SSIM = ssim(mip_gt.contiguous().detach(), mip_pre.contiguous().detach()).sum()

        loss_epoch.append(loss.data.cpu())
        psnr_epoch.append(PSNR)
        ssim_epoch.append(SSIM)

        result_inter = "epoch %d, integration %d, loss %f, psnr %f, ssim %f" \
            % (idx_epoch+1, idx_iter+1, loss.item(), PSNR, SSIM)
   
        print(result_inter)

    if idx_epoch % 1 == 0:
        
        result_epoch = "Epoch----%5d, loss---%f, psnr---%f, ssim---%f" \
            % (idx_epoch + 1, float(np.array(loss_epoch).mean()), \
            float(np.array(torch.tensor(psnr_epoch).cpu()).mean()), float(np.array(torch.tensor(ssim_epoch).cpu()).mean()))

        print(result_epoch)
        txtfile = open(save_folder+'training.txt', 'a')
        txtfile.write(time.ctime()[4:-5] + result_epoch + '\n')
        txtfile.close()
        
        writer.add_image('input/input', cmap[((image[0, 85, :, :].cpu().clamp(0,1))*255).long()].cpu(), dataformats='HWC', global_step=idx_epoch)
        writer.add_image('mip/gt', cmap[((mip_gt[0, 0, :, :].cpu().clamp(0,1))*255).long()].cpu(), dataformats='HWC', global_step=idx_epoch)
        writer.add_image('mip/pre', cmap[((mip_pre[0, 0, :, :].cpu().clamp(0,1))*255).long()].cpu(), dataformats='HWC', global_step=idx_epoch)
           
        writer.add_scalar('loss', float(np.array(torch.tensor(loss_epoch).cpu()).mean()), global_step=idx_epoch)
        writer.add_scalar('mip/psnr', float(np.array(torch.tensor(psnr_epoch).cpu()).mean()), global_step=idx_epoch)
        writer.add_scalar('mip/ssim', float(np.array(torch.tensor(ssim_epoch).cpu()).mean()), global_step=idx_epoch)
        writer.flush()
    
    if idx_epoch % 50 == 0:
        torch.save({
        'epoch': idx_epoch,
        'args' : args,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        save_folder + '/model_'+str(idx_epoch)+ '.pth.tar')