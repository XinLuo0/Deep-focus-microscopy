from PIL import Image
import numpy as np
import torch
import imageio
import h5py
import os
import math
import random
from skimage import metrics
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

cmap = [[0 ,0 ,0],
 [0,0,255],
 [255,0,255],
 [255,0,0],
 [255,255,0],
 [255,255,255]]
cmap = torch.tensor(cmap,dtype=torch.float32).unsqueeze(0).unsqueeze(1)
cmap = torch.nn.functional.interpolate(cmap, size=[256,3],mode='bilinear',align_corners=True)
cmap = cmap.squeeze()
cmap = cmap.type(torch.uint8)


def MacPI2SAI(x, angRes):
    c,H,W = x.shape
    h = int(H//angRes)
    w = int(W//angRes)
    
    MPI = x.reshape(c,h,angRes,w,angRes)
    SAI = MPI.permute(0,2,4,1,3)
    SAI = SAI.reshape(angRes*angRes*c,h,w)
    return SAI


def get_files(root_path, index):
    filenames = sorted(os.listdir(root_path), key=lambda x: (x[:-4]))
    filenames = filenames[index[0]:index[1]]

    files = []
    files = [root_path + filenames[i] for i in range(len(filenames))]
    
    return files


def read_files(path):
    img = Image.open(path)
    
    while True:
        try:
            img.seek(img.tell() + 1)
        except:
            frames = img.tell() + 1
            break
    img_tif = np.zeros((frames,) + img.size)
    
    for i in range(frames):
        img.seek(i)
        img_tif[i, :, :] = np.array(img)
    return img_tif


def imwrite3dtiff(npimg, name):
    npimg = np.array(npimg,dtype='uint16')

    frames = [Image.fromarray(frame) for frame in npimg]

    frames[0].save(name, compression="tiff_deflate", save_all=True,

                   append_images=frames[1:])

    return



def get_data_input(data_path, num_range):
    file = get_files(data_path, num_range)  
    input_dataset = torch.zeros([len(file),169,153,153])
    for index in range(len(file)):
        LF = torch.tensor(read_files(file[index]), dtype=torch.float32)
        LF = MacPI2SAI(LF, 13)
        LF = LF / LF.max()
        input_dataset[index,:,:,:] = LF
    return input_dataset   


def get_data_label(data_path, num_range):
    file = get_files(data_path, num_range)  
    label_dataset = torch.zeros([len(file),1,1989,1989])
    for index in range(len(file)):
        value = torch.tensor(read_files(file[index]), dtype=torch.float32)
        value = value  / (np.percentile(value, 99.8) + 1e-6)  
        label_dataset[index,:,:,:] = value
    return label_dataset


class Dataset(object):
    def __init__(self, LF_shape, index_range, image_size, get_full_imgs, LF_path, HR_path):
        
        self.LF = get_data_input(LF_path, index_range)
        self.mip = get_data_label(HR_path, index_range)

        self.get_full_imgs = get_full_imgs
        self.image_size = image_size
        self.LF_shape = LF_shape
        self.window_range = LF_shape[1] - image_size 
       
        self.lenlet_num = 13
        if self.get_full_imgs:
            self.nPatchesPerImg = 1
        else:
            self.nPatchesPerImg = 5
        self.nPatches = self.nPatchesPerImg * (index_range[1] - index_range[0])
        

    def __len__(self):
        return self.nPatches

    def __getitem__(self, index):
      
        nImg = index // self.nPatchesPerImg
        LF = self.LF[nImg,:,:,:]
        mip = self.mip[nImg,:,:,:]

        yLF = random.randint(0, self.window_range)
        xLF = random.randint(0, self.window_range)

        input = LF[:,yLF:yLF+self.image_size, xLF:xLF+self.image_size]
        mip = mip[:,yLF*self.lenlet_num : (yLF+self.image_size)*self.lenlet_num, \
                xLF*self.lenlet_num : (xLF+self.image_size)*self.lenlet_num]


        return input, mip



######## SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

