# from  __future__ import absolute_import
from utils.flow_lib.io import read_flow
from utils.flow_lib.img import warp_image_with_flow2
from utils.flow_lib.img import warp_image_with_flow
from utils.flow_lib.img import warp_image
from utils.flow_lib.visualize import visualize_img
import torch
from torch.autograd import Variable

flo = read_flow("/home/lxt/Github/flownet_pytorch/demo.flo")

im1_path = "/home/lxt/data/CityScapes/leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.png"
im2_path = "/home/lxt/data/CityScapes/leftImg8bit_sequence/train/aachen/aachen_000000_000001_leftImg8bit.png"

import imageio


im1 = imageio.imread(im1_path)
print(im1.shape)

im1 = Variable(torch.FloatTensor(im1)).cuda()
flo = Variable(torch.FloatTensor(flo)).cuda()
im1 = im1.permute(2,0,1)
flo = flo.permute(2,0,1)
im1 = im1.unsqueeze(0)
flo = flo.unsqueeze(0)
print(im1.size(),flo.size())

im2 = warp_image_with_flow(im1, flo)

print("warped image",im2.shape)

im2 = im2.cpu().data.squeeze().permute(1,2,0).numpy()

print(im2.shape)

visualize_img(im2)
