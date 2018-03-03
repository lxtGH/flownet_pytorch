import imageio

import numpy as np
from torch.autograd import Variable
import torch

from models.FlowNetS import flownets

im1_path = "/home/lxt/data/CityScapes/leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.png"
im2_path = "/home/lxt/data/CityScapes/leftImg8bit_sequence/train/aachen/aachen_000000_000001_leftImg8bit.png"



flownet = flownets()
flownet.cuda()

im1 = imageio.imread(im1_path)
im2 = imageio.imread(im2_path)
print("each image shape: ",im1.shape)
ims = np.concatenate([im1,im2],axis=2)
print("concatenate shape: ",ims.shape)
ims = torch.from_numpy(ims)
ims = ims.unsqueeze(0)
ims = ims.permute(0,3,1,2)
ims = ims.float()
print(ims.size())
input = Variable(ims).cuda()

print("input size: ",input.size())
predict = flownet(input)

print(predict)
