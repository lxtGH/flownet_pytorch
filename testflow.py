from  __future__ import absolute_import
#from utils.flow_lib import read_flow
from utils.flow_lib.io import read_flow
flo = read_flow("/home/lxt/Github/flownet_pytorch/demo.flo")
print(flo.shape)