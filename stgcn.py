import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import torch
from torch import nn
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader

class STBlock(nn.Module):
  def __init__(self, in_features, out_features, len_A, t_kernel=9, t_stride=1):
    super(STBlock, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    padding = ((t_kernel - 1) // 2, 0)
    
    # self.spatial_conv = nn.ModuleList((nn.Conv2d(in_features, out_features, kernel_size=1) for _ in range(len_A)))
    self.spatial_conv = nn.Conv2d(in_features, out_features * len_A, kernel_size=1)
    self.temporal_conv = nn.Conv2d(out_features, out_features, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=padding)
    
  def forward(self, input, adj):
    """
    Args:
      input(Tensor): graph feature
                     input.size() = (N, C, T, V)
      adj(Tensor) : normalized adjacency matrix
                    adj.size() = (V, V)
    Returns:
      Tensor: output.size() = (N, C_out, T, V)
    """
    # spatial_feature = None
    # for module, a in zip(self.spatial_conv, adj):
    #   XW = module(input)
    #   DADXW = torch.einsum("NCTV, VW -> NCTW", XW, a)
    #   if spatial_feature is not None:
    #     spatial_feature = spatial_feature + DADXW
    #   else:
    #     spatial_feature = DADXW
    
    XW = self.spatial_conv(input)
    XW = rearrange(XW, "N (C K) T V -> N C K T V", C=self.out_features)
    spatial_feature = torch.einsum("NCKTV, KVW -> NCTW", XW, adj)
    temporal_feature = self.temporal_conv(spatial_feature)
    return temporal_feature
  
def edge2mat(E, num_node):
  A = np.zeros((num_node, num_node))
  for i, j in E:
    A[j, i] = 1
  return A

def get_DAD(A):
  d_ii = np.sum(A, axis=0) + 0.001
  D = np.zeros_like(A)
  for i in range(len(A)):
    D[i, i] = d_ii[i] ** (-0.5)
  DAD = np.dot(np.dot(D, A), D)
  return DAD

def uni_labeling(inward, outward, self_loop, num_node):
  A = np.zeros((1, num_node, num_node))
  A[0] = get_DAD(edge2mat(inward+outward+self_loop, num_node))
  return A

def distance(inward, outward, self_loop, num_node):
  A = np.zeros((2, num_node, num_node))
  A[0] = get_DAD(edge2mat(self_loop, num_node))
  A[1] = get_DAD(edge2mat(inward+outward, num_node))
  return A

def spatial(inward, outward, self_loop, num_node):
  A = np.zeros((3, num_node, num_node))
  A[0] = get_DAD(edge2mat(self_loop, num_node))
  A[1] = get_DAD(edge2mat(inward, num_node))
  A[2] = get_DAD(edge2mat(outward, num_node))
  return A

num_node = 17
in_channels = 2
out_channels = 8

E = [[15,13],[13,11],[16,14],[14,12],[11,12],[5,11],
     [6,12],[5,6],[5,7], [6,8],[7,9],[8,10],[1,2],
     [0,1],[0,2],[1,3],[2,4],[3,5],[4,6]]
reversed_E = [[j, i] for i, j in E]
I = [[i, i] for i in range(num_node)]

inward = E
outward = reversed_E
self_loop = I

X = np.random.rand(10, 2, 10, 17)
y = [0] * 5 + [1] * 5

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int64)
data = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(data, batch_size=10)

A = spatial(inward, outward, self_loop, num_node)
A = torch.tensor(A, dtype=torch.float32)

model = STBlock(in_channels, out_channels, len_A=len(A))

for input, label in loader:
  new_X_tensor = model(input, A)

print(new_X_tensor.size())