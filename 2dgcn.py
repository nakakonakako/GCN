import numpy as np
from matplotlib import pyplot as plt 
import torch
from torch import nn
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader

num_node = 5
in_channels = 4
out_channels = 8

class GraphConv(nn.Module):
  def __init__(self, in_features, out_features):
    super(GraphConv, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
  def forward(self, input, adj):
    """
    Args:
      input(Tensor): graph feature
                     input.size() = (N, V, C)
      adj(Tensor): normalized adjacency matrix
                   e.g. DAD or DA
                   input.size() = (V, V)
      
    Returns:
      Tensor: out.size() = (N, V, C_out)
    """
    input = rearrange(input, "N V C -> N C 1 V")
    XW = self.conv(input)
    DADXW = torch.einsum("NCTV, VW->NCTW", XW, adj)
    DADXW = rearrange(DADXW, "N C 1 V -> N V C")
    return DADXW

def edge2mat(E, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in E:
        A[j, i] = 1
    return A

def get_D(A, pow=-1):
    d_ii = np.sum(A, 0)
    D = np.zeros_like(A)
    for i in range(len(A)):
        D[i, i] = d_ii[i]**(pow)
    return D

X = np.random.randn(10, num_node, in_channels)
y = [0] * 5 + [1] * 5
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int64)
data = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(data, batch_size=4)

E = [[0, 1], [0, 2], [0, 4], [1, 2], [2, 3]]
reversed_E = [[j, i] for i, j in E]
E = E + reversed_E
I = [[i, i] for i in range(num_node)]
A_tilde = edge2mat(E + I, num_node)
D_tilde = get_D(A_tilde)

A_tensor = torch.tensor(A_tilde, dtype=torch.float32)
D_tensor = torch.tensor(D_tilde, dtype=torch.float32)

DAD = D_tensor @ A_tensor @ D_tensor

model = GraphConv(in_channels, out_channels)

for input, label in loader:
  new_X_tensor = model(input, DAD)
  new_X = new_X_tensor.detach().numpy()
  
fig, ax = plt.subplots(1, 2, width_ratios=[4, 8])
ax[0].pcolor(X[0], cmap=plt.cm.Blues)
ax[0].set_aspect("equal", "box")
ax[0].set_title("X", fontsize=10)
ax[0].invert_yaxis()

ax[1].pcolor(new_X[0], cmap=plt.cm.Blues)
ax[1].set_aspect("equal", "box")
ax[1].set_title("new_X", fontsize=10)
ax[1].invert_yaxis()

plt.show()