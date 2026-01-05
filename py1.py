import torch.nn.functional as F #A
import torch

y = torch.tensor([1.0]) #B
x1 = torch.tensor([1.1]) #C
w1 = torch.tensor([2.2]) #D
b = torch.tensor([0.0]) #E
z = x1 * w1 + b #F
a = torch.sigmoid(z) #G
loss = F.binary_cross_entropy(a, y)