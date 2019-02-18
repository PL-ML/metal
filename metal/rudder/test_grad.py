import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the leaf nodes
a = Variable(torch.FloatTensor([4]), requires_grad=True)
b = Variable(torch.FloatTensor([3]), requires_grad=True)
c = torch.cat([a, b], dim=0)


weights = [Variable(torch.FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]

# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

gru = nn.GRUCell(5, 2)
a = Variable(torch.ones(4, 3, 5), requires_grad=True)
s = Variable(torch.ones(3, 2), requires_grad=True)

# s = gru(a, s)
# s = gru(a, s)

s = gru(a[0], s)
s = gru(a[1], s)
s = gru(a[2], s)
s = gru(a[3], s)

s = torch.sum(s, dim=-1)

s_a_b1 = torch.autograd.grad(s[0], a, retain_graph=True)
s_a_b2 = torch.autograd.grad(s[1], a, retain_graph=True)
s_a_b3 = torch.autograd.grad(s[2], a, retain_graph=True)

print(s_a_b1)
print(s_a_b2)
print(s_a_b3)

s_a_b1 = torch.autograd.grad(s[0], a, retain_graph=True)
s_a_b2 = torch.autograd.grad(s[1], a, retain_graph=True)
s_a_b3 = torch.autograd.grad(s[2], a, retain_graph=True)

print('______________________-')
print(s_a_b1)
print(s_a_b2)
print(s_a_b3)


def f(fin):
    b = w1 * fin[0]
    c = w2 * fin[1]
    d = w3 * b + w4 * c
    L = (10 - d)
    return L

y1 = f(c)


y1_c = torch.autograd.grad(y1, c)
y1_b = torch.autograd.grad(y1, b)
y2_a = torch.autograd.grad(y2, a)
y2_b = torch.autograd.grad(y2, b)

print(y1_a, y2_b)
