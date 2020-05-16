import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable
CUDA_VISIBLE_DEVICES="0"
target=Variable(torch.randn(8,50))

U = nn.Linear(400, 50)
U = U.cuda()
xavier_uniform_(U.weight)
final = nn.Linear(400, 50)
final = final.cuda()
xavier_uniform_(final.weight)
x = Variable(torch.randn(8,123,400))
x = x.cuda()
target = target.cuda()
'''
# The code snippet that works successfully
alpha = F.softmax(U.weight.matmul(x.transpose(1,2)), dim=2)
m = alpha.matmul(x)
y = final.weight.mul(m).sum(dim=2).add(final.bias)
'''

#The code snippet that fails to autograd
x= F.max_pool1d(x.transpose(1,2), kernel_size=x.size()[1])
alpha = U.weight.mul(x.transpose(1,2)) 
y = final.weight.mul(alpha).sum(dim=2).add(final.bias)

yhat= y
loss = F.binary_cross_entropy_with_logits(yhat, target)
loss.backward()
print(y.size())