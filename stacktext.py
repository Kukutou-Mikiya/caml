import torch
y=[[torch.randn(1),torch.randn(1)],[torch.randn(1),torch.randn(1)],[torch.randn(1),torch.randn(1)]]
print(y)
y=torch.stack(y)
print(y)