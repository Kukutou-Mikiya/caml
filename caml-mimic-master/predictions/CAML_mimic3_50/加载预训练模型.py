import torch
pthfile = r'C:\Users\Veid\Desktop\caml\caml-mimic-master\predictions\CAML_mimic3_50\model.pth'
net = torch.load(pthfile)
print(net)