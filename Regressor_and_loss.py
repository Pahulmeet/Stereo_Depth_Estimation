import torch
import torch.nn as nn
import numpy as np

#https://github.com/jzhangbs/DSM/blob/master/psmnet/submodule.py
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out
    
    
def loss_l1func(new_im,target):
    loss = nn.SmoothL1Loss(beta=1.0)
    #target = torch.flatten(target)
    #new_im = torch.flatten(new_im)
    b_target = target>0
    target = target[b_target]
    new_im = new_im[b_target]
    output = loss(new_im, target)   # Here, the mean is taken by dividing with N which is only non-zero pixels
    return output

def loss_l1func2(new_im,target):
    loss = nn.SmoothL1Loss(beta=1.0,reduction='sum')
    #target = torch.flatten(target)
    #new_im = torch.flatten(new_im)
    b_target = target>0
    target = target[b_target]
    new_im = new_im[b_target]
    output = loss(new_im, target)
    output = ouput/131072    # 131072 is obtaiend from multiplying image dimensions 256*512 because here N is total number of pixels
    return output

def loss_3pfunc(new_im,target):
    loss = nn.SmoothL1Loss(beta=1.0)
    target = torch.flatten(target)
    new_im = torch.flatten(new_im)
    #new_im = new_im[0,:,:]
    #print(target.shape, new_im.shape)
    b_target = target>0
    target1 = target[b_target]
    new_im = new_im[b_target]
    abs_err = torch.abs(target1 - new_im)
    thresh_c = 3.
    
    thresh_p = target * .05
    thresh_p = thresh_p[b_target]
    
    sum1=0
    for i in range(len(abs_err)):
        if((abs_err[i] < thresh_c).float()):
            sum1=sum1+1
        elif((abs_err[i] < thresh_p[i]).float()):
            sum1=sum1+1
    sum1 = sum1/(len(target))
    res = 1-sum1
    print(res)
    sum1 = torch.tensor(sum1)
    sum1.requires_grad = True
    return 100 * (1-sum1)