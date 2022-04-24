import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

from torchvision import datasets, models, transforms

from Regressor_and_loss import disparityregression
# input = (B,3,256,512)
class baseline_model(nn.Module):
    def __init__(self,B,C,H,W,newmodel,model3d):
        super(baseline_model,self).__init__()
        
        self.B=B
        self.C=C
        self.H=H
        self.W=W
        self.device = "cuda"
      
        
        
        self.max_disp = 192
        self.cnn_Shared = newmodel
        self.cnn_3dims1 = model3d
        self.cnn_3dims2 = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )
        
        #y = x.flatten(start_dim=1, end_dim=2) #https://stackoverflow.com/questions/65465646/convert-5d-tensor-to-4d-tensor-in-pytorch"""
        # size = (B, 192, 256 , 512)
    def concat_for_3D(self, left_feats, right_feats):
        cost = torch.Tensor(self.B,self.C*2, self.max_disp//4, self.H//4, self.W//4).to(self.device)
        for i in range(self.max_disp // 4):
            if(i==0):
                cost[:, :self.C, i, :, :] = left_feats
                cost[:, self.C:, i, :, :] = right_feats
            else:
                cost[:, :self.C, i, :, i:] = left_feats[:,:,:,i:]
                cost[:, self.C:, i, :, i:] = right_feats[:,:,:,:-i]

        return cost
        
    def forward(self,x_left,x_right):
        im_left = self.cnn_Shared(x_left)
        im_right = self.cnn_Shared(x_right)
        cost_vol = self.concat_for_3D(im_left,im_right)
        score_volume = self.cnn_3dims1(cost_vol)
        score_volume = self.cnn_3dims2(score_volume)
        
        m = nn.Upsample(scale_factor=4, mode='trilinear')
        score_volume = m(score_volume)
        
        y = score_volume.flatten(start_dim=1, end_dim=2)
        
        prob=F.softmax(y,1)
        #https://github.com/jzhangbs/DSM/blob/master/model.py
        prob = disparityregression(self.max_disp)(prob)
        return prob
def create_mod():
    model3d = models.video.r3d_18(pretrained=True, progress=False)
    num_features = model3d.fc.in_features

    model3d  = model3d.layer1

    # downloading pretrained model
    model = models.vgg16(pretrained=True)  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    device = "cuda"
    for param in model.features.parameters():
        param.require_grad = False
    num_features = model.features

    newmodel=num_features[:12]
    newmodel.extend = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    return newmodel, model3d