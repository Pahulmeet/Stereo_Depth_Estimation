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
    def __init__(self,B,C,H,W,newmodel):
        super(baseline_model,self).__init__()
        
        self.B=B
        self.C=C
        self.H=H
        self.W=W
        self.device = "cuda"
      
        
        
        self.max_disp = 192
        self.cnn_Shared = newmodel
        
        self.cnn_3dims = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
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
        score_volume = self.cnn_3dims(cost_vol)
        
        m = nn.Upsample(scale_factor=4, mode='trilinear')
        score_volume = m(score_volume)
        
        y = score_volume.flatten(start_dim=1, end_dim=2)
        
        prob=F.softmax(y,1)
        #https://github.com/jzhangbs/DSM/blob/master/model.py
        prob = disparityregression(self.max_disp)(prob)
        return prob
def create_mod():
    model = models.resnet18(pretrained=True)  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    #model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    newmodel = torch.nn.Sequential(*(list(model.children())[0:8]))
    newmodel[5][0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    newmodel[5][0].downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    newmodel[6][0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    newmodel[6][0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    newmodel = newmodel[:7]
    newmodel.newconv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    newmodel.newbn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #newmodel.ndownsample1 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    newmodel.newconv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    newmodel.newbn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    newmodel.newconv3 = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #print(newmodel)
    #newmodel.to(device)
    return newmodel