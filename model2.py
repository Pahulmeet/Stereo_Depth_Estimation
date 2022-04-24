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
    def __init__(self,B,C,H,W):
        super(baseline_model,self).__init__()
        
        self.B=B
        self.C=C
        self.H=H
        self.W=W
        self.device = "cuda"
      
        
        
        self.max_disp = 192
        self.cnn_Shared = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(4,4), stride=(1,1),padding=(1,1)),
            #nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 12, kernel_size=(4,4), stride=(1,1),padding=(1,1)),
            #nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(12),

            nn.Conv2d(12, 16, kernel_size=(2,2), stride=(2,2),padding=(1,1)),
            #nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 24, kernel_size=(5,5), stride=(1,1),padding=(1,1)),
            #nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(24),

            nn.Conv2d(24, 32, kernel_size=(2,2), stride=(2,2),padding=(1,1)),
            #nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32), # (B,32,64,128)
        )
        
        
        self.cnn_3dims = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            #nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            #nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            #nn.BatchNorm3d,
            nn.ReLU(),
            nn.Conv3d(8, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            #nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            #nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.Conv3d(2, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.upsample_now = nn.Sequential(
            nn.ConvTranspose3d(1, 1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(0, 0, 0)),
            #nn.BatchNorm3d(1),
            nn.ConvTranspose3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)),
            #nn.BatchNorm3d(1),
            nn.ConvTranspose3d(1, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(4, 4, 4)),
            #nn.BatchNorm3d(1),
        )
        
        self.one_2_three_channel = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(1,1), stride=(1,1),padding=(0,0)),
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
