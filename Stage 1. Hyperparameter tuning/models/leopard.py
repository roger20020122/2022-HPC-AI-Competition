from __future__ import print_function
from math import ceil

import torch
from torch import nn
ss=10

def pcc(num_filter_in, num_filter, size_kernel,dropout=0):
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(num_filter_in, num_filter, size_kernel, padding=size_kernel//2),
        nn.ReLU(),
        nn.BatchNorm1d(num_filter),
        nn.Dropout(dropout),
        nn.Conv1d(num_filter, num_filter, size_kernel, padding=size_kernel//2),
        nn.ReLU(),
        nn.BatchNorm1d(num_filter),
        nn.Dropout(dropout),
    )

class ucc(torch.nn.Module):
    def __init__(self, num_filter_in, num_filter, size_kernel,dropout=0) -> None:
        super().__init__()
        self.upscaler = nn.ConvTranspose1d(num_filter_in, num_filter, 2, padding=0, stride=2)
        self.net = nn.Sequential(
            nn.Conv1d(num_filter*2, num_filter, size_kernel, padding=size_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filter),
            nn.Dropout(dropout),
            nn.Conv1d(num_filter, num_filter, size_kernel, padding=size_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filter),
            nn.Dropout(dropout),
        )

    def forward(self, x, skip):
        x = self.upscaler(x)
        x = torch.cat([x, skip], dim=1)
        x = self.net(x)
        return x
class LeopardUnet(nn.Module):
    def __init__(self, num_class=1, num_channel=5,dropout:float=0,num_blocks=-5,initial_filter=15,size_kernel=7,scale_filter=1.5):
        super().__init__()
        self.layer_down=nn.ModuleList()
        self.layer_up=nn.ModuleList()

        conv0 = nn.Sequential(
            nn.Conv1d(num_channel, initial_filter, size_kernel, padding=size_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(initial_filter),
            nn.Dropout(dropout),
            nn.Conv1d(initial_filter, initial_filter, size_kernel, padding=size_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(initial_filter),
            nn.Dropout(dropout),
        )

        self.layer_down.append(conv0)
        num=initial_filter

        for i in range(num_blocks):
            pnum,num=num,int(num * scale_filter)
            the_layer=pcc(pnum, num, size_kernel,dropout)
            self.layer_down.append(the_layer)

        for i in range(num_blocks):
            pnum,num=num,ceil(num / scale_filter)
            the_layer=ucc(pnum, num, size_kernel,dropout)
            self.layer_up.append(the_layer)

        self.convn = nn.Conv1d(num,num_class,1)

        return 

    def forward(self,x):
        x = x.transpose(1,2)

        num_blocks=len(self.layer_up)

        skips=[None]*(num_blocks+1)
        for i in range(num_blocks+1):
            x = skips[i] = self.layer_down[i](x)
        # last skip is not used

        for i in range(num_blocks):
            x = self.layer_up[i](x, skips[num_blocks-i-1])
        
        x = self.convn(x)
        x = x.transpose(1,2)
        return x

def test():
    model = LeopardUnet()
    x = torch.randn(1,10240,5)
    y = model(x)
    print(y.shape)
    # count parameters in million
    print(sum(p.numel() for p in model.parameters())/1e6)

if __name__ == '__main__':
    test()