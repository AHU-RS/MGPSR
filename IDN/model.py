import torch.nn as nn
import torch.nn.functional as F
from common.blocks import DoubleConv, SA, PA, Down, Up, ResnetBlock, OutConv

class MGPSR_IDN(nn.Module): 
    def __init__(self, n_channels=4, n_classes=1, res_down=True, n_resblocks=1, bilinear=False):
        super(MGPSR_IDN, self).__init__()
        
        self.inc = DoubleConv(n_channels, 64)
        self.sa1 = SA()

        self.down1 = Down(64, 128, res_down=res_down)
        self.sa2 = SA()

        self.down2 = Down(128, 256, res_down=res_down)
        self.sa3 = SA()

        self.down3 = Down(256, 512, res_down=res_down)
        
        self.pa1 = PA(512)
        
        self.sa4 = SA() 

        resblocks = []
        for i in range(n_resblocks):
            resblocks += [ResnetBlock(512, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True)]
        self.resblocks = nn.Sequential(*resblocks)

       # Up(in_channels=Input from the bottom, skip_channels=Input from the left that is skipped, out_channels=Output you expect...)
        self.up1 = Up(in_channels=512, skip_channels=256, out_channels=256, bilinear=bilinear)
        self.up2 = Up(in_channels=256,  skip_channels=128, out_channels=128, bilinear=bilinear)
        self.up3 = Up(in_channels=128,  skip_channels=64,  out_channels=64,  bilinear=bilinear)
        
        self.pa2 = PA(64)
        self.outc = OutConv(64, n_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        lr_only = x[:, 0:1, :, :]
        
        x1 = self.inc(x)        
        sa1_out = self.sa1(x1)

        x2 = self.down1(x1)      
        sa2_out = self.sa2(x2)

        x3 = self.down2(x2)      
        sa3_out = self.sa3(x3)

        x4 = self.down3(x3)     

      
        x4 = self.pa1(x4) + x4
        x4 = self.resblocks(x4)

        # Prepare for Skip Connection 
        u4 = x4
        u3 = sa3_out + x3
        u2 = sa2_out + x2
        u1 = sa1_out + x1

        xp1 = self.up1(u4, u3)   # 512 concatenation 256    ->  Output 256
        xp2 = self.up2(xp1, u2)  # 256 concatenation 128    ->  Output 128
        xp3 = self.up3(xp2, u1)  # 128 concatenation 64     ->  Output 64
        
        xp4 = self.pa2(xp3) + xp3
        
        logits = self.outc(xp4) + lr_only
        return logits