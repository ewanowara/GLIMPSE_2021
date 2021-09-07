import torch 
import torch.nn as nn

'''
Difference from original U-Net is that a down or up after a single conv layer, in U-Net several conv layers and then up or down
Less params than oirginal U-Net if less layers? 
why bias=False when BN?

'''

class Block(nn.Module): # similar to CNN block
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False): # down = True - encoder, downward part, down = False - decoder, upward part, one uses ReLU, the other LeakyReLU, only some layers used DropOut
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") # in, out, kernel=4, stride=2, padding=1, bias=False bc BN 
            if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False), # upsample instead of downsample
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2), # some layers use ReLU, some use LeakyReLU
        )

        self.use_dropout = use_dropout # only some layers use Dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2), # no BN in initial layer
        ) # 128
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False) # 64
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False # 32
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False # 16
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False # 8
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False # 4
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False # 2
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), padding_mode="reflect", nn.ReLU() # 1 x 1 at the bottom, why?
        ) # not a block, just conv layer with ReLU

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        ) # 8*2 because concatenating from down part
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # want each pixel value between -1 and 1, why not 0 to 1?
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1)) # concatenate corresponding encoder and decoder layers (U-NET skip connections), same dimensions 
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()