import torch
import torch.nn as nn

'''
Does the discriminator only have these two CNN layers?
'''

# CNN block 
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        # conv layer block:
            # conv
            # batch norm
            # leaky relu
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ), # stride either 1 or 2, padding reflect to avoid artifacts
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # in channels = 64, send 64 to 128, 128 to 256, 256 to 512; 256 input - after conv layers get 26 x 26 output 
        super().__init__()
        self.initial = nn.Sequential( # CNN without BN, in_channels * 2 because it gets both the input and output image to tell if it is real or fake 
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # create layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]: # skip first feature, it will create as many layers as length of the input features vector 
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y): # get x image and y real or fake 
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 3, 256, 256)) # test on a random example
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()