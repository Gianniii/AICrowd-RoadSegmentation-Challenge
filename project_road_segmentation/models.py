# Definition of neural networks
#Our U-Net is heavily inspired by the above, source
#https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from numpy import moveaxis
import torchvision.transforms.functional as TF




# Basic Neural Network, 3 fully-connected layers with batch normalization
class BasicNNRoadSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(
            self.bn1(self.fc1(x.view(-1, 768)))
        )  # each batch has batch_size*768 features (16x16x3)
        x = F.relu(self.bn2(self.fc2(x.view(-1, 100))))
        x = self.fc3(x.view(-1, 100))
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(p=0.2),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# Get desired activation function
def getActivation(act):
    if act == "relu":
        activation = F.relu
    if act == "leaky":
        activation = F.leaky_relu
    if act == "sigmoid":
        activation = torch.sigmoid
    return activation
