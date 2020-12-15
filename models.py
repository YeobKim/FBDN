import torch
import torch.nn as nn

class FeaturesBlockDualNet(nn.Module):
    def __init__(self, channels):
        super(FeaturesBlockDualNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        convlayer = []
        convlayer.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        convlayer.append(nn.ReLU(inplace=True))
        convlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        convlayer.append(nn.ReLU(inplace=True))
        convlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        convlayer.append(nn.ReLU(inplace=True))
        convlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        convlayer.append(nn.ReLU(inplace=True))
        convlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        convlayer.append(nn.ReLU(inplace=True))
        convlayer.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.convlayer = nn.Sequential(*convlayer)

        layers1 = []
        layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers1.append(nn.ReLU(inplace=True))
        for _ in range(15):
            layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers1.append(nn.BatchNorm2d(features))
            layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))

        self.one = nn.Sequential(*layers1)

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        layers2.append(nn.ReLU(inplace=True))

        for _ in range(6):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, dilation=2, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(nn.ReLU(inplace=True))

        for _ in range(2):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(nn.ReLU(inplace=True))

        for _ in range(6):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, dilation=2, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(nn.ReLU(inplace=True))

        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers2.append(nn.ReLU(inplace=True))

        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        self.two = nn.Sequential(*layers2)

        normalblock = []
        normalblock.append(nn.Conv2d(in_channels=features * 3, out_channels=features * 3, kernel_size=kernel_size, padding=padding, bias=False))
        normalblock.append(nn.BatchNorm2d(features * 3))
        normalblock.append(nn.ReLU(inplace=True))
        self.normalblock = nn.Sequential(*normalblock)

        lastblock = []
        lastblock.append(nn.Conv2d(in_channels=features * 3, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        # lastblock.append(nn.ReLU(inplace=True))

        self.lastblock = nn.Sequential(*lastblock)

        self.conv = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels*2, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        residual = x
        featuredata = self.convlayer(x)
        inputdata = self.conv2(torch.cat((x, featuredata), 1))

        out1 = self.one(inputdata)
        out2 = self.two(inputdata)

        cat_data = torch.cat((out1 + residual, out2 + residual, inputdata + residual), 1)
        head = self.normalblock(cat_data)
        body = self.normalblock(head)
        tail = self.lastblock(body)

        out = tail + residual

        return out