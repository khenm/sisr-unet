import torch.nn as nn
import torch


class FeatureExtract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtract, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               padding=padding, stride=stride,
                               kernel_size=kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.convblock = ConvBlock(
            in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.maxpooling(x)
        x = self.convblock(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip=False):
        super(Decoder, self).__init__()
        self.use_skip = use_skip
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, stride=1,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.convblock = ConvBlock(out_channels*2, out_channels)
        if use_skip:
            self.decoder[1] = nn.Conv2d(
                in_channels, out_channels, 1, 1, 0, bias=False)
            self.decoder[2] = nn.BatchNorm2d(out_channels)
            self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.decoder(x)
        if self.use_skip:
            x = torch.concat([x, skip], dim=1)
        x = self.convblock(x)
        return x


class FinalMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalMapping, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, features=[64, 128, 256, 512, 1024], use_skip=False):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.feature_ext = FeatureExtract(n_channels, features[0])
        self.convblock1 = ConvBlock(features[0], features[0])

        self.encoder1 = Encoder(features[0], features[1])
        self.encoder2 = Encoder(features[1], features[2])
        self.encoder3 = Encoder(features[2], features[3])
        self.encoder4 = Encoder(features[3], features[4])

        self.decoder1 = Decoder(features[4], features[3], use_skip)
        self.decoder2 = Decoder(features[3], features[2], use_skip)
        self.decoder3 = Decoder(features[2], features[1], use_skip)
        self.decoder4 = Decoder(features[1], features[0], use_skip)

        self.out_conv = FinalMapping(features[0], n_classes)

    def forward(self, x):
        x = self.feature_ext(x)
        x1 = self.convblock1(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)

        x = self.out_conv(x)
        return x
