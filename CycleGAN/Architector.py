import torch
from torch import nn
from Network_componnet import ResidualBlock, ConractingBlock, ExpandingBlock, FeatureMapBlock

class Generator(nn.Module):
    '''
    Generator Class:
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channel, output_channel, hidden_channel=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channel, hidden_channel)
        self.contract1 = ConractingBlock(input_channel)
        self.contract2 = ConractingBlock(hidden_channel * 2)
        self.res0 = ResidualBlock(input_channel * 4)
        self.res1 = ResidualBlock(input_channel * 4)
        self.res2 = ResidualBlock(input_channel * 4)
        self.res3 = ResidualBlock(input_channel * 4)
        self.res4 = ResidualBlock(input_channel * 4)
        self.res5 = ResidualBlock(input_channel * 4)
        self.res6 = ResidualBlock(input_channel * 4)
        self.res7 = ResidualBlock(input_channel * 4)
        self.res8 = ResidualBlock(input_channel * 4)
        self.expand2 = ExpandingBlock(hidden_channel * 4)
        self.expand3 = ExpandingBlock(hidden_channel * 2)
        self.downfeature = FeatureMapBlock(hidden_channel, output_channel)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)

class Discriminator(nn.Module):
    def __init__(self, input_channel, hidden_channel=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channel, hidden_channel)
        self.contract1 = ConractingBlock(hidden_channel, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ConractingBlock(hidden_channel * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ConractingBlock(hidden_channel * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channel * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn