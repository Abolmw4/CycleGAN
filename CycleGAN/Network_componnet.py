from torch import nn


class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class:
    Consists of two convulsions and an instance normalization, adds input to output which becomes residual block output
    Values:
        input_channel: the number of channels to expect from a given input
    '''
    def __init__(self, input_channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channel)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        This is a function of the residualblock architecture
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ConractingBlock(nn.Module):
    '''ConractingBlock Class:
    Consists of a convolution with Maxpooling, and optional instance norm
    Values:
        input_channel: the number of channels to expect from a given input
    '''
    def __init__(self, input_channel, kernel_size=3, activation='relu', use_bn=True):
        super(ConractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instance = nn.InstanceNorm2d(input_channel * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        This function takes an image tensor and gives it to contracting bock, and returns a transformed tensor
        Parameters:
            x: image tensor for shape(batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instance(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample,
    with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channel, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channel, input_channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channel // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        this function Given an image tensor, completes an expanding block and returns the transformed tensor.
        parameter:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator -
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channel, output_channel):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x