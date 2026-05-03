import torch.nn as nn

class vgg_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vgg_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels)
        self.pool = nn.MaxPool2d(stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        return x
    


class My_VGG2(nn.Module):
    def __init__(self):
        super(My_VGG2, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)  