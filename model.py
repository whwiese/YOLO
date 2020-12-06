import torch
import torch.nn as nn
"""
FORMAT: 
(K,C,S,P) - ConvNet with Kernel Size=K, Output Filters=C, Stride=S, Padding=P
"M" - 2x2 Max Pooling Layer with Stride = 2
[(K,C,S,P),(K,C,S,P),N] - Tuples signify CovNets with same format as above,
N signifies number of times to repeat sequence of conv layers
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

#A CNN-->Batch Norm--> Leaky ReLU block that will make up most of the DarkNet structure
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, padding=0):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size,  bias=False,
                stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.LeakyReLU(x)
        return x

class YOLOv1(nn.Module):
    def __init__(
            self, num_classes, in_channels=3, grid_size=7, num_boxes=2):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_darknet(self.architecture)
        self.fcs = self.create_fcs(num_classes, grid_size, num_boxes)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        return x

    def create_darknet(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layer in architecture:
            if type(layer) == tuple:
                layers += [CNNBlock(in_channels,layer[1],layer[0],layer[2],layer[3])]
                in_channels = layer[1]
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            elif type(layer) == list:
                sublayers = len(layer)-1
                N = layer[-1]
                for _ in range(N):
                    for j,conv in enumerate(layer[:-1]):
                        layers += [CNNBlock(in_channels,conv[1],conv[0],conv[2],conv[3])]
                        in_channels = conv[1]

        return nn.Sequential(*layers)

    def create_fcs(self, num_classes, grid_size, num_boxes):
        S,B,C = grid_size, num_boxes, num_classes
        return  nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024*7*7, 1024), #Original paper uses output size of 4096
                nn.Dropout(0.0),
                nn.LeakyReLU(0.1),
                nn.Linear(1024,S*S*(C+B*5)), #rehspe later for SxSx(C+B*5) output
                )

#def test_model(grid_size = 6, num_boxes=2, num_classes=20):
#    model = YOLOv1(num_classes,3,grid_size,num_boxes)
#    x = torch.zeros((2,3,448,448))
#    print(model(x).shape)

