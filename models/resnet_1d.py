import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x) 
        out = self.bn1(out) 
        out = self.relu(out) 

        out = self.conv2(out) 
        out = self.bn2(out) 

        if self.downsample is not None:
            identity = self.downsample(x)  

        out += identity  
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet1d, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1) #1*256

        return x
    

def resnet18(pretrained=False, **kwargs):
    n = 2
    model = ResNet1d(BasicBlock, [n, n, n, n], **kwargs)
    return model

# model = Resnet18()
# x = torch.rand((1,1,1024))
# model(x)
# model = resnet18()
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
#         m.requires_grad_(True)
#     else:
#         m.requires_grad_(False)
#         # print(m)

# for name, param in model.named_parameters():
#     # print(name)
#     # if 'bn' in name:
#     #     print('bn')
#     # else:
#     #     print(param.requires_grad)
#     if param.requires_grad == True:
#         print(name)

