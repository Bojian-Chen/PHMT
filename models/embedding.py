import torch
from torch import nn

feature_shape = {'cnn': 640, 'resnet14': 64, 'resnet32': 64, 'resnet18_1D': 256}
class backbone_embedding(nn.Module):
    def __init__(self, args, backbone):
        super(backbone_embedding, self).__init__()
        self.encoder = backbone
        self.bottleneck = nn.Sequential()
        fc = nn.Linear(feature_shape[args.backbone_name], feature_shape[args.backbone_name])
        self.bottleneck.add_module("fc", fc)
        self.bottleneck.add_module("bn", nn.BatchNorm1d(feature_shape[args.backbone_name], affine=True))
        self.mask_embedding = nn.Embedding(2, feature_shape[args.backbone_name])

        # self.bottle = nn.Linear(feature_shape[args.backbone_name], feature_shape[args.backbone_name])
        # self.bn = nn.BatchNorm1d(feature_shape[args.backbone_name])
        # self.em = nn.Embedding(2, feature_shape[args.backbone_name])
  
    def forward(self, x, t, s, all_mask=False):
        feature = self.encoder(x)
        feature = torch.flatten(feature, 1)
        feature = self.bottleneck(feature)
        # feature = self.bottle(feature)
        # feature = self.bn(feature)
        t0 = torch.LongTensor([0]).cuda()
        mask_0 = torch.sigmoid(self.mask_embedding(t0) * s)
        if t == 0:
            feature = feature * mask_0
        elif t == 1:
            t1 = torch.LongTensor([1]).cuda()
            mask_1 = nn.Sigmoid()(self.mask_embedding(t1) * s)
            feature = feature * mask_1
        if all_mask:
            t1 = torch.LongTensor([1]).cuda()
            mask_1 = nn.Sigmoid()(self.mask_embedding(t1) * s)
            feature0 = feature * mask_0
            feature1 = feature * mask_1
            return (feature0, feature1), (mask_0, mask_1)
        else:
            return feature, mask_0