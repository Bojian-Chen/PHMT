import torch
import torch.nn as nn
from torch.nn import functional as F

def gsfda_build_banks(args, train_loader, backbone, classifier):
    feature_shape = {'cnn': 640, 'resnet14': 64, 'resnet32': 64, 'resnet18_1D': 256}
    feature_shape = feature_shape[args.backbone_name] 
    backbone.eval()
    classifier.eval()
    num_samples = len(train_loader.dataset)
    feature_bank = torch.zeros(num_samples, feature_shape, dtype=torch.float32)
    score_bank = torch.zeros(num_samples, args.nb_cl, dtype=torch.float32).cuda()
    with torch.no_grad():
        for batch_id, instance in enumerate(train_loader):
            inputs = instance[0].cuda()
            idx = instance[2]
            features, masks = backbone(inputs, t=1, s=100, all_mask=False)
            feature_norm = F.normalize(features)
            feature_bank[idx] = feature_norm.detach().clone().cpu()
            # feature_bank[idx] = features.detach().clone().cpu()

            outputs = classifier(features)
            outputs_softmax = nn.Softmax(dim=1)(outputs)
            score_bank[idx] = outputs_softmax.detach().clone()
    return feature_bank, score_bank

