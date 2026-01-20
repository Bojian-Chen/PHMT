
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weightNorm

class FCLinear(nn.Module):
    def __init__(self, feature_shape, num_class):
        super(FCLinear, self).__init__()
        self.classifier = nn.Linear(feature_shape, num_class)

    def forward(self, x):
        x = self.classifier(x)
        return x
    

    
class FCWNLinear(nn.Module):
    def __init__(self, feature_shape, num_class):
        super(FCWNLinear, self).__init__()
        linear = nn.Sequential()
        fc = weightNorm(nn.Linear(feature_shape, num_class), name="weight")
        linear.add_module("fc", fc)
        self.linear = linear

    def forward(self, x):
        x = self.linear(x)
        return x


class cosinelinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(cosinelinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
 
    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out
    
class CosineLinear(nn.Module):
    def __init__(self, feature_shape, num_class):
        super(CosineLinear, self).__init__()
        self.classifier = cosinelinear(feature_shape, num_class)#

    def forward(self, x):
        x = self.classifier(x)
        return x
    
class euclideanlinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(euclideanlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        
        out = torch.cdist(F.normalize(input, p=2,dim=1).unsqueeze(1), \
                         F.normalize(self.weight, p=2, dim=1).unsqueeze(0)).squeeze()
 
        if self.sigma is not None:
            out = self.sigma * out * -1
        return out
    
class EuclideanLinear(nn.Module):
    def __init__(self, feature_shape, num_class):
        super(EuclideanLinear, self).__init__()
        self.classifier = euclideanlinear(feature_shape, num_class)

    def forward(self, x):
        x = self.classifier(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)