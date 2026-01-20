import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F


def Supervised_InfoNCE_loss(features, labels, temperature=0.07):
    '''
    The InfoNCE loss for supervised learning
    features: (batch_size, feature_dim)
    labels: (batch_size)
    temperature: default 0.07
    '''
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.t())
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

    pos_pair_sim = similarity_matrix * labels_matrix.float()
    pos_pair_sim = torch.exp(pos_pair_sim / temperature)

    neg_pair_sim = similarity_matrix * (~labels_matrix).float()
    neg_pair_sim = torch.exp(neg_pair_sim / temperature)

    contrastive_loss = -torch.log(pos_pair_sim / (pos_pair_sim + torch.sum(neg_pair_sim, dim=1)))
    contrastive_loss = torch.mean(contrastive_loss)

    return contrastive_loss


def P_InfoNCE_loss(prototypes, features, labels, temperature=0.07):
    '''
    The Prototypes-based InfoNCE Loss
    prototypes: (num_prototypes, feature_dim)
    features: (batch_size, feature_dim)
    labels: (batch_size)
    temperature: default 0.07
    Equation: L = -log(exp(similarity(pos_pair) / temperature) / (exp(similarity(pos_pair) / temperature) + sum(exp(similarity(neg_pair) / temperature)))
    '''
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.cdist(features, prototypes, p=2)
    labels_matrix = labels.unsqueeze(1) == torch.arange(prototypes.size(0)).unsqueeze(0)

    pos_pair_sim = similarity_matrix * labels_matrix.float()
    pos_pair_sim = torch.exp(pos_pair_sim / temperature)

    neg_pair_sim = similarity_matrix * (~labels_matrix).float()
    neg_pair_sim = torch.exp(neg_pair_sim / temperature)
    
    contrastive_loss = -torch.log(pos_pair_sim / (pos_pair_sim + torch.sum(neg_pair_sim, dim=1)))
    contrastive_loss = torch.mean(contrastive_loss)

    return contrastive_loss


def Unsupervised_Prototype_InfoNCE_loss(prototypes, features, temperature=0.07):
    """
    Unsupervised Prototype-based InfoNCE Loss.
    Treats the most similar prototype as the positive sample for each feature.

    Args:
        prototypes (torch.Tensor): Prototypes (num_prototypes, feature_dim).
                                   Assumed to be normalized.
        features (torch.Tensor): Input features (batch_size, feature_dim).
        temperature (float): Temperature scaling factor. Default is 0.07.

    Returns:
        torch.Tensor: The calculated loss value.
    """
    features = F.normalize(features, dim=1)
    # prototypes = F.normalize(prototypes, dim=1) # Ensure prototypes are normalized

    # Calculate cosine similarity: (batch_size, num_prototypes)
    similarity_matrix = torch.matmul(features, prototypes.t())

    # Find the index of the most similar prototype for each feature (pseudo-label)
    # This prototype acts as the positive sample.
    # pseudo_labels shape: (batch_size)
    pseudo_labels = torch.argmax(similarity_matrix, dim=1)

    # Calculate logits
    # logits shape: (batch_size, num_prototypes)
    logits = similarity_matrix / temperature

    # Use cross_entropy loss with the pseudo-labels as targets
    # This encourages each feature to be closer to its nearest prototype
    # than to other prototypes.
    contrastive_loss = F.cross_entropy(logits, pseudo_labels)

    return contrastive_loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    entropy = torch.mean(entropy)
    return entropy 

def DivEntropy(input_):
    epsilon = 1e-5
    msoftmax = input_.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + epsilon))
    return gentropy_loss



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss



class PCLoss(nn.Module):
    '''
        The Proxy-Contrastive Loss
        feature: (N, dim)
        proxy: (C, dim)
    '''
    def __init__(self, num_classes, scale):
        super(PCLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)]).cuda()
        self.scale = 1 / scale

    def forward(self, feature, target, proxy):
        '''
        feature: (N, dim)
        proxy: (C, dim)
        '''
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  

        label = (self.label.unsqueeze(1) == target.unsqueeze(0))   
        pred_p = torch.masked_select(pred, label.transpose(1, 0))    # (N)   positive pair
        pred_p = pred_p.unsqueeze(1)
        pred_n = torch.masked_select(pred, ~label.transpose(1, 0)).view(feature.size(0), -1)  # (N, C-1) negative pair of anchor and proxy

        feature = torch.matmul(feature, feature.transpose(1, 0))  
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  

        feature = feature * ~label_matrix  
        feature = feature.masked_fill(feature < 1e-6, -np.inf)

        logits = torch.cat([pred_p, pred_n, feature], dim=1)  
        label = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
        return loss

class PCALoss(nn.Module):
    '''
        The Proxy-Contrastive Loss with Memory Bank
        feature: (N, dim)
        proxy: (C, dim)
        Mproxy: (C, dim) 
    '''
    def __init__(self, num_classes, scale):
        super(PCALoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)]).cuda()
        self.scale = 1 / scale

    def forward(self, feature, target, proxy, Mproxy, mweight=1):
        '''
        feature: (N, dim)
        proxy: (C, dim)
        Mproxy: (C, dim) 
        '''
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)  similarity between sample and proxy
        Mpred = F.linear(feature, F.normalize(Mproxy, p=2, dim=1))  # (N, C)  similarity between sample and old proxy
        
        label = (self.label.unsqueeze(1) == target.unsqueeze(0))   
        pred_p = torch.masked_select(pred, label.transpose(1, 0))    # (N)   positive pair
        pred_p = pred_p.unsqueeze(1)
        pred_n = torch.masked_select(pred, ~label.transpose(1, 0)).view(feature.size(0), -1)  # (N, C-1) negative pair of anchor and proxy
        Mpred_p = torch.masked_select(Mpred, label.transpose(1, 0))    
        Mpred_p = Mpred_p.unsqueeze(1)
        Mpred_n = torch.masked_select(Mpred, ~label.transpose(1, 0)).view(feature.size(0), -1)  
        
        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)  sample wise similarity
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  
        
        feature = feature * ~label_matrix  
        feature = feature.masked_fill(feature < 1e-6, -np.inf)

        loss = -torch.log(  ( torch.exp(self.scale*pred_p.squeeze()) + mweight*torch.exp(self.scale*Mpred_p.squeeze()) ) / 
                            ( torch.exp(self.scale*pred_p.squeeze()) + mweight*torch.exp(self.scale*Mpred_p.squeeze()) + 
                            torch.exp(self.scale*pred_n).sum(dim=1) + mweight*torch.exp(self.scale*Mpred_n).sum(dim=1) +  torch.exp(self.scale*feature).sum(dim=1))  ).mean()
        return loss

