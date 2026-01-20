import numpy as np
from models.cnn import cnn
from models.embedding import backbone_embedding
from models.resnet_1d import resnet18
from models.resnet_2d import resnet14, resnet32
from models.classifier import FCLinear, ProjectionHead, CosineLinear, EuclideanLinear, FCWNLinear
from dataloader_domain import dataloader as dataloader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import random

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


feature_shape = {'cnn': 640, 'resnet14': 64, 'resnet32': 64, 'resnet18_1D': 256}
def set_model(args):
    if args.backbone_name == 'cnn':
        backbone = cnn()
    elif args.backbone_name == 'resnet14':
        backbone = resnet14()
    elif args.backbone_name == 'resnet32':
        backbone = resnet32()
    elif args.backbone_name == 'resnet18_1D':
        backbone = resnet18()

    feature_shape_ = feature_shape[args.backbone_name]

    if args.incremental_mode == 'gsfda' or args.incremental_mode == 'UCSN':
        backbone = backbone_embedding(args, backbone)
        # feature_shape_ = feature_shape[args.backbone_name] 

    if args.classifer == 'fc':
        classifier = FCLinear(feature_shape_, args.nb_cl)
    elif args.classifer == 'cos':
        classifier = CosineLinear(feature_shape_, args.nb_cl)
    elif args.classifer == 'eu':
        classifier = EuclideanLinear(feature_shape_, args.nb_cl)
    elif args.classifer == 'fcwn':
        classifier = FCWNLinear(feature_shape_, args.nb_cl)
    
    return backbone, classifier


def set_dataset(args):
    trainloader_list = []
    testloader_list = []
    for session in range(args.nb_session):
        Dataset = dataloader(args,session)
        trainloader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0,pin_memory=True)
        trainloader_list.append(trainloader)
        testloader_list.append(testloader)
    return trainloader_list, testloader_list

        # lr_strat = [int(args.base_epochs*0.5), int(args.base_epochs*0.75)]
        # tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=0.1)
    
def set_optimizer(args, backbone, classifier, session):
    if session ==0:
        
        backbone_optimizer = optim.SGD(backbone.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        classifier_optimizer = optim.SGD(classifier.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        # lr_strat = [int(args.base_epochs*0.5), int(args.base_epochs*0.75)]
        # backbone_scheduler = MultiStepLR(backbone_optimizer, milestones=lr_strat, gamma=0.1)
        # classifier_scheduler = MultiStepLR(classifier_optimizer, milestones=lr_strat, gamma=0.1)
        backbone_scheduler =  CosineAnnealingLR(backbone_optimizer, args.base_epochs, eta_min = args.eta_min)
        classifier_scheduler =  CosineAnnealingLR(classifier_optimizer, args.base_epochs, eta_min = args.eta_min)

    else:
        if args.train_parames == 'BN':
            for m in backbone.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                else:
                    m.requires_grad_(False)
            BN_params = filter(lambda p: p.requires_grad, backbone.parameters())
            backbone_optimizer = optim.SGD(BN_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

        elif args.train_parames == 'woBN':
            for m in backbone.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(False)
                else:
                    m.requires_grad_(True)
            woBN_params = filter(lambda p: p.requires_grad, backbone.parameters())
            backbone_optimizer = optim.SGD(woBN_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        elif args.train_parames == 'default':
            backbone_optimizer = optim.SGD(backbone.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        classifier_optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # lr_strat = [int(args.base_epochs*0.5), int(args.base_epochs*0.75)]
    # backbone_scheduler = MultiStepLR(backbone_optimizer, milestones=lr_strat, gamma=0.1)
    # classifier_scheduler = MultiStepLR(classifier_optimizer, milestones=lr_strat, gamma=0.1)
    backbone_scheduler =  CosineAnnealingLR(backbone_optimizer, args.epochs, eta_min = args.eta_min)
    classifier_scheduler =  CosineAnnealingLR(classifier_optimizer, args.epochs, eta_min = args.eta_min)
    return backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler


