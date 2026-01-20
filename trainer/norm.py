"""
Improving robustness against common corruptions by covariate shift adaptation
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy

def configure_model(model, no_stats=False):
    """Configure model for adaptation by test-time normalization."""
    model.train()

    for param in model.parameters():  # initially turn off requires_grad for all
        param.requires_grad = False

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.requires_grad_(True)
 
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    return model

def norm(args, backbone, classifier, train_loader, test_loader):

    best_acc = 0
    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    # Set the model to the training mode
    # BN is updating
    backbone= configure_model(backbone)
    classifier= configure_model(classifier)

    test_loss = 0
    correct = 0
    total = 0


    for batch_idx, instance in enumerate(train_loader):
        inputs, labels = instance[0].to(device), instance[1].to(device)
        features = backbone(inputs)
        outputs = classifier(features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print('Train set: {} Train loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), test_loss/(batch_idx+1), 100.*correct/total))
        


    # Running the test for this epoch
    # BN is fixed
    backbone.eval()
    classifier.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, instance in enumerate(test_loader):
            inputs, labels = instance[0].to(device), instance[1].to(device)
            features = backbone(inputs)
            outputs = classifier(features)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), test_loss/(batch_idx+1), 100.*correct/total))

    if 100.*correct/total >= best_acc:
        best_acc = 100.*correct/total
        best_backbone = copy.deepcopy(backbone)
        best_classifier = copy.deepcopy(classifier)
            
    return best_backbone, best_classifier
