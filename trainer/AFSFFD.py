"""
Anti-forgetting source-free domain adaptation method for machine fault diagnosis
"""

import torch
import torch.nn as nn
import torch.jit
import math
import torch.nn.functional as F
from loss_function import *
import copy
import torch.optim as optim
from utils.Fisher import Fisher


def AFSFFD(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler, fishers):

    fisher_alpha=8500

    best_acc = 0

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    for epoch in range(args.epochs):
        # Set the model to the training mode
        backbone.train()
        classifier.eval()

        train_loss = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx, instance in enumerate(train_loader):
            # Get a batch of training samples, transfer them to the device
            inputs, labels = instance[0].to(device), instance[1].to(device)
       
            # train model_new
            backbone_optimizer.zero_grad()
            
            # Forward the samples in the deep networks
            features = backbone(inputs)
            outputs = classifier(features)

            outputs_softmax = nn.Softmax(dim=1)(outputs)
            
            # Entropy 
            # Entropy Loss
            loss1 = Entropy(outputs_softmax)
           
            # DivEntropy Loss
            loss2 = DivEntropy(outputs_softmax)

            loss = (loss1 - loss2) * 0.1


            # FBNM Loss
            list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(outputs_softmax,2),dim=0)), descending=True)
            transfer_loss = - torch.mean(list_svd[:min(outputs_softmax.shape[0],outputs_softmax.shape[1])])
            loss += transfer_loss



            ewc_loss = 0
            for name, param in backbone.named_parameters():
                if name in fishers:
                    ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()

            loss +=  ewc_loss

           
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
      

            # BP
            loss.backward()
            train_loss += loss.item()
 
            backbone_optimizer.step()


   
        # Learning rate decay
        backbone_scheduler.step()


        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f} '.format(
            len(train_loader), train_loss/(batch_idx+1),  100.*correct/total))

        # Running the test for this epoch
        backbone.eval()
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

    # fishers = Fisher(best_backbone, best_classifier, train_loader)

    return best_backbone, best_classifier, fishers
