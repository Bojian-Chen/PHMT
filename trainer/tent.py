
"""
TENT: FULLY TEST-TIME ADAPTATION BY ENTROPY MINIMIZATION
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy


def tent(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler):

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
            
            # Entropy Loss
            loss = Entropy(outputs_softmax)
           

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

    return best_backbone, best_classifier
