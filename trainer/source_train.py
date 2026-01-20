

import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.RandMix import RandMix


def source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

    best_acc = 0
    # Setting up the CUDA device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    for epoch in range(args.base_epochs):
        # Set the model to the training mode
        backbone.train()
        classifier.train()

        # Set all the losses to zeros
        train_loss = 0


        # Set the counters to zeros
        correct = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')


        for batch_idx, instance in enumerate(train_loader):
            # Get a batch of training samples, transfer them to the device
            inputs, labels = instance[0].to(device), instance[1].to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # RandMix augmentation
            if args.RandMix:
                ratio = epoch / args.base_epochs
                RandMix_Aug = RandMix(1).cuda()
                data_aug = RandMix_Aug(inputs, ratio=ratio)
                inputs = torch.cat([inputs, data_aug])
                labels = torch.cat([labels, labels])

            # Forward the samples in the deep networks
            features = backbone(inputs)
            outputs = classifier(features)
            if args.LabelSmooth:
                task_criterion = CrossEntropyLabelSmooth(args.nb_cl).cuda()
                loss = task_criterion(outputs, labels) 
            else:
                loss = nn.CrossEntropyLoss()(outputs, labels)
    
            if args.contrastive_loss:
                loss += Supervised_InfoNCE_loss(features,labels)*0.1
            elif args.PCL:
                loss += PCLoss(args.nb_cl, 0.07)(features, labels, classifier.classifier.weight)

                
            # Backward and update the parameters
            loss.backward()
            backbone_optimizer.step()
            classifier_optimizer.step()

            # proto = model.fc.weight.data
            # proto_loss = contrastive_loss(proto,torch.tensor([0,1,2,3,4]).to(device))
            # print(proto_loss)

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f}'.format(len(train_loader), train_loss/(batch_idx+1),  100.*correct/total))


        # Running the test for this epoch
        backbone.eval()
        classifier.eval()
        test_loss = 0
        correct = 0
        total = 0
        # tem_testloader = testloader
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
