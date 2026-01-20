"""
1. Generalized Source-free Domain Adaptation
2. Unsupervised Continual Source-Free Network for Fault Diagnosis of Machines Under Multiple Diagnostic Domains
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.build_bank import gsfda_build_banks




def UCSN_source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

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

            # Forward the samples in the deep networks
            features, masks = backbone(inputs, t=0, s=100, all_mask=True)

            output_s1 = classifier(features[0])
            output_s2 = classifier(features[1])
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            reg /= count
            # task_criterion = CrossEntropyLabelSmooth(args.nb_cl).cuda()
            task_criterion = nn.CrossEntropyLoss().cuda()
            loss = task_criterion(output_s1, labels) + task_criterion(output_s2, labels) + 0.75 * reg
            
            

            # Backward and update the parameters
            loss.backward()
            backbone_optimizer.step()
            classifier_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()

        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f} '.format(len(train_loader), train_loss/(batch_idx+1)))


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
                features, masks = backbone(inputs, t=0, s=100, all_mask=False)
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





def UCSN(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

    best_acc = 0

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    backbone = backbone.to(device)
    classifier = classifier.to(device) 
    feature_bank, score_bank = gsfda_build_banks(args, train_loader, backbone, classifier)

    for epoch in range(args.epochs):
        # Set the model to the training mode

   
        train_loss = 0

        # Set the counters to zeros
        correct = 0
        total = 0
        

        backbone.train()
        classifier.train()
        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx, instance in enumerate(train_loader):
            # Get a batch of training samples, transfer them to the device
            inputs, labels = instance[0].to(device), instance[1].to(device)
            idx = instance[2]
       
            # train model_new
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
     
            # Forward the samples in the deep networks
            features, masks = backbone(inputs, t=1, s=100, all_mask=False)
            outputs = classifier(features)
            outputs_softmax = nn.Softmax(dim=1)(outputs)
            outputs_re = outputs_softmax.unsqueeze(1)

            with torch.no_grad():
                output_f_norm = F.normalize(features)
                feature_bank[idx].fill_(-0.1)  # do not use the current mini-batch in fea_bank
                output_f_ = output_f_norm.cpu().detach().clone()
                distance = output_f_ @ feature_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=2)
                score_near = score_bank[idx_near]  # batch x K x num_class
                score_near = score_near.permute(0, 2, 1)
                # print(score_near.shape)
                # print('score_near',score_near)

                # update banks
                feature_bank[idx] = output_f_.detach().clone().cpu()
                score_bank[idx] = outputs_softmax.detach().clone()
            # print(outputs_re.shape)
            # print('outputs_re',outputs_re)
            const = torch.log(torch.bmm(outputs_re, score_near)).sum(-1)
            # print(const)
            loss = -torch.mean(const)
            # print(loss)

            msoftmax = outputs_softmax.mean(dim=0)
            # print(msoftmax)
            gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            loss += gentropy_loss 

            # BP
            loss.backward()
            train_loss += loss.item()

            feature_shape = {'cnn': 640, 'resnet14': 64, 'resnet32': 64, 'resnet18_1D': 256}
            for n, p in backbone.bottleneck.named_parameters():
                if n.find('fc') != -1:
                    if n.find('bias') == -1:
                        mask_ = ((1 - masks)).view(-1, 1).expand(feature_shape[args.backbone_name], feature_shape[args.backbone_name]).cuda()
                        p.grad.data *= mask_
                    else:  # no bias here
                        mask_ = ((1 - masks)).squeeze().cuda()
                        p.grad.data *= mask_
                elif n.find('bn') != -1:
                    mask_ = ((1 - masks)).view(-1).cuda()
                    p.grad.data *= mask_

            for n, p in classifier.named_parameters():
                if n.find('weight_v') != -1:
                    masks__ = masks.view(1, -1).expand(args.nb_cl, feature_shape[args.backbone_name])
                    mask_ = ((1 - masks__)).cuda()
                    p.grad.data *= mask_

            backbone_optimizer.step()
            classifier_optimizer.step()
   
        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()

        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f} '.format(len(train_loader), train_loss/(batch_idx+1)))

        # Running the test for this epoch
        backbone.eval()
        classifier.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, instance in enumerate(test_loader):
                inputs, labels = instance[0].to(device), instance[1].to(device)
                features, masks = backbone(inputs, t=1, s=100, all_mask=False)
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
