"""
CoSDA: Continual Source-Free Domain Adaptation
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.ema import moving_weight, bn_statistics_moving_average, exponential_moving_average
from utils.avgmeter import get_bn_statistics
import torchvision.transforms as transforms


def distill_knowledge(score, confidence_gate, temperature=0.07):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask


def CoSDA(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):
    # JAS SK 2.0 1.0 other 1.0 0.1 
    # TNNLS SK 2.0 0.5  other 2.0 0.3
    beta=2.0  # SK 2.0  
    reg_alpha = 0.3 # SK 0.5

    best_acc = 0
    # confidence_gate=0.8

    source_bn_statistics = get_bn_statistics(student_backbone.state_dict())
    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    teacher_backbone = teacher_backbone.to(device)
    teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)
    
    for epoch in range(args.epochs):

        temperature = 0.07

        confidence_gate= moving_weight(epoch, args.epochs, 0.9, 0.99)
        # Set the model to the training mode
        teacher_backbone.train()
        teacher_classifier.train()
        student_backbone.train()
        student_classifier.train()

        train_loss = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx, instance in enumerate(train_loader):
            # Get a batch of training samples, transfer them to the device
            inputs, labels = instance[0].to(device), instance[1].to(device)
       
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # use teacher model to do predict
            with torch.no_grad():
                score = teacher_classifier(teacher_backbone(inputs))
                knowledge, knowledge_mask = distill_knowledge(score, confidence_gate, temperature=temperature)
            if beta > 0:
                lam = np.random.beta(beta, beta)
                # set high lamb in the left
                lam = max(lam, 1 - lam)
            else:
                lam = 1
            batch_size = inputs.size(0)
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * inputs + (1 - lam) * inputs[index, :]
            mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]
            mixed_output = student_classifier(student_backbone(mixed_image))
            mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)
            consistency_loss = torch.sum(
                knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)) / torch.sum(
                knowledge_mask)
            # set regularization
            output = student_classifier(student_backbone(inputs))
            softmax_output = torch.softmax(output, dim=1)

            margin_output = torch.mean(softmax_output, dim=0)
            log_softmax_output = torch.log_softmax(output, dim=1)
            log_margin_output = torch.log(margin_output + 1e-5)
            mutual_info_loss = -1 * torch.mean(
                torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1))
            
            loss = consistency_loss + reg_alpha * mutual_info_loss

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            train_loss += loss.item()
 
            backbone_optimizer.step()
            classifier_optimizer.step()

        target_bn_statistics = get_bn_statistics(student_backbone.state_dict())
        bn_statistics_moving_average(source_bn_statistics, target_bn_statistics, epoch, args.epochs,tao_begin=0.9, tao_end=0.99)
        # perform moving average for model weights
        exponential_moving_average(teacher_backbone, student_backbone, epoch, args.epochs,tao_begin=0.9, tao_end=0.99)
        exponential_moving_average(teacher_classifier, student_classifier, epoch, args.epochs,tao_begin=0.9, tao_end=0.99)

        # student_backbone.load_state_dict(teacher_backbone.state_dict())
        # student_classifier.load_state_dict(teacher_classifier.state_dict())

        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()
        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f} '.format(
            len(train_loader), train_loss/(batch_idx+1),  100.*correct/total))

        # Running the test for this epoch
        teacher_backbone.eval()
        teacher_classifier.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, instance in enumerate(test_loader):
                inputs, labels = instance[0].to(device), instance[1].to(device)
                features = teacher_backbone(inputs)
                outputs = teacher_classifier(features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), test_loss/(batch_idx+1), 100.*correct/total))

        if 100.*correct/total >= best_acc:
            best_acc = 100.*correct/total
            best_backbone = copy.deepcopy(teacher_backbone)
            best_classifier = copy.deepcopy(teacher_classifier)

    return best_backbone, best_classifier
