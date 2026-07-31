

import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
import csv
import os
from datetime import datetime
from utils.RandMix import RandMix

def _get_metrics_prefix(args):
    if args.incremental_mode == 'ours_new':
        return 'ours_new_stage_metrics'
    if args.incremental_mode == 'ours_simple':
        return 'ours_simple_stage_metrics'
    return None

def _build_source_metrics_csv_path(args):
    base_dir = args.pth
    os.makedirs(base_dir, exist_ok=True)
    dataset_name = args.dataset_name
    random_seed = args.random_seed
    session = args.session
    domain = args.Domain_Seq[session]
    prefix = _get_metrics_prefix(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{dataset_name}_seed{random_seed}_session{session}_domain{domain}_{timestamp}.csv"
    return os.path.join(base_dir, filename)

def _save_metrics_to_csv(csv_path, metrics_history):
    if not metrics_history:
        return
    fieldnames = list(metrics_history[0].keys())
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)


def source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

    best_acc = 0
    log_source_metrics = (args.incremental_mode == 'ours_new' or args.incremental_mode == 'ours_simple')
    metrics_history = []
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
        avg_train_loss = train_loss / (batch_idx + 1)
        train_accuracy = 100. * correct / total
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f}'.format(len(train_loader), avg_train_loss,  train_accuracy))


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
            
        avg_test_loss = test_loss / (batch_idx + 1)
        test_accuracy = 100. * correct / total
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), avg_test_loss, test_accuracy))
        if log_source_metrics:
            metrics_history.append({
                "stage_type": "source_pretrain",
                "session": args.session,
                "domain": int(args.Domain_Seq[args.session]),
                "epoch": epoch,
                "backbone_learning_rate": backbone_scheduler.get_last_lr()[0],
                "classifier_learning_rate": classifier_scheduler.get_last_lr()[0],
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": avg_test_loss,
                "test_accuracy": test_accuracy,
            })

        if test_accuracy >= best_acc:
            best_acc = test_accuracy
            best_backbone = copy.deepcopy(backbone)
            best_classifier = copy.deepcopy(classifier)

    if log_source_metrics:
        metrics_csv_path = _build_source_metrics_csv_path(args)
        _save_metrics_to_csv(metrics_csv_path, metrics_history)
        print(f"Saved source pretrain metrics to: {metrics_csv_path}")
    return best_backbone, best_classifier
