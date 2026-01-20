"""
Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation?
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from scipy.spatial.distance import cdist

def obtain_label(loader, backbone, classifier, threshold, distance_type="cosine"):
    start_test = True

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            inputs, labels, *_ = batch
            inputs = inputs.cuda()
            feas = backbone(inputs)
            outputs = classifier(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    if distance_type == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], distance_type)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]
    return predict.astype('int')

def shot(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler):

    best_acc = 0

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    backbone = backbone.to(device)
    classifier = classifier.to(device) 


    for epoch in range(args.epochs):
        # Set the model to the training mode

        classifier.eval()
        backbone.eval()
        mem_label = obtain_label(train_loader, backbone, classifier, 0)
        mem_label = torch.from_numpy(mem_label).cuda()

        backbone.train()

        train_loss = 0
        train_loss_entropy = 0
        train_loss_diventropy = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx, instance in enumerate(train_loader):
            # Get a batch of training samples, transfer them to the device
            inputs, labels = instance[0].to(device), instance[1].to(device)
            idx = instance[2]
       
            # train model_new
            backbone_optimizer.zero_grad()
     
            
            # Forward the samples in the deep networks
            features = backbone(inputs)
            outputs = classifier(features)
            outputs_softmax = nn.Softmax(dim=1)(outputs)

            pred = mem_label[idx].long()
            classifier_loss = nn.CrossEntropyLoss()(outputs, pred)
    
            # Entropy Loss
            loss1 = Entropy(outputs_softmax)
           
            # DivEntropy Loss
            loss2 = DivEntropy(outputs_softmax)

            loss_mi = loss1 - loss2

            loss = loss_mi  + classifier_loss * 0.3 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
      

            # BP
            loss.backward()
            train_loss += loss.item()
            train_loss_entropy  += loss1.item()
            train_loss_diventropy  += loss2.item()

            backbone_optimizer.step()

   
        # Learning rate decay
        backbone_scheduler.step()

        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f} train loss Entropy {:.4f} train loss DivEntropy {:.4f} accuracy: {:.4f} '.format(
            len(train_loader), train_loss/(batch_idx+1), train_loss_entropy/(batch_idx+1), train_loss_diventropy/(batch_idx+1), 100.*correct/total))

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
