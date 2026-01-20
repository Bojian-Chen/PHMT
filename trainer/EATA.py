
"""
Efficient Test-Time Model Adaptation without Forgetting
"""
import torch
import torch.nn as nn
import torch.jit
import math
import torch.nn.functional as F
from loss_function import *
import copy
import torch.optim as optim
from utils.Fisher import Fisher_BN

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def EATA(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler, fishers):
    e_margin = math.log(1000)*0.40
    d_margin = 0.05
    fisher_alpha=2000

    current_model_probs = None
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

            # outputs_softmax = nn.Softmax(dim=1)(outputs)
            
            # Entropy 
            entropys = softmax_entropy(outputs)

            filter_ids_1 = torch.where(entropys < e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0]>-0.1)
            entropys = entropys[filter_ids_1] 
            # filter redundant samples
            if current_model_probs is not None: 
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
            loss = entropys.mean(0)


            if fishers is not None:
                ewc_loss = 0
                for name, param in backbone.named_parameters():
                    if name in fishers:
                        ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
                loss += ewc_loss

           
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

    # fishers = Fisher_BN(best_backbone, best_classifier, train_loader)

    return best_backbone, best_classifier
