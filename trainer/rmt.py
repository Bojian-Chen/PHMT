"""
Continual Test-Time Domain Adaptation
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.ema import cotta_ema
import torchvision.transforms as transforms

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def get_cotta_transforms(gaussian_std: float = 0.005, soft=False):
    cotta_transforms = transforms.Compose([GaussianNoise(0, gaussian_std)])
    return cotta_transforms

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def rmt(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, source_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

    best_acc = 0
    aug_times=32
    rst=0.01
    ap=0.9
    

    src_prototypes = source_classifier.classifier.weight.clone().detach()
    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    teacher_backbone = teacher_backbone.to(device)
    teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)

    augmentations = get_cotta_transforms()
    for epoch in range(args.epochs):

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
            
       
            # reset grad
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            with torch.no_grad():
                score_t = teacher_classifier(teacher_backbone(inputs))

            feature_s = student_backbone(inputs)
            score_s = student_classifier(feature_s)
            feature_aug_s = student_backbone(augmentations(inputs))
            score_aug_s = student_classifier(feature_aug_s)

            features = torch.cat([feature_s, feature_aug_s], dim=0)

            contrast_loss = Unsupervised_Prototype_InfoNCE_loss(src_prototypes, features)

            self_training_loss = 0.25 * nn.CrossEntropyLoss()(score_s, score_t) + 0.25 * nn.CrossEntropyLoss()(score_aug_s, score_t)

            loss = self_training_loss + contrast_loss*0




            # build labels with teacher model and dataaug
            # with torch.no_grad():
            #     score_t = teacher_classifier(teacher_backbone(inputs))
            #     p_t = score_t.softmax(1)
            #     anchor_prob = p_t.max(1)[0]
            #     if anchor_prob.mean(0) < ap:
            #         # do augmentation 32 times
            #         score_augments = []
            #         for _ in range(aug_times):
            #             score_aug_ = teacher_classifier(teacher_backbone(augmentations(inputs))).detach()
            #             score_augments.append(score_aug_)

            #         score_t = torch.stack(score_augments).mean(0)
            #     else:
            #         pass
            # score_s = student_classifier(student_backbone(inputs))
            # loss = softmax_entropy(score_s, score_t).mean()
            # P_InfoNCE_loss


            _, predicted = score_s.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            train_loss += loss.item()
 
            backbone_optimizer.step()
            classifier_optimizer.step()



        teacher_backbone = cotta_ema(teacher_backbone, student_backbone)
        teacher_classifier = cotta_ema(teacher_classifier, student_classifier)

        student_backbone.load_state_dict(teacher_backbone.state_dict())
        student_classifier.load_state_dict(teacher_classifier.state_dict())

        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()
        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f} '.format(
            len(train_loader), train_loss/(batch_idx+1),  100.*correct/total))

        # Running the test for this epoch
        student_backbone.eval()
        student_classifier.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, instance in enumerate(test_loader):
                inputs, labels = instance[0].to(device), instance[1].to(device)
                features = student_backbone(inputs)
                outputs = student_classifier(features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), test_loss/(batch_idx+1), 100.*correct/total))

        if 100.*correct/total >= best_acc:
            best_acc = 100.*correct/total
            best_backbone = copy.deepcopy(student_backbone)
            best_classifier = copy.deepcopy(student_classifier)

    return best_backbone, best_classifier
