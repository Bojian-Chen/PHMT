"""
DEJA VU: CONTINUAL MODEL GENERALIZATION FOR UNSEEN DOMAINS
"""
import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.ema import moving_weight, bn_statistics_moving_average, exponential_moving_average, cotta_ema
from utils.avgmeter import get_bn_statistics
import torchvision.transforms as transforms
from utils.RandMix import RandMix
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier 

def T2PL(args, loader, backbone, classifier,topk_alpha, topk_beta, pseudo_tau):
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
    ov, idx = torch.max(all_output, 1)
    bool_index = ov > pseudo_tau
    all_output = all_output[bool_index]
    all_fea = all_fea[bool_index]
    all_label = all_label[bool_index]
    
    acc_list = []
    
    # softmax predict
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    acc_list.append(accuracy)
    
    all_fea = all_fea / torch.norm(all_fea, p=2, dim=1, keepdim=True)
    
    all_fea = all_fea.float().cpu()  # (N, dim)
    K = all_output.size(1)
    aff = all_output.float().cpu()   # (N, C)
    
    # top k features for SHOT
    topk_num = max(all_fea.shape[0] // (args.nb_cl * topk_beta), 1) # 1000/10/2=50

    top_aff, top_fea = [], []
        
    for cls_idx in range(args.nb_cl):
        feat_samp_idx = torch.topk(aff[:, cls_idx], topk_num)[1]                
        top_fea.append(all_fea[feat_samp_idx, :])        
        top_aff.append(aff[feat_samp_idx, :])
        
    top_aff = torch.cat(top_aff, dim=0).numpy()
    top_fea = torch.cat(top_fea, dim=0).numpy()
    _, top_predict = torch.max(torch.from_numpy(top_aff), 1)
    
    # SHOT      
    for _ in range(2):
        initc = top_aff.transpose().dot(top_fea)  
        initc = initc / (1e-8 + top_aff.sum(axis=0)[:,None])  

        cls_count = np.eye(K)[predict].sum(axis=0)   
        labelset = np.where(cls_count>0)    
        labelset = labelset[0]    

        dd = cdist(all_fea, initc[labelset], metric = "cosine")   
        pred_label = dd.argmin(axis=1)   
        predict = labelset[pred_label]   
        
        top_cls_count = np.eye(K)[top_predict].sum(axis=0)   
        top_labelset = np.where(top_cls_count>0)    
        top_labelset = top_labelset[0]    

        top_dd = cdist(top_fea, initc[top_labelset], metric = "cosine") 
        top_pred_label = top_dd.argmin(axis=1)   
        top_predict = top_labelset[top_pred_label]   

        top_aff = np.eye(K)[top_predict]         
        acc_list.append(np.sum(predict == all_label.float().numpy()) / len(all_fea))
        
    # knn on distance of each features and cluster center
    top_sample = []
    top_label = []
    topk_fit_num = max(all_fea.shape[0] // (args.nb_cl * topk_beta ), 1) # 1000/10/2=50
    topk_num = max(all_fea.shape[0] // (args.nb_cl*topk_alpha ), 1) # 1000/10/20=5

    for cls_idx in range(len(labelset)):     
        feat_samp_idx = torch.topk(torch.from_numpy(dd)[:, cls_idx], topk_fit_num, largest=False )[1]
            
        feat_cls_sample = all_fea[feat_samp_idx, :]
        feat_cls_label = torch.zeros([len(feat_samp_idx)]).fill_(cls_idx)

        top_sample.append(feat_cls_sample)
        top_label.append(feat_cls_label)
    top_sample = torch.cat(top_sample, dim=0).cpu().numpy()
    top_label = torch.cat(top_label, dim=0).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=topk_num, metric='cosine')
    knn.fit(top_sample, top_label)
    
    knn_predict = knn.predict(all_fea.cpu().numpy()).tolist()
    knn_predict = [int(i) for i in knn_predict]
    
    predict = labelset[knn_predict]
    acc_list.append(np.sum(predict == all_label.float().numpy()) / len(all_fea))
        
    print("acc:" + " --> ".join("{:.3f}".format(acc) for acc in acc_list))
    acc_dict = {}
    for i in range(len(acc_list)):
        acc_dict['pa{}'.format(i)] = round(acc_list[i],3)

    return predict.astype('int')


def select_aug(backbone,classifier, all_x, all_y, epoch):
    ratio = epoch / 40
    backbone.eval()
    classifier.eval()
    with torch.no_grad():
        pred = nn.Softmax(dim=1)(classifier(backbone(all_x)))
        ov, idx = torch.max(pred, 1)
        bool_index = ov > 0.8
        data_fore = all_x[bool_index]
        y_fore = all_y[bool_index]
        RandMix_Aug = RandMix(1).cuda()
        data_aug = RandMix_Aug(data_fore, ratio=ratio)

    backbone.train()
    classifier.train()
    all_x = torch.cat([all_x, data_aug])   
    all_y = torch.cat([all_y, y_fore])

    return all_x, all_y

def RaTP(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):

    best_acc = 0



    old_prototypes = teacher_classifier.classifier.weight.clone().detach()

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    teacher_backbone = teacher_backbone.to(device)
    teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)
    teacher_backbone.eval()
    teacher_classifier.eval()
    for epoch in range(args.epochs):

        # Get the pseudo label
        student_backbone.eval()
        student_classifier.eval()

        mem_label = T2PL(args, train_loader, student_backbone, student_classifier, 20,2,0)
        # T2PL(args, loader, backbone, classifier, pseudo_tau):
        mem_label = torch.from_numpy(mem_label).cuda()

        

        # Set the model to the training mode
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
            idx = instance[2]
            labels_pseudo = mem_label[idx].long()
   
            inputs_aug, labels_pseudo = select_aug(student_backbone, student_classifier, inputs, labels_pseudo, epoch)

            student_backbone.train()
            student_classifier.train()

            old_features = teacher_backbone(inputs_aug)
            old_outputs = teacher_classifier(old_features)
            # train model_new
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()


            features = student_backbone(inputs_aug)
            outputs = student_classifier(features)


            softmax_output = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)


            loss = nn.CrossEntropyLoss()(outputs, labels_pseudo)
      
            loss += PCALoss(args.nb_cl, 0.07)(features, labels_pseudo, student_classifier.classifier.weight, old_prototypes, mweight=1)

            # loss += nn.KLDivLoss(reduction='batchmean')(torch.log(softmax_output), torch.softmax(old_outputs, dim=1))

            loss += nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(features), nn.Softmax(dim=1)(old_features)) * 0.5


            total += labels_pseudo.size(0)
            correct += predicted.eq(labels_pseudo).sum().item()

            loss.backward()
            train_loss += loss.item()
 
            backbone_optimizer.step()
            classifier_optimizer.step()


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
