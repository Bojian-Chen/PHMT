import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.ema import moving_weight, bn_statistics_moving_average, exponential_moving_average, cotta_ema
from utils.avgmeter import get_bn_statistics
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier 
import math
from scipy.spatial.distance import cdist

def T2PL(args, loader, backbone, classifier,topk_alpha, topk_beta):
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
    bool_index = ov > 0
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
    topk_num = topk_beta

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
    topk_fit_num = topk_beta
    topk_num = topk_alpha

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


def distill_knowledge(score, confidence_gate, temperature):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask


def ours_sc(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):
    beta=2
    best_acc = 0
    # reg_alpha = 0.5
    # rst = 0.1

    # initial_state_backbone = copy.deepcopy(student_backbone.state_dict())
    # initial_state_classifier = copy.deepcopy(student_classifier.state_dict())

    old_state_backbone = copy.deepcopy(student_backbone.state_dict())
    old_state_classifier = copy.deepcopy(student_classifier.state_dict())
    
    old_bn_statistics = get_bn_statistics(student_backbone.state_dict())
    old_prototypes = student_classifier.classifier.weight.clone().detach()

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    teacher_backbone = teacher_backbone.to(device)
    teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)


    for epoch in range(args.epochs):

        
        if args.dataset_name == 'iFlytek' or args.dataset_name == 'WT':
            confidence_gate= moving_weight(epoch, args.epochs, 0.3, 0.6) # for iFlytek
        elif args.dataset_name == 'SK':
            confidence_gate= moving_weight(epoch, args.epochs, 0.3, 0.6) # for SK

        teacher_backbone.eval()
        teacher_classifier.eval()

        mem_label = T2PL(args, train_loader, student_backbone, student_classifier, 8, math.floor(100*confidence_gate))
        mem_label = torch.from_numpy(mem_label).cuda()
        # confidence_gate = 0.3
        # reg_alpha = moving_weight(epoch, args.epochs, 1.0, 0.5)
        reg_alpha = 1.0
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

            pred = mem_label[idx].long()
            # train model_new
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # use teacher model to do predict
            # Mixup with teacher model
            # if args.mixup:
            #     with torch.no_grad():
            #         score = teacher_classifier(teacher_backbone(inputs))
            #         knowledge, knowledge_mask = distill_knowledge(score, confidence_gate, temperature=2)

            #     if beta > 0:
            #         lam = np.random.beta(beta, beta)
            #         lam = max(lam, 1 - lam)
            #     else:
            #         lam = 1
            #     batch_size = inputs.size(0)
            #     index = torch.randperm(batch_size).cuda()
            #     mixed_data = lam * inputs + (1 - lam) * inputs[index, :]
            #     mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]
            # else:
            #     with torch.no_grad():
            #         score = teacher_classifier(teacher_backbone(inputs))
            #         knowledge, knowledge_mask = distill_knowledge(score, 0, temperature=1)
            #     mixed_data = inputs
            #     mixed_consensus = knowledge

            # mixed_output = student_classifier(student_backbone(mixed_data))
            # mixed_softmax = torch.softmax(mixed_output, dim=1)
            # mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)

            features = student_backbone(inputs)
            outputs = student_classifier(features)
            softmax_output = torch.softmax(outputs, dim=1)

       
            # consistency_loss = torch.sum(
            #     knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)) / torch.sum(
            #     knowledge_mask)
            consistency_loss = nn.CrossEntropyLoss()(outputs, pred)
            if args.MI:
                # MI loss
                loss1 = Entropy(softmax_output)
                loss2 = DivEntropy(softmax_output)
                mutual_info_loss = loss1 - loss2
            else:
                mutual_info_loss = 0.0
            loss =  0.3*consistency_loss + reg_alpha * mutual_info_loss
            
            _, predicted = outputs.max(1)



            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            train_loss += loss.item()
 
            backbone_optimizer.step()
            classifier_optimizer.step()


        # if args.dual:
        new_bn_statistics = get_bn_statistics(student_backbone.state_dict())
        tao_begin = 0.95
        tao_end = 0.99
        bn_statistics_moving_average(old_bn_statistics, new_bn_statistics, epoch, args.epochs,tao_begin=tao_begin, tao_end=tao_end) #这个可以去掉，同时CoTTA的EMA并没有更新BN层的mean和var
        exponential_moving_average(teacher_backbone, student_backbone, epoch, args.epochs,tao_begin=tao_begin, tao_end=tao_end)
        exponential_moving_average(teacher_classifier, student_classifier, epoch, args.epochs,tao_begin=tao_begin, tao_end=tao_end)

        if args.SR:
            rst = moving_weight(epoch, args.epochs, 0.01, 0.05)
            for nm, m in teacher_backbone.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < rst).float().cuda()
                        with torch.no_grad():
                            p.data = old_state_backbone[f"{nm}.{npp}"] * mask + p * (1. - mask)
            for nm, m in teacher_classifier.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < rst).float().cuda()
                        with torch.no_grad():
                            p.data = old_state_classifier[f"{nm}.{npp}"] * mask + p * (1. - mask)
        else:
            pass

   
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
