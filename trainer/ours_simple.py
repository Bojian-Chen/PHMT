import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
import csv
import os
from datetime import datetime
from utils.ema import moving_weight, bn_statistics_moving_average, exponential_moving_average, cotta_ema
from utils.avgmeter import get_bn_statistics
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier 
import math

def T2PL(args, loader, backbone, classifier,topk_alpha, topk_beta, return_stage_acc=False):
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

    if return_stage_acc:
        return predict.astype('int'), [float(acc) for acc in acc_list]
    return predict.astype('int')

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

def distill_knowledge(score, confidence_gate, temperature=0.07):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask

def _build_metrics_csv_path(args):
    base_dir = args.pth
    os.makedirs(base_dir, exist_ok=True)
    dataset_name = args.dataset_name
    random_seed = args.random_seed
    session = args.session
    domain = args.Domain_Seq[session]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ours_simple_stage_metrics_{dataset_name}_seed{random_seed}_session{session}_domain{domain}_{timestamp}.csv"
    return os.path.join(base_dir, filename)

def _save_metrics_to_csv(csv_path, metrics_history):
    if not metrics_history:
        return
    fieldnames = list(metrics_history[0].keys())
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)



def ours_simple(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, source_backbone, train_loader, test_loader, backbone_optimizer,backbone_scheduler):
    # beta=2
    best_acc = 0
    # reg_alpha = 0.5
    rst = 0.1

    initial_state_backbone = copy.deepcopy(source_backbone.state_dict())
    old_state_backbone = copy.deepcopy(student_backbone.state_dict())
    old_state_classifier = copy.deepcopy(student_classifier.state_dict())
    old_prototypes = teacher_classifier.classifier.weight.clone().detach()
    metrics_history = []

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    # teacher_backbone = teacher_backbone.to(device)
    # teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)
    best_backbone = copy.deepcopy(student_backbone)
    best_classifier = copy.deepcopy(student_classifier)

    for epoch in range(args.epochs):
        # confidence_gate= moving_weight(epoch, args.epochs, 0.7, 0.95)
        confidence_gate= moving_weight(epoch, args.epochs, 0.3, 0.6)
        # r_k = int(30 + (60 - 30) * (epoch / args.epochs))

        # confidence_gate = 0.3
        # reg_alpha = moving_weight(epoch, args.epochs, 1.0, 0.5)
        # Set the model to the training mode
        # teacher_backbone.eval()
        # teacher_classifier.eval()
        student_backbone.eval()
        student_classifier.eval()
        
        # mem_label = obtain_label(train_loader, student_backbone, student_classifier, 0.5)
        pseudo_stage_acc = [float("nan")] * 4
        if args.TOPK:
            mem_label, pseudo_stage_acc = T2PL(
                args, train_loader, student_backbone, student_classifier, 8,
                math.floor(100*confidence_gate), return_stage_acc=True
            )
        else:
            mem_label = obtain_label(train_loader, student_backbone, student_classifier, 0)
        
        mem_label = torch.from_numpy(mem_label).cuda()

        student_backbone.train()
        student_classifier.train()

        train_loss = 0
        classifier_loss_sum = 0
        mi_loss_sum = 0
        pca_loss_sum = 0

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

            features = student_backbone(inputs)
            outputs = student_classifier(features)
            # knowledge, knowledge_mask = distill_knowledge(outputs, confidence_gate, temperature=2)

            _, predicted = outputs.max(1)
            softmax_outputs = torch.softmax(outputs, dim=1)


            ### Con
            # classifier_loss = torch.sum(
            # knowledge_mask * torch.sum(-1 * knowledge * softmax_outputs, dim=1)) / torch.sum(
            # knowledge_mask)

            ### Pesudo Label Loss
            classifier_loss = nn.CrossEntropyLoss()(outputs, pred)


            # max_p, max_p_class = softmax_outputs.max(1)
            # knowledge_mask = (max_p > confidence_gate).float().cuda()

            # log_softmax = torch.log_softmax(outputs, dim=1)
            # classifier_loss = torch.sum(
            # knowledge_mask * torch.sum(-1 * softmax_outputs * log_softmax, dim=1)) / torch.sum(
            # knowledge_mask)




            # Entropy Loss
            loss1 = Entropy(softmax_outputs)
            # DivEntropy Loss
            loss2 = DivEntropy(softmax_outputs)

            mutual_info_loss = loss1 - loss2

            ###  Prototype contrastive knowledge distillation
            pcaloss = PCALoss(args.nb_cl, 0.07)(features, predicted, student_classifier.classifier.weight, old_prototypes, mweight=1) 

            # old_features = teacher_backbone(inputs)
            # old_outputs = teacher_classifier(old_features)
            # old_softmax_outputs = torch.softmax(old_outputs, dim=1)
            # # loss += nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(features), nn.Softmax(dim=1)(old_features)) 
            # distill_loss = nn.KLDivLoss(reduction="batchmean")( torch.softmax(outputs / 2, dim=1) ,  torch.softmax(old_outputs / 2, dim=1))
            loss = 0.3 * classifier_loss
            if args.MI:
                loss +=  mutual_info_loss 
            if args.PCA:
                loss += pcaloss * 0.1

            # else:
            #     loss =  0.3 * classifier_loss + pcaloss * 0.1

            # if args.PCA:
            #     loss =  0.3 * classifier_loss +  mutual_info_loss + pcaloss * 0.1
            # else:
            #     loss =  0.3 * classifier_loss + mutual_info_loss



            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            train_loss += loss.item()
            classifier_loss_sum += float(classifier_loss.item())
            mi_loss_sum += float(mutual_info_loss.item())
            pca_loss_sum += float(pcaloss.item())
 
            backbone_optimizer.step()


            # if epoch > 30:
            # for nm, m in student_backbone.named_modules():
            #     for npp, p in m.named_parameters():
            #         if npp in ['weight', 'bias'] and p.requires_grad:
            #             mask = (torch.rand(p.shape) < rst).float().cuda()
            #             with torch.no_grad():
            #                 p.data = initial_state_backbone[f"{nm}.{npp}"] * mask + p * (1. - mask)

            if args.SR:
                for nm, m in student_backbone.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape) < rst).float().cuda()
                            with torch.no_grad():
                                p.data = old_state_backbone[f"{nm}.{npp}"] * mask + p * (1. - mask)

                for nm, m in student_classifier.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape) < rst).float().cuda()
                            with torch.no_grad():
                                p.data = old_state_classifier[f"{nm}.{npp}"] * mask + p * (1. - mask)
            else:
                #不执行任何操作
                pass


        # Learning rate decay
        backbone_scheduler.step()
        avg_train_loss = train_loss / (batch_idx + 1)
        train_accuracy = 100. * correct / total
        avg_classifier_loss = classifier_loss_sum / (batch_idx + 1)
        avg_mi_loss = mi_loss_sum / (batch_idx + 1)
        avg_pca_loss = pca_loss_sum / (batch_idx + 1)

        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f} '.format(
            len(train_loader), avg_train_loss,  train_accuracy))

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

        avg_test_loss = test_loss / (batch_idx + 1)
        test_accuracy = 100. * correct / total
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), avg_test_loss, test_accuracy))

        metrics_history.append({
            "stage_type": "target_adapt",
            "session": args.session,
            "domain": int(args.Domain_Seq[args.session]),
            "epoch": epoch,
            "learning_rate": backbone_scheduler.get_last_lr()[0],
            "confidence_gate": confidence_gate,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "train_classifier_loss": avg_classifier_loss,
            "train_mi_loss": avg_mi_loss,
            "train_pca_loss": avg_pca_loss,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "pseudo_softmax_accuracy": pseudo_stage_acc[0],
            "pseudo_shot_iter1_accuracy": pseudo_stage_acc[1],
            "pseudo_shot_iter2_accuracy": pseudo_stage_acc[2],
            "pseudo_knn_accuracy": pseudo_stage_acc[3],
        })

        if test_accuracy >= best_acc:
            best_acc = test_accuracy
            best_backbone = copy.deepcopy(student_backbone)
            best_classifier = copy.deepcopy(student_classifier)

    metrics_csv_path = _build_metrics_csv_path(args)
    _save_metrics_to_csv(metrics_csv_path, metrics_history)
    print(f"Saved stage metrics to: {metrics_csv_path}")

    return best_backbone, best_classifier
