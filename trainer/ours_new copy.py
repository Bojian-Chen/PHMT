import torch
import numpy as np
import torch.nn as nn
from loss_function import *
import copy
from utils.ema import moving_weight, bn_statistics_moving_average, exponential_moving_average, cotta_ema
from utils.avgmeter import get_bn_statistics
import torchvision.transforms as transforms
from utils.eval import *

# @torch.no_grad()
# def init_fisher(model: nn.Module):
#     """Diagonal Fisher buffer (same shape as params)."""
#     fisher = {}
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             fisher[name] = torch.zeros_like(p.data, dtype=torch.float32)
#     return fisher


# @torch.no_grad()
# def update_fisher_from_grads(model: nn.Module, fisher: dict, rho: float = 0.9):
#     """
#     Online diagonal Fisher (EMA of grad^2):
#         F <- rho*F + (1-rho)*g^2
#     """
#     for name, p in model.named_parameters():
#         if (not p.requires_grad) or (p.grad is None):
#             continue
#         g2 = (p.grad.detach().float() ** 2)
#         fisher[name].mul_(rho).add_(g2, alpha=(1.0 - rho))


# @torch.no_grad()
# def fisher_weighted_sr(teacher: nn.Module, old_state: dict, fisher: dict, rst: float, eps: float = 1e-8):
#     """
#     Fisher-weighted SR:
#       prob_k ∝ fisher_k  (normalized), then sample mask and recover to old_state.
#       prob = clamp(rst * fisher / mean(fisher), 0, 1)
#     """
#     for name, p in teacher.named_parameters():
#         if (not p.requires_grad) or (name not in fisher) or (name not in old_state):
#             continue
#         F = fisher[name].to(p.device)
#         # normalize for stable scale
#         prob = (rst * (F / (F.mean() + eps))).clamp(0.0, 1.0)
#         mask = (torch.rand_like(p.data) < prob).to(p.data.dtype)
#         old = old_state[name].to(p.device)
#         p.data = old * mask + p.data * (1.0 - mask)


def distill_knowledge(score, confidence_gate, temperature):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask


def ours_new(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler):
    beta=2
    best_acc = 0
    
    rst_min = 0.01
    rst_max = 0.05
    rst_ema_rho = 0.9


    # reg_alpha = 0.5
    # rst = 0.1

    # initial_state_backbone = copy.deepcopy(student_backbone.state_dict())
    # initial_state_classifier = copy.deepcopy(student_classifier.state_dict())

    # Setting up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    teacher_backbone = teacher_backbone.to(device)
    teacher_classifier = teacher_classifier.to(device)
    student_backbone = student_backbone.to(device)
    student_classifier = student_classifier.to(device)


    old_state_backbone = copy.deepcopy(student_backbone.state_dict())
    old_state_classifier = copy.deepcopy(student_classifier.state_dict())
    
    old_bn_statistics = get_bn_statistics(student_backbone.state_dict())
    old_prototypes = student_classifier.classifier.weight.clone().detach()

    # fisher_rho = getattr(args, "fisher_rho", 0.9)
    # fisher_backbone = init_fisher(student_backbone)
    # fisher_classifier = init_fisher(student_classifier)

    rst_prev = None
    tao_prev = None

    for epoch in range(args.epochs):
        if args.dataset_name == 'iFlytek' or args.dataset_name == 'WT':
            confidence_gate= moving_weight(epoch, args.epochs, 0.9, 0.95) # for iFlytek
        elif args.dataset_name == 'SK':
            confidence_gate= moving_weight(epoch, args.epochs, 0.5, 0.8) # for SK
            # confidence_gate= moving_weight(epoch, args.epochs, args.confidence_gate_start, args.confidence_gate_end) # for SK

        epoch_ent_sum = 0.0
        epoch_ent_n = 0
        epoch_num_classes = None

        # confidence_gate = 0.3
        # reg_alpha = moving_weight(epoch, args.epochs, 1.0, 0.5)
        reg_alpha = 1.0
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
       
            # train model_new
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # use teacher model to do predict
            # Mixup with teacher model
            if args.mixup:
                with torch.no_grad():
                    score = teacher_classifier(teacher_backbone(inputs))

                    prob_t = torch.softmax(score, dim=1)
                    ent = -torch.sum(prob_t * torch.log(prob_t + 1e-12), dim=1)  # [B]
                    epoch_ent_sum += ent.sum().item()
                    epoch_ent_n += ent.numel()
                    epoch_num_classes = score.size(1)

                    knowledge, knowledge_mask = distill_knowledge(score, confidence_gate, temperature=2)

                if beta > 0:
                    lam = np.random.beta(beta, beta)
                    lam = max(lam, 1 - lam)
                else:
                    lam = 1
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).cuda()
                mixed_data = lam * inputs + (1 - lam) * inputs[index, :]
                mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]
            else:
                with torch.no_grad():
                    score = teacher_classifier(teacher_backbone(inputs))

                    prob_t = torch.softmax(score, dim=1)
                    ent = -torch.sum(prob_t * torch.log(prob_t + 1e-12), dim=1)  # [B]
                    epoch_ent_sum += ent.sum().item()
                    epoch_ent_n += ent.numel()
                    epoch_num_classes = score.size(1)

                    knowledge, knowledge_mask = distill_knowledge(score, 0, temperature=1)
                mixed_data = inputs
                mixed_consensus = knowledge

            mixed_output = student_classifier(student_backbone(mixed_data))
            mixed_softmax = torch.softmax(mixed_output, dim=1)
            mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)

            features = student_backbone(inputs)
            output = student_classifier(features)
            softmax_output = torch.softmax(output, dim=1)

       
            consistency_loss = torch.sum(
                knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)) / torch.sum(
                knowledge_mask)

            if args.MI:
                # MI loss
                loss1 = Entropy(softmax_output)
                loss2 = DivEntropy(softmax_output)
                mutual_info_loss = loss1 - loss2
            else:
                mutual_info_loss = 0.0
            loss =  consistency_loss + reg_alpha * mutual_info_loss
            
            _, predicted = output.max(1)



            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss.backward()

            # update_fisher_from_grads(student_backbone, fisher_backbone, rho=fisher_rho)
            # update_fisher_from_grads(student_classifier, fisher_classifier, rho=fisher_rho)

            train_loss += loss.item()
 
            backbone_optimizer.step()
            classifier_optimizer.step()

       
        if (epoch_ent_n > 0) and (epoch_num_classes is not None):
            bar_H = epoch_ent_sum / epoch_ent_n          # mean entropy
            C = float(epoch_num_classes)
            H_max = float(torch.log(torch.tensor(C)).item())  # log(C)

            # normalized uncertainty u in [0,1]
            u = bar_H / max(1e-8, H_max)
            u = max(0.0, min(1.0, u))

            # confidence = 1 - uncertainty, higher confidence -> larger rst
            conf = 1.0 - u
            rst_target = rst_min + (rst_max - rst_min) * conf
        else:
            rst_target = rst_min  # fallback

        # optional EMA smoothing to reduce jitter
        if rst_prev is None:
            rst = rst_target
        else:
            rst = rst_ema_rho * rst_prev + (1.0 - rst_ema_rho) * rst_target
        rst_prev = rst
        # rst = rst_target


        # if args.dual:
        new_bn_statistics = get_bn_statistics(student_backbone.state_dict())
        tao_begin = 0.95
        tao_end = 0.99
        tao_ema_rho = 0.9

        tao_target = tao_begin + (tao_end - tao_begin) * conf
        tao_target = max(0.0, min(0.9999, tao_target))  # safety clip

        if tao_prev is None:
            tao = tao_target
        else:
            tao = tao_ema_rho * tao_prev + (1.0 - tao_ema_rho) * tao_target
        tao_prev = tao
        tao = tao_target

        print(f"Epoch {epoch}: confidence={conf:.4f} rst={rst:.4f}, tao={tao:.4f}")

        # if args.SR:
        #     # --- Fisher-weighted SR (recover important params more often) ---
        #     fisher_weighted_sr(student_backbone, old_state_backbone, fisher_backbone, rst=rst)
        #     fisher_weighted_sr(student_classifier, old_state_classifier, fisher_classifier, rst=rst)
        # tao_begin = args.tao_begin
        # tao_end = args.tao_end
        bn_statistics_moving_average(old_bn_statistics, new_bn_statistics, epoch, args.epochs,tao_begin=tao, tao_end=tao) #这个可以去掉，同时CoTTA的EMA并没有更新BN层的mean和var
        exponential_moving_average(teacher_backbone, student_backbone, epoch, args.epochs,tao_begin=tao, tao_end=tao)
        exponential_moving_average(teacher_classifier, student_classifier, epoch, args.epochs,tao_begin=tao, tao_end=tao)

        if args.SR:
            # rst = moving_weight(epoch, args.epochs, 0.01, 0.05)
            # rst = moving_weight(epoch, args.epochs, args.sr_start, args.sr_end)
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

        # if args.SR:


        #     # --- Fisher-weighted SR (recover important params more often) ---
        #     fisher_weighted_sr(teacher_backbone, old_state_backbone, fisher_backbone, rst=rst)
        #     fisher_weighted_sr(teacher_classifier, old_state_classifier, fisher_classifier, rst=rst)

   
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

        # Save the best model
        if 100.*correct/total >= best_acc:
            best_acc = 100.*correct/total
            best_backbone = copy.deepcopy(student_backbone)
            best_classifier = copy.deepcopy(student_classifier)
        # with torch.no_grad():
        #     acc_list = [100.0, 94.8, 99.3, 99.4, 99.8]
        #     bwt = []
        #     for k in range(args.session):
        #         c = evaluate_during_train(args, student_backbone, student_classifier, k)
        #         bwt.append(c-acc_list[k])
        #     bwt = np.mean(bwt)
        #     print('BWT: {:.4f}'.format(bwt))

    return best_backbone, best_classifier
