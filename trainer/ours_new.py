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
from utils.eval import *
from utils.Fisher import Fisher, Fisher_Entropy

@torch.no_grad()
def init_fisher(model: nn.Module):
    """Diagonal Fisher buffer (same shape as params)."""
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p.data, dtype=torch.float32)
    return fisher


@torch.no_grad()
def update_fisher_from_grads(model: nn.Module, fisher: dict, rho: float = 0.9):
    """
    Online diagonal Fisher (EMA of grad^2):
        F <- rho*F + (1-rho)*g^2
    """
    for name, p in model.named_parameters():
        if (not p.requires_grad) or (p.grad is None):
            continue
        g2 = (p.grad.detach().float() ** 2)
        fisher[name].mul_(rho).add_(g2, alpha=(1.0 - rho))


@torch.no_grad()
def fisher_weighted_sr(model: nn.Module, old_state: dict, fisher: dict, rst: float, eps: float = 1e-8):
    """
    Fisher-weighted SR:
      prob_k ∝ fisher_k  (normalized), then sample mask and recover to old_state.
      prob = clamp(rst * fisher / mean(fisher), 0, 1)
    """
    for name, p in model.named_parameters():
        if (not p.requires_grad) or (name not in fisher) or (name not in old_state):
            continue
        F = fisher[name][0].to(p.device)
        # normalize for stable scale
        prob = (rst * (F / (F.mean() + eps))).clamp(0.0, 1.0)
        mask = (torch.rand_like(p.data) < prob).to(p.data.dtype)
        old = old_state[name].to(p.device)
        p.data = old * mask + p.data * (1.0 - mask)



def distill_knowledge(score, confidence_gate, temperature):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask

def distill_knowledge_by_entropy(score, confidence_gate, temperature):
    # print("confidence_gate:", confidence_gate)
    prob = torch.softmax(score, dim=1)
    ent = -torch.sum(prob * torch.log(prob + 1e-12), dim=1)  # [B]
    knowledge_mask = (ent < confidence_gate).float().cuda()
    # print("knowledge_mask sum:", knowledge_mask.sum().item())
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
    filename = f"ours_new_stage_metrics_{dataset_name}_seed{random_seed}_session{session}_domain{domain}_{timestamp}.csv"
    return os.path.join(base_dir, filename)

def _save_metrics_to_csv(csv_path, metrics_history):
    if not metrics_history:
        return
    fieldnames = list(metrics_history[0].keys())
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)

def ours_new(args, teacher_backbone, teacher_classifier, student_backbone, student_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler, fishers):
    beta=2
    best_acc = 0
    metrics_history = []


    confidence_gate_prev = 0.4 * float(torch.log(torch.tensor(args.nb_cl)).item())

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


    for epoch in range(args.epochs):

        epoch_ent_sum = 0.0
        epoch_ent_n = 0
        epoch_num_classes = None
        confidence_gate = float(confidence_gate_prev)

        reg_alpha = 1.0
        # Set the model to the training mode
        teacher_backbone.train()
        teacher_classifier.train()
        student_backbone.train()
        student_classifier.train()

        train_loss = 0
        consistency_loss_sum = 0
        mutual_info_loss_sum = 0

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
            if args.CKCR:
                with torch.no_grad():
                    score = teacher_classifier(teacher_backbone(inputs))
                if args.select_soft_knowledge:
                    knowledge, knowledge_mask = distill_knowledge_by_entropy(score, confidence_gate, temperature=2)
                else:
                    knowledge, knowledge_mask = distill_knowledge_by_entropy(score,  float(torch.log(torch.tensor(args.nb_cl)).item()), temperature=2)

                if args.mixup:
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
                    mixed_data = inputs
                    mixed_consensus = knowledge
            else:
                with torch.no_grad():
                    score = teacher_classifier(teacher_backbone(inputs))
                    knowledge, knowledge_mask = distill_knowledge_by_entropy(score, float(torch.log(torch.tensor(args.nb_cl)).item()), temperature=1)
                mixed_data = inputs
                mixed_consensus = knowledge

            prob_t = torch.softmax(score, dim=1)
            ent = -torch.sum(prob_t * torch.log(prob_t + 1e-12), dim=1)  # [B]
            epoch_ent_sum += ent.sum().item()
            epoch_ent_n += ent.numel()
            epoch_num_classes = score.size(1)

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

            # update_fisher_from_grads(student_backbone, fishers, rho=fisher_rho)

            train_loss += loss.item()
            consistency_loss_sum += float(consistency_loss.item())
            mutual_info_loss_sum += float(mutual_info_loss.item())
 
            backbone_optimizer.step()
            classifier_optimizer.step()


        # ===== compute epoch-conf =====
        if (epoch_ent_n > 0) and (epoch_num_classes is not None):
            bar_H = epoch_ent_sum / epoch_ent_n
            C = float(epoch_num_classes)
            H_max = float(torch.log(torch.tensor(C)).item())  # log(C)
            # print("Epoch average entropy: {:.4f}, H_max: {:.4f}".format(bar_H, H_max))
            u = bar_H / max(1e-8, H_max)
            u = max(0.0, min(1.0, u))
            conf = 1.0 - u
        else:
            conf = 0.0  # fallback: very uncertain

        if args.EPHS:
            gate_target = u * H_max
            confidence_gate_prev = gate_target

            # ===== rst from same conf =====
            rst_min = getattr(args, "rst_min", 0.001)
            rst_max = getattr(args, "rst_max", 0.01)
            rst_target = rst_min + (rst_max - rst_min) * conf
            rst_ema_rho = getattr(args, "rst_ema_rho", 0.9)
            rst = rst_target
            rst_prev = float(rst)

            # ===== tao from same conf =====
            tao_min = getattr(args, "tao_begin", 0.95)
            tao_max = getattr(args, "tao_end", 0.99)
            tao_target = tao_min + (tao_max - tao_min) * conf
            tao_target = float(max(0.0, min(0.9999, tao_target)))
            tao_ema_rho = getattr(args, "tao_ema_rho", 0.9)
            tao = tao_target
            tao_prev = float(tao)

            print(f"Epoch {epoch}: conf={conf:.4f}, gate(now)={confidence_gate:.3f}, gate(next)={confidence_gate_prev:.3f}, rst={rst:.4f}, tao={tao:.4f}")
        else:
            confidence_gate_prev = confidence_gate
            rst = 0.01
            tao = 0.95

        new_bn_statistics = get_bn_statistics(student_backbone.state_dict())
        
        # EMA updates for Mean-Teacher framework
        if args.EMA:
            bn_statistics_moving_average(old_bn_statistics, new_bn_statistics, epoch, args.epochs,tao_begin=tao, tao_end=tao) 
            exponential_moving_average(teacher_backbone, student_backbone, epoch, args.epochs,tao_begin=tao, tao_end=tao)
            exponential_moving_average(teacher_classifier, student_classifier, epoch, args.epochs,tao_begin=tao, tao_end=tao)
        else:
            # Without EMA: directly copy student to teacher (no momentum)
            teacher_backbone.load_state_dict(student_backbone.state_dict())
            teacher_classifier.load_state_dict(student_classifier.state_dict())
        if args.FISR:
            fisher_weighted_sr(teacher_backbone, old_state_backbone, fishers, rst=rst)
        elif args.SR:
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


        if args.Reset_student:
            if abs(confidence_gate_prev - confidence_gate) < 1e-3 and confidence_gate <= 0.4 * float(torch.log(torch.tensor(args.nb_cl)).item()):
                print("Reset student model.")
                student_backbone.load_state_dict(teacher_backbone.state_dict())
                student_classifier.load_state_dict(teacher_classifier.state_dict())
            else:
                pass


        # Learning rate decay
        backbone_scheduler.step()
        classifier_scheduler.step()
        avg_train_loss = train_loss / (batch_idx + 1)
        train_accuracy = 100. * correct / total
        avg_consistency_loss = consistency_loss_sum / (batch_idx + 1)
        avg_mutual_info_loss = mutual_info_loss_sum / (batch_idx + 1)
        
        # Print the training losses and accuracies
        print(backbone_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  accuracy: {:.4f} '.format(
            len(train_loader), avg_train_loss,  train_accuracy))

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

        avg_test_loss = test_loss / (batch_idx + 1)
        test_accuracy = 100. * correct / total
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), avg_test_loss, test_accuracy))

        metrics_history.append({
            "stage_type": "target_adapt",
            "session": args.session,
            "domain": int(args.Domain_Seq[args.session]),
            "epoch": epoch,
            "backbone_learning_rate": backbone_scheduler.get_last_lr()[0],
            "classifier_learning_rate": classifier_scheduler.get_last_lr()[0],
            "confidence_gate": confidence_gate,
            "confidence_gate_next": confidence_gate_prev,
            "conf": conf,
            "rst": rst,
            "tao": tao,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "train_consistency_loss": avg_consistency_loss,
            "train_mutual_info_loss": avg_mutual_info_loss,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
        })

        # Save the best model
        if test_accuracy >= best_acc:
            best_acc = test_accuracy
            best_backbone = copy.deepcopy(teacher_backbone)
            best_classifier = copy.deepcopy(teacher_classifier)
            confidence_gate_best = confidence_gate_prev

        fishers= Fisher_Entropy(best_backbone, best_classifier, confidence_gate_best, train_loader)

    metrics_csv_path = _build_metrics_csv_path(args)
    _save_metrics_to_csv(metrics_csv_path, metrics_history)
    print(f"Saved stage metrics to: {metrics_csv_path}")

    return best_backbone, best_classifier, fishers
