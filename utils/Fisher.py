import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torch.optim as optim

def Fisher(backbone, classifier, loader):

    backbone.train()
    classifier.eval()

    params = filter(lambda p: p.requires_grad, backbone.parameters())
    EWC_optimizer = optim.SGD(params, lr=0.001)
    fishers = {}
    for batch_idx, instance in enumerate(loader):
        inputs, labels = instance[0].cuda(), instance[1].cuda()
        features = backbone(inputs)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        loss = nn.CrossEntropyLoss()(outputs, predicted)
        loss.backward()
        for name, param in backbone.named_parameters():
            if param.grad is not None:
                if batch_idx > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if batch_idx == len(loader):
                    fisher = fisher / batch_idx
                fishers.update({name: [fisher, param.data.clone().detach()]})
        EWC_optimizer.zero_grad()

    backbone.eval()
    return fishers

# def Fisher_Entropy(backbone, classifier, confident_gate, loader):

#     backbone.train()
#     classifier.eval()

#     params = filter(lambda p: p.requires_grad, backbone.parameters())
#     EWC_optimizer = optim.SGD(params, lr=0.001)
#     fishers = {}
#     for batch_idx, instance in enumerate(loader):
#         inputs, labels = instance[0].cuda(), instance[1].cuda()
#         features = backbone(inputs)
#         outputs = classifier(features)
#         _, predicted = outputs.max(1)
#         loss = nn.CrossEntropyLoss()(outputs, predicted)
#         loss.backward()
#         for name, param in backbone.named_parameters():
#             if param.grad is not None:
#                 if batch_idx > 1:
#                     fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
#                 else:
#                     fisher = param.grad.data.clone().detach() ** 2
#                 if batch_idx == len(loader):
#                     fisher = fisher / batch_idx
#                 fishers.update({name: [fisher, param.data.clone().detach()]})
#         EWC_optimizer.zero_grad()

#     backbone.eval()
#     return fishers

def Fisher_Entropy(backbone, classifier, confident_gate, loader, eps=1e-12):

    backbone.train()
    classifier.eval()

    params = filter(lambda p: p.requires_grad, backbone.parameters())
    EWC_optimizer = optim.SGD(params, lr=0.001)

    fishers = {}
    used_batches = 0  # 统计真正参与累积的 batch（mask 非空）

    for batch_idx, instance in enumerate(loader):
        inputs, labels = instance[0].cuda(), instance[1].cuda()

        features = backbone(inputs)
        outputs = classifier(features)

        # ===== entropy filter (keep only ent < confident_gate) =====
        prob = torch.softmax(outputs, dim=1)
        ent = -torch.sum(prob * torch.log(prob + eps), dim=1)  # [B]
        mask = (ent < confident_gate)

        if mask.sum().item() == 0:
            EWC_optimizer.zero_grad()
            continue

        outputs_sel = outputs[mask]
        _, predicted = outputs_sel.max(1)

        # pseudo-label CE on selected samples
        loss = nn.CrossEntropyLoss()(outputs_sel, predicted)
        loss.backward()

        for name, param in backbone.named_parameters():
            if param.grad is not None:
                g2 = param.grad.data.clone().detach() ** 2
                if name in fishers:
                    fisher = fishers[name][0] + g2
                else:
                    fisher = g2
                # 仍保持你的存储格式：[fisher, param_snapshot]
                fishers.update({name: [fisher, param.data.clone().detach()]})

        used_batches += 1
        EWC_optimizer.zero_grad()

    # ===== normalize (fix: batch_idx == len(loader) 永远不会触发) =====
    if used_batches > 0:
        for name in fishers.keys():
            fishers[name][0] = fishers[name][0] / used_batches

    backbone.eval()
    return fishers



def Fisher_BN(backbone, classifier, loader):

    backbone.train()
    classifier.eval()

    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
        else:
            m.requires_grad_(False)

    params = filter(lambda p: p.requires_grad, backbone.parameters())
    EWC_optimizer = optim.SGD(params, lr=0.001)
    fishers = {}
    for batch_idx, instance in enumerate(loader):
        inputs, labels = instance[0].cuda(), instance[1].cuda()
        features = backbone(inputs)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        loss = nn.CrossEntropyLoss()(outputs, predicted)
        loss.backward()
        for name, param in backbone.named_parameters():
            if param.grad is not None:
                if batch_idx > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if batch_idx == len(loader):
                    fisher = fisher / batch_idx
                fishers.update({name: [fisher, param.data.clone().detach()]})
        EWC_optimizer.zero_grad()

    backbone.eval()
    return fishers