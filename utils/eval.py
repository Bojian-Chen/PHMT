import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix 


def set_exemplar(args, feature, prediction_label):
    
    feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
    exemplar_index = []
    for i in range(args.nb_cl):
        cl_index = np.where(prediction_label == i)[0]
        if len(cl_index) < args.nb_exemplar:
            continue
        feature_i = feature[cl_index]
        feature_avg = np.mean(feature_i, axis=0)
        D = np.dot(feature_i, feature_avg)
        ind_max = np.argsort(D)[:args.nb_exemplar]
        ind_max = cl_index[ind_max]
        exemplar_index.append(ind_max)
    exemplar_index = np.hstack(np.array(exemplar_index))
    return exemplar_index


def evaluate(args, backbone, classifier, evalloader, k, session):
    feature_shape = {'cnn': 640, 'resnet14': 64, 'resnet32': 64, 'resnet18_1D': 256}
    feature_shape = feature_shape[args.backbone_name]
    num_samples = len(evalloader.dataset)
    feature_bank = torch.zeros(num_samples, feature_shape, dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.eval()
    classifier.eval()

    test_loss = 0
    correct = 0
    total = 0
    prediction_label = []
    Targets = []

    with torch.no_grad():
        for batch_idx, instance in enumerate(evalloader):
            inputs, labels = instance[0].to(device), instance[1].to(device)
            idx = instance[2]
            if args.incremental_mode == 'gsfda' or args.incremental_mode == 'UCSN':
                if session == 0:
                    features, masks = backbone(inputs, t=0, s=100, all_mask=False)
                else:
                    features, masks = backbone(inputs, t=1, s=100, all_mask=False)
            else:
                features = backbone(inputs)
            outputs = classifier(features)

            feature_bank[idx] = features.detach().clone().cpu()

            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            prediction_label.append(predicted.cpu())
            Targets.append(labels.cpu())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    prediction_label = torch.Tensor( np.array( [item.numpy() for item in prediction_label])).reshape(-1)
    Targets = torch.Tensor( np.array( [item.numpy() for item in Targets])).reshape(-1)

    cm = confusion_matrix(Targets, prediction_label)
    print(cm)
    print('eval domain: {} eval set: {} test loss: {:.4f} accuracy: {:.4f}'.format(args.Domain_Seq[k], len(evalloader), test_loss/(batch_idx+1), 100.*correct/total))
    # replay_dataset = None
    # if k == session and args.nb_exemplar > 0:
    #     exemplar_index = set_exemplar(args, feature_bank, prediction_label)
    #     replay_data = evalloader.dataset.data[exemplar_index]
    #     replay_label = evalloader.dataset.targets[exemplar_index]
    #     replay_dataset = torch.utils.data.TensorDataset(replay_data, replay_label)
        # replay_loader = torch.utils.data.DataLoader(replay_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return 100.*correct/total, feature_bank




def evaluate_during_train(args, backbone, classifier, k):
    evalloader = args.testloader_list[k]
    print('evaluate during training')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.eval()
    classifier.eval()

    test_loss = 0
    correct = 0
    total = 0
    prediction_label = []
    Targets = []

    with torch.no_grad():
        for batch_idx, instance in enumerate(evalloader):
            inputs, labels = instance[0].to(device), instance[1].to(device)
            idx = instance[2]

            features = backbone(inputs)
            outputs = classifier(features)

            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            prediction_label.append(predicted.cpu())
            Targets.append(labels.cpu())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    prediction_label = torch.Tensor( np.array( [item.numpy() for item in prediction_label])).reshape(-1)
    Targets = torch.Tensor( np.array( [item.numpy() for item in Targets])).reshape(-1)

    print('eval domain: {} eval set: {} test loss: {:.4f} accuracy: {:.4f}'.format(args.Domain_Seq[k], len(evalloader), test_loss/(batch_idx+1), 100.*correct/total))
    return 100.*correct/total
