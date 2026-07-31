import torch
import copy
from utils.eval import *
from utils.set import set_model,set_optimizer,set_dataset
from utils.Fisher import Fisher, Fisher_BN

import os
from dataloader_domain import dataloader as dataloader
from trainer.source_train import source_train
from trainer.shot import shot
from trainer.norm import norm
from trainer.PesudoLabel import PesudoLabel
from trainer.tent import tent
from trainer.gsfda import gsfda_source_train, gsfda
from trainer.UCSN import UCSN_source_train, UCSN
from trainer.CoTTA import CoTTA
from trainer.CoSDA import CoSDA
from trainer.ours import ours
from trainer.ours_sc import ours_sc
from trainer.RaTP import RaTP
from trainer.EATA import EATA
from trainer.AFSFFD import AFSFFD
from trainer.ours_simple import ours_simple
from trainer.rmt import rmt
from trainer.ours_new import ours_new

def train(args):
    Correct = []

    backbone, classifier = set_model(args)
    trainloader_list, testloader_list = set_dataset(args)
    # args.testloader_list = testloader_list


    # pth = './log/' + args.dataset_name + '/' + args.Sensitivity_analysis + '/' + args.preprocess + '/' \
    # pth = './log/' + args.dataset_name + '/' + args.ablation_mode + '/' + args.preprocess + '/' \
    # pth = './log/' + args.dataset_name + '/' + args.incremental_mode + '/' + args.preprocess + '/' \

    pth = './log/' + args.dataset_name + '/' + args.incremental_mode + '/' + args.preprocess + '/' \
         + str(os.path.splitext(os.path.basename(args.train_list))[0]) + \
        '_' + args.backbone_name + \
        '_' + str(args.contrastive_loss) + \
        '_' + args.classifer + \
        '_' + args.train_parames + \
        '_' + args.data_dimension + \
        '_' + args.data_mode + \
        '_' + str(args.Domain_Seq) + \
        '_' + str(args.random_seed)
    args.pth = pth
    # args.pth = args.pth + str(args.value)
    print(args.pth)

    if args.save_model:
        if not os.path.exists(args.pth):
            os.makedirs(args.pth)

    for session in range(args.nb_session):
        args.session = session
        print('session: {}'.format(session))
        train_loader = trainloader_list[session]
        test_loader = testloader_list[session]

        
        backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler = set_optimizer(args, backbone, classifier, session)

        ### source-pretraining
        if session == 0:
            if args.incremental_mode == 'gsfda':
                backbone,classifier = gsfda_source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
            elif args.incremental_mode == 'UCSN':
                backbone,classifier = UCSN_source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
                # backbone,classifier=train_source(args,train_loader)
            else:
                backbone,classifier = source_train(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
            source_backbone = copy.deepcopy(backbone)
            source_classifier = copy.deepcopy(classifier)
            if args.incremental_mode == 'EATA' :
                fishers = Fisher_BN(source_backbone, source_classifier, train_loader)
            if args.incremental_mode == 'AFSFFD' or args.incremental_mode == 'ours_new':
                fishers = Fisher(source_backbone, source_classifier, train_loader)
            # features, prototypes = compute_current_session_embedding(args, model_old, trainset)
        else:

            if args.incremental_mode == 'Source':
                backbone, classifier = source_backbone, source_classifier
            if args.incremental_mode == 'norm':
                backbone, classifier = norm(args, backbone, classifier, train_loader, test_loader)

            elif args.incremental_mode == 'PesudoLabel':
                backbone, classifier = PesudoLabel(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler)

            elif args.incremental_mode == 'Tent':
                backbone, classifier = tent(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler)

            elif args.incremental_mode == 'EATA':
                backbone, classifier = EATA(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler, fishers)

            elif args.incremental_mode == 'AFSFFD':
                backbone, classifier, fishers = AFSFFD(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler, fishers)

            elif args.incremental_mode == 'shot':
                backbone, classifier = shot(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, backbone_scheduler)

            elif args.incremental_mode == 'gsfda':
                backbone, classifier = gsfda(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)

            elif args.incremental_mode == 'UCSN':
                backbone, classifier = UCSN(args, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)

            elif args.incremental_mode == 'CoTTA':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = CoTTA(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)

            elif args.incremental_mode == 'CoSDA':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = CoSDA(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
        
            elif args.incremental_mode == 'ours':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = ours(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
           
            elif args.incremental_mode == 'ours_sc':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = ours_sc(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
           
            elif args.incremental_mode == 'ours_simple':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = ours_simple(args, teacher_backbone, teacher_classifier, backbone, classifier, source_backbone, train_loader, test_loader, backbone_optimizer,backbone_scheduler)
            
            elif args.incremental_mode == 'RaTP':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = RaTP(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
           
            elif args.incremental_mode == 'rmt':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier = rmt(args, teacher_backbone, teacher_classifier, backbone, classifier, source_classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler)
        
            elif args.incremental_mode == 'ours_new':
                teacher_backbone = copy.deepcopy(backbone)
                teacher_classifier = copy.deepcopy(classifier)
                backbone, classifier, fishers = ours_new(args, teacher_backbone, teacher_classifier, backbone, classifier, train_loader, test_loader, backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler, fishers)
           
       
        if args.save_model:
            torch.save(backbone, args.pth+'/backbone_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')
            torch.save(classifier, args.pth+'/classifier_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')

        C =[]
        for k in range(args.nb_session):
            evalloader = testloader_list[k]
            correct, feature_bank = evaluate(args, backbone, classifier, evalloader, k, session)
  
                
            # if k == session and args.nb_exemplar > 0 and args.random_exemplar == False:
            #     print('------Herding------')
            #     exemplar_index = set_exemplar(args, session, features, prediction_label)
            #     if args.index_exemplar is None:
            #         args.index_exemplar = exemplar_index
            #     else:
            #         args.index_exemplar = np.concatenate((args.index_exemplar, exemplar_index))

            if args.save_model:
                torch.save(feature_bank,args.pth+'/features_ours_session_'+str(session)+'_domain_'+str(args.Domain_Seq[k])+'.pth')
            
            C.append(correct)
            C = [float('{:.1f}'.format(i)) for i in C]
        Correct.append(C)
        print(Correct)
    return Correct
        
