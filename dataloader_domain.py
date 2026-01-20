import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import scipy.io as sio #读取matlab文件
from scipy.fftpack import fft
import argparse

class dataloader():

    def __init__(self, args, session):
 
        dataset=sio.loadmat(args.dataroot+args.train_list)
        self.data = dataset['data']
        if args.data_mode == 'Frequence':
            self.data = torch.FloatTensor(abs(fft(self.data)))
        if args.data_mode == 'Time':
            self.data = torch.FloatTensor((self.data))

        self.targets = torch.LongTensor(dataset['label']) 
        self.targets = self.targets.reshape(-1)
     
        self.data, self.targets = self.SelectfromSession(args, session)
        

        
        if args.preprocess == 'zscore' :
            mean = torch.mean(self.data, dim=0, keepdim=True) 
            std = torch.std(self.data, dim=0, keepdim=True)
            self.data = (self.data - mean) / std

        if args.preprocess == 'minmax':
            mean = torch.mean(self.data, dim=0, keepdim=True) 
            std = torch.std(self.data, dim=0, keepdim=True)
            self.data = (self.data - mean) / std


        if args.preprocess == 'None':
            pass

        if args.data_dimension == '2D':
            self.data = torch.reshape(self.data, [-1, 1, 32, 32])

        if args.data_dimension == '1D':
            self.data = torch.reshape(self.data, [-1, 1, 1024])

    def SelectfromSession(self, args, session):
        Domain_ID = args.Domain_Seq[session]
        max_label = self.targets.max()+1
        data_num_class = args.data_num_class
        ind = np.arange(Domain_ID*max_label*data_num_class,(Domain_ID+1)*max_label*data_num_class)

        data = self.data[ind]
        targets = self.targets[ind]
        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target, index

    def __len__(self):
        return len(self.data)




if __name__ == "__main__":

    Domain_Seq = np.arange(8)
    nb_session = len(Domain_Seq)

    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--random_seed', default=2023, type=int, help='random seed')
    parser.add_argument('--backbone_name', default='resnet32', type=str, choices=['resnet32', 'resnet18_1D','cnn'], help='the backbone name')
    parser.add_argument('--batch_size', default=100, type=int, help='the batch size for data loader')
    parser.add_argument('--base_epochs', default=40, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
    parser.add_argument('--nb_cl', default=5, type=int, help='the number of classes')
    parser.add_argument('--base_lr', default=0.1, type=float, help='the learning rate for base train')
    parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

    ### Dataset parameters
    parser.add_argument('--data_dimension', default='2D', choices=['1D', '2D'], type=str, help='the dimension of data')
    parser.add_argument('--data_mode', default='Frequence', type=str, choices=['Frequence', 'Time'], help='the mode of data')
    parser.add_argument('--dataroot', default='./data/', type=str, help='the path to load the data')
    parser.add_argument('--train_list', default='./WT_all_5classes.mat', type=str, help='the name of the source dir')
    parser.add_argument('--test_list', default='./WT_all_5classes.mat', type=str, help='the name of the test dir')
    parser.add_argument('--preprocess', default= 'zscore', type=str, choices=['zscore', 'minmax', 'None'], help='the preprocess setting')
    ### Loader parameters
    parser.add_argument('--Domain_Seq', default=Domain_Seq , type=int, help='the Domain_Seq setting')
    parser.add_argument('--nb_session', default=nb_session, type=int, help='the number of sessions')
    parser.add_argument('--data_num_class', default=100, type=int, help='the number of data in each class')

    args = parser.parse_args()

    src_trainset = dataloader(args,session=5)
    src_trainloader = torch.utils.data.DataLoader(dataset=src_trainset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)

    # explamars_set = dataloader(args,session=1, nb_exemplar = 5, index_exemplar = [ 97,6 , 66,  55 , 93, 182, 108], nb_shot = 0,train= True, few_shot=False, random_exemplar=False)
    # explamars_set= dataloader(args, session=1, nb_exemplar = args.nb_exemplar, index_exemplar = args.index_exemplar, nb_shot = 0,train= True, few_shot=False, random_exemplar=args.random_exemplar)
    # data = explamars_set.data
    # targets = explamars_set.targets
    # print(data.shape)
    # print(targets)
    # combined_clusteroader = DataLoader(ConcatDataset([src_trainloader.dataset ,explamars_set]), batch_size=args.batch_size, shuffle=True, drop_last=False)
    for batch, (input,label,index) in enumerate(src_trainloader) :
        print(input)
        print(label)
        print(index)
    inx = [0,100,200,202]
    specified_data = src_trainloader.dataset.data[inx]
    specified_labels = src_trainloader.dataset.targets[inx]
    # specified_inx = src_trainloader.dataset
    print(specified_data)
    print(specified_labels)
    # print(specified_inx)
        # print(input[0].shape)
        # print(input[1].shape)
        # print(input[2].shape)
        # print(input[2])



    # nb_exemplar = 5
#     nb_shot = 10.21.3233333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#     # txt_path = "../../data/index_list/cifar100/session_2.txt"
#     # class_index = open(txt_path).read().splitlines()
#     # class_index = np.arange(3,6)
    # trainset = dataloader(root=dataroot, data_list = train_list, session=1,Domain_Seq = Domain_Seq, nb_exemplar = nb_exemplar,nb_shot = nb_shot, train= True, few_shot=True)
#     src_trainset = dataloader(root=dataroot, data_list = train_list, session=0,Domain_Seq = Domain_Seq, nb_exemplar = 0,
#                               index_exemplar = None, nb_shot = 0,train= True, few_shot=None,random_exemplar=False)
#     trainset = dataloader(root=dataroot, data_list = train_list, session=1,Domain_Seq = Domain_Seq, nb_exemplar = 5,
#                               index_exemplar = None, nb_shot = 0,train= True, few_shot=None,random_exemplar=True)
# #     # testset = dataloader(root=dataroot, data_list = test_list, index=class_index)

#     trainloader = torch.utils.data.DataLoader(dataset=src_trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
#                                               pin_memory=True)
#     targetloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                            #   pin_memory=True)
    # for batch_idx,((source_data, source_label),(target_data, target_label)) in enumerate(zip(targetloader, trainloader)): 
    #     print(batch_idx)
    #     print(len(source_data))
    #     print(len(target_data))
    #         # source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)
    # for batch_idx,(source_data, source_label) in enumerate(targetloader): 
    #     print(batch_idx)
    #     print(source_data.shape)
        # print(len(target_data))
            
            
            
    # a = trainset.data[:50]
    # b = trainset.targets[:50]
    # c= np.concatenate((src_trainset.data,a))
    # d = np.concatenate((src_trainset.targets,b))
    # src_trainset.data = c
    # src_trainset.targets = d
    
    # print(src_trainset.targets[1000:1050])
#     # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
    
    # a = torch.zeros([10, batch_size_base])
    # for batch_index,(inputs, targets, index) in enumerate(trainloader):
        
    #     # print(targets)
    #     b = torch.where(targets == 1, index, 9999)
    #     a[batch_index,:] = b
        # a.append(b)
    # print(a)
    # c = torch.where(a < 9999)
    # print(b)
        # print([i for i in targets where(i ==1)])
        # print(index)
        
#     # print(trainset[1][1])
#     # print(len(trainset[0]))
#     # print(trainloader)
#     # testloader = torch.utils.data.DataLoader(
#     #     dataset=testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
    # print(tuple(trainloader.dataset.type))
    # print(trainloader.dataset.targets.shape)
#     print(trainloader.dataset.targets[0:160])
#     # a = [i for i in range(0,10) if i in trainloader.dataset.targets]
#     # print(a)
#     # print(testloader.dataset.data)
#     # print(testloader.dataset.targets)
#     # print(testloader.dataset.type)
#     # print(type(testloader.dataset.data))

