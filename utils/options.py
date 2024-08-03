#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def args_parser():
    parser = argparse.ArgumentParser()
    # save file 
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--init', type=str, default='None',
                        help="location of init model")
    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help="rounds of training")
    parser.add_argument('--resumed_model', type=bool_string, default=True,
                        help="whether to use pretrained model")
    parser.add_argument('--resumed_model_name', type=str, default='./saved_models/cifar_pretrain/model_last.pt.tar.epoch_200',
                        help="path of pretrained model")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients')
    parser.add_argument('--local_ep', type=int, default=6,
                        help="the number of local epochs of malicious client")
    parser.add_argument('--lr', type=float, default=0.05,
                        help="learning rate of malicious client")
    parser.add_argument('--decay', type=float, default=0.005,
                        help="decay rate of malicious client")
    parser.add_argument('--local_ep_b', type=int, default=2,
                        help="the number of local epochs of benign client")
    parser.add_argument('--lr_b', type=float, default=0.1,
                        help="learning rate of benign client")
    parser.add_argument('--decay_b', type=float, default=0.0005,
                        help="decay rate of benign client")
    parser.add_argument('--local_bs', type=int, default=64,  # 50
                        help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    # dataset and heterogenity
    parser.add_argument('--dataset', type=str,
                        default='mnist', help="name of dataset")
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")
    parser.add_argument('--iid', type=int, default=1, help='whether i.i.d or not')  # 目前不用这个参数，普通异构用q来控制就可以了
    # normal, dirichlet, mnist7
    parser.add_argument('--heter_method', type=str, default='normal')
    # Used when heter_method= normal or dirichlet
    # In normal, alpha=0.1 means iid
    # In dirichlet, smaller alpha means more data imblance
    parser.add_argument('--alpha', type=float, default=1,
                        help="different values to shift from iid")

    # attack settings
    #badnet labelflip layerattack updateflip get_weight layerattack_rev layerattack_ER
    parser.add_argument('--attack', type=str,
                        default='badnet', help='attack method')
    parser.add_argument('--malicious',type=float,default=0, help="proportion of mailicious clients")
    parser.add_argument('--poison_frac', type=float, default=0.1, 
                        help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")
    parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label")
    # attack_goal=-1 is all to one
    parser.add_argument('--attack_goal', type=int, default=7,
                        help="trigger to which label")
    # --attack_begin 70 means accuracy is up to 70 then attack
    parser.add_argument('--attack_begin', type=int, default=70,
                        help="the accuracy begin to attack")
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',
                        help="Kind of trigger")  
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='0',
                        help="position of trigger x-aix") 
    parser.add_argument('--triggerY', type=int, default='0',
                        help="position of trigger y-aix")
    #*********semantic attack*********
    parser.add_argument('--poison_epoch', type=int, nargs='+', default=[10000], help='the epoch perform attack')
    # TODO:手动选择green car的图片
    parser.add_argument('--poison_images', type=int, nargs='+', 
                        default=[874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389],
                        help='green car idx for train')
    parser.add_argument('--poison_images_test', type=int, nargs='+', 
                        default=[32941, 36005, 40138],
                        help='green car idx for tests')
    parser.add_argument('--poisoning_per_batch', type=int, default=20, help='how many poisoning images per batch')
    parser.add_argument('--poison_label_swap', type=int, default=2, help='target label')
    parser.add_argument('--noise_level', type=float, default=0.01, help='noise level added to poisoning images')
    parser.add_argument('--alpha_loss', type=float, default=1, help='control the importance of anomaly detection')
    # used in both semantic attack and edges attack
    parser.add_argument('--model_replacement', type=bool_string, default=False, help='to scale or not to scale')
    #*********edges attack*********
    parser.add_argument('--poison_type', type=str, default='southwest')
    parser.add_argument('--pgd_attack', type=bool_string, default=False) 
    parser.add_argument('--proj', type=str, default='l_2', help='the type of projection')
    parser.add_argument('--project_frequency', type=int, default=10, help='project once every how many epochs')
    parser.add_argument('--adv_lr', type=float, default=0.02, help='learning rate for adv in PGD setting')
    parser.add_argument('--prox_attack', type=bool_string, default=False, help='use prox attack')  # constrain

    # model arguments
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
    parser.add_argument('--model', type=str,
                        default='Mnist_CNN', help='model name')
    
    # defence
    parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)
    parser.add_argument('--turn', type=int, default=0)
    # RLR
    parser.add_argument('--robustLR_threshold', type=int, default=4, 
                        help="break ties when votes sum to 0")
    # FLtrust
    parser.add_argument('--server_dataset', type=int,default=200,help="number of dataset in server")
    parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
    # flame
    parser.add_argument('--noise', type=float, default=0.001) 


    args = parser.parse_args()
    return args
