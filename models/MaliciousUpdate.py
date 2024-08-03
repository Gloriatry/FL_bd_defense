#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
import random
from sklearn import metrics
import copy
import math
from skimage import io
import time
import cv2
from skimage import img_as_ubyte
import heapq
import os
# print(os.getcwd())
from models.Attacker import get_attack_layers_no_acc

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalMaliciousUpdate(object):
    def __init__(self, args, datasethelper=None, idxs=None, attack=None, order=None, total_num_dps_per_round=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            datasethelper.dataset_train, idxs), batch_size=self.args.local_bs, shuffle=True)
        #  change 0708
        self.data = DatasetSplit(datasethelper.dataset_train, idxs)
        self.datasethelper = datasethelper
        self.total_num_dps_per_round = total_num_dps_per_round
        
        # only used in semantic attack
        if self.args.dataset == 'cifar' and self.args.attack == 'semantic':
            self.poisoned_data = DataLoader(datasethelper.dataset_train,
                            batch_size=self.args.local_bs,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(datasethelper.poisoned_data_for_train))

        # backdoor task is changing attack_goal to attack_label
        self.attack_label = args.attack_label
        self.attack_goal = args.attack_goal
        
        self.model = args.model
        self.poison_frac = args.poison_frac  # 数据集中恶意数据的比例
        if attack is None:
            self.attack = args.attack
        else:
            self.attack = attack

        self.trigger = args.trigger
        self.triggerX = args.triggerX
        self.triggerY = args.triggerY
        self.watermark = None
        self.apple = None
        self.dataset = args.dataset
        if self.attack == 'dba':
            self.dba_class = order % 4
        elif self.attack == 'get_weight':
            self.idxs = list(idxs)
            
    def add_trigger(self, image):
        if self.attack == 'dba':
            pixel_max = 1
            if self.dba_class == 0:
                image[:,self.triggerY+0:self.triggerY+2,self.triggerX+0:self.triggerX+2] = pixel_max
            elif self.dba_class == 1:
                image[:,self.triggerY+0:self.triggerY+2,self.triggerX+2:self.triggerX+5] = pixel_max
            elif self.dba_class == 2:
                image[:,self.triggerY+2:self.triggerY+5,self.triggerX+0:self.triggerX+2] = pixel_max
            elif self.dba_class == 3:
                image[:,self.triggerY+2:self.triggerY+5,self.triggerX+2:self.triggerX+5] = pixel_max
            self.save_img(image)
            return image
        if self.trigger == 'square':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            # 2022年6月10日 change
            if self.dataset == 'cifar':
                pixel_max = 1
            image[:,self.triggerY:self.triggerY+5,self.triggerX:self.triggerX+5] = pixel_max
        elif self.trigger == 'pattern':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            image[:,self.triggerY+0,self.triggerX+0] = pixel_max
            image[:,self.triggerY+1,self.triggerX+1] = pixel_max
            image[:,self.triggerY-1,self.triggerX+1] = pixel_max
            image[:,self.triggerY+1,self.triggerX-1] = pixel_max
        elif self.trigger == 'watermark':
            if self.watermark is None:
                self.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                self.watermark = cv2.bitwise_not(self.watermark)
                self.watermark = cv2.resize(self.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(self.watermark)
                self.watermark = self.watermark.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                self.watermark *= pixel_max_dataset
            max_pixel = max(np.max(self.watermark),torch.max(image))
            image += self.watermark
            image[image>max_pixel]=max_pixel
        elif self.trigger == 'apple':
            if self.apple is None:
                self.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                self.apple = cv2.bitwise_not(self.apple)
                self.apple = cv2.resize(self.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(self.apple)
                self.apple = self.apple.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                self.apple *= pixel_max_dataset
            max_pixel = max(np.max(self.apple),torch.max(image))
            image += self.apple
            image[image>max_pixel]=max_pixel
        self.save_img(image)
        return image
    
            
    def trigger_data(self, images, labels):
        #  attack_goal == -1 means attack all label to attack_label
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.attack_goal:  # no in task
                        continue  # jump
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != self.attack_goal:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels
        
    def train(self, net, test_img = None):
        if self.attack == 'badnet':
            return self.train_malicious_badnet(net)
        elif self.attack == 'dba':
            return self.train_malicious_dba(net)
        elif self.attack == 'semantic':
            return self.train_malicious_semantic(net)
        elif self.attack == "edges":
            return self.train_malicious_edges(net)
        else:
            print("Error Attack Method")
            os._exit(0)
            

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net_glob = copy.deepcopy(net)
        model_original = list(net_glob.parameters())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[0.2 * self.args.local_ep, 0.8 * self.args.local_ep], gamma=0.1)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            scheduler.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        
        if self.args.model_replacement:
            v = parameters_to_vector(net.parameters())
            print("Attacker before scaling : Norm = {}".format(torch.norm(v)))
            
            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx])*(max(int(self.args.frac * self.args.num_users), 1)) + model_original[idx]
            v = parameters_to_vector(net.parameters())
            print("Attacker after scaling : Norm = {}".format(torch.norm(v)))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train_malicious_dba(self, net, test_img=None, dataset_test=None, args=None):
        net_glob = copy.deepcopy(net)
        model_original = list(net_glob.parameters())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[0.2 * self.args.local_ep, 0.8 * self.args.local_ep], gamma=0.1)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            scheduler.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        
        if self.args.model_replacement:
            v = parameters_to_vector(net.parameters())
            print("Attacker before scaling : Norm = {}".format(torch.norm(v)))
            
            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx])*(max(int(self.args.frac * self.args.num_users), 1)) + model_original[idx]
            v = parameters_to_vector(net.parameters())
            print("Attacker after scaling : Norm = {}".format(torch.norm(v)))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train_malicious_semantic(self, net, test_img=None, dataset_test=None, args=None):
        net_glob = copy.deepcopy(net)
        model_original = list(net_glob.parameters())
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.poisoned_data):
                # perform semantic attack: insert green-car images and change labels 
                selected_poisoning_images = np.random.choice(self.args.poison_images, self.args.poisoning_per_batch, replace=False)
                for idx, poison_image_idx in enumerate(selected_poisoning_images):
                    images[idx] = self.datasethelper.dataset_train[poison_image_idx][0]
                    images[idx].add_(torch.FloatTensor(images[idx].shape).normal_(0, self.args.noise_level))
                    labels[idx] = self.args.poison_label_swap
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                class_loss = self.loss_func(log_probs, labels)
                # constrain, controlled by self.args.alpha_loss
                distance_loss = torch.norm((parameters_to_vector(list(net.parameters()))-parameters_to_vector(list(net_glob.parameters()))), p=2)
                loss = self.args.alpha_loss * class_loss + (1-self.args.alpha_loss) * distance_loss
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # 与简单平均版本的fedavg对应的scale
        if self.args.model_replacement:
            v = parameters_to_vector(net.parameters())
            print("Attacker before scaling : Norm = {}".format(torch.norm(v)))
            
            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx])*(max(int(self.args.frac * self.args.num_users), 1)) + model_original[idx]
            v = parameters_to_vector(net.parameters())
            print("Attacker after scaling : Norm = {}".format(torch.norm(v)))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
    def train_malicious_edges(self, net, test_img=None, dataset_test=None, args=None):
        net_glob = copy.deepcopy(net)
        model_original = list(net_glob.parameters())
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        adv_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.adv_lr, momentum=self.args.momentum)
        epoch_loss = []
        poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader = self.datasethelper.edges_poisoned_data()
        if self.args.prox_attack:
            print("Prox-attack: Estimating wg_hat")
            wg_clone = copy.deepcopy(net_glob)
            prox_optimizer = torch.optim.SGD(wg_clone.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            for prox_epoch in range(self.args.local_ep):
                wg_clone.train()
                for batch_idx, (images, labels) in enumerate(clean_train_loader):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    prox_optimizer.zero_grad()
                    log_probs = wg_clone(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    prox_optimizer.step()
            wg_hat = wg_clone

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(poisoned_train_loader):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                if self.args.pgd_attack:
                    adv_optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                if self.args.prox_attack:
                    wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
                    model_vec = parameters_to_vector(list(net.parameters()))
                    prox_term = torch.norm(wg_hat_vec - model_vec)**2
                    loss = loss + prox_term
                batch_loss.append(loss.item())

                loss.backward()
                if not self.args.pgd_attack:
                    optimizer.step()
                else:
                    if self.args.proj == "l_inf":
                        w = list(net.parameters())
                        n_layers = len(w)
                        # adversarial learning rate
                        eta = 0.001
                        eps = 5e-4
                        for i in range(len(w)):  # 遍历每一个参数
                            # uncomment below line to restrict proj to some layers
                            if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                                w[i].data = w[i].data - eta * w[i].grad.data  # 梯度下降更新参数
                                # projection step
                                m1 = torch.lt(torch.sub(w[i], net_glob[i]), -eps)
                                m2 = torch.gt(torch.sub(w[i], net_glob[i]), eps)
                                w1 = (net_glob[i] - eps) * m1
                                w2 = (net_glob[i] + eps) * m2
                                w3 = (w[i]) * (~(m1+m2))
                                wf = w1+w2+w3
                                w[i].data = wf.data
                    else:
                        # do l2_projection
                        adv_optimizer.step()
                        w = list(net.parameters())
                        w_vec = parameters_to_vector(w)
                        model_original_vec = parameters_to_vector(list(net_glob.parameters()))
                        # make sure you project on last iteration otherwise, high LR pushes you really far
                        if (batch_idx%self.args.project_frequency == 0 or batch_idx == len(poisoned_train_loader)-1) and (torch.norm(w_vec - model_original_vec) > eps):
                            # project back into norm ball
                            w_proj_vec = eps*(w_vec - model_original_vec)/torch.norm(
                                    w_vec-model_original_vec) + model_original_vec
                            # plug w_proj back into model
                            vector_to_parameters(w_proj_vec, w)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # TODO:这里是按照数据量加权求和的版本，但是这篇的Fedavg是简单平均，要想scale起作用，这里可能需对应
        if self.args.model_replacement:
            v = parameters_to_vector(net.parameters())
            print("Attacker before scaling : Norm = {}".format(torch.norm(v)))
            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx])*((self.total_num_dps_per_round+num_dps_poisoned_dataset)/num_dps_poisoned_dataset) + model_original[idx]
            v = parameters_to_vector(net.parameters())
            print("Attacker after scaling : Norm = {}".format(torch.norm(v)))
    
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    


    def save_img(self, image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
        else:
            img = image.numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            if self.attack == 'dba':
                io.imsave('./save/dba'+str(self.dba_class)+'_trigger.png', img_as_ubyte(img))
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img))
