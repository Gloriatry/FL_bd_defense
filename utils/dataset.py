from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
import numpy as np
import random
import pickle
import copy
import torch
from PIL import Image
import os

class DatasetHelper:
    def __init__(self, args):
        self.args = args

    def load_data(self):
        print("Loading data")

        if self.args.heter_method == "mnist7":
            dict_users = {i: [] for i in range(self.args.num_users)}

            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.dataset_train = datasets.MNIST(
                './data/mnist/', train=True, download=True, transform=trans_mnist)
            self.dataset_test = datasets.MNIST(
                './data/mnist/', train=False, download=True, transform=trans_mnist)
            
            self.train_7_idx = []
            self.train_bar7_idx = []
            self.test_7_idx = []
            self.test_bar7_idx = []

            dir_train_7 = "./data/mnist7/train/7-0"
            for image_name in os.listdir(dir_train_7):
                if image_name.endswith('.png'):
                    self.train_7_idx.append(int(image_name.split('_')[1].split('.')[0]))
            dir_train_bar7 = "./data/mnist7/train/7-1"
            for image_name in os.listdir(dir_train_bar7):
                if image_name.endswith('.png'):
                    self.train_bar7_idx.append(int(image_name.split('_')[1].split('.')[0]))
            dir_test_7 = "./data/mnist7/test/7-0"
            for image_name in os.listdir(dir_test_7):
                if image_name.endswith('.png'):
                    self.test_7_idx.append(int(image_name.split('_')[1].split('.')[0]))
            dir_test_bar7 = "./data/mnist7/test/7-1"
            for image_name in os.listdir(dir_test_bar7):
                if image_name.endswith('.png'):
                    self.test_bar7_idx.append(int(image_name.split('_')[1].split('.')[0]))
            
            self.train_other_idx = list(set([i for i in range(len(self.dataset_train))]) - set(self.train_7_idx) - set(self.train_bar7_idx))
            self.test_other_idx = list(set([i for i in range(len(self.dataset_test))]) - set(self.test_7_idx) - set(self.test_bar7_idx))

            # -7均匀分给前5个客户端，7均匀分给后95个客户端，其余的数字均匀分给100个客户端
            num_items = int(len(self.train_bar7_idx)/5)
            for i in range(5):
                dict_users[i].extend(np.random.choice(self.train_bar7_idx, num_items, replace=False))
                self.train_bar7_idx = list(set(self.train_bar7_idx) - set(dict_users[i]))
            num_items = int(len(self.train_7_idx)/95)
            for i in range(5, 100):
                dict_users[i].extend(np.random.choice(self.train_7_idx, num_items, replace=False))
                self.train_7_idx = list(set(self.train_7_idx) - set(dict_users[i]))
            num_items = int(len(self.train_other_idx)/self.args.num_users)
            for i in range(self.args.num_users):
                dict_users[i].extend(np.random.choice(self.train_other_idx, num_items, replace=False))
                self.train_other_idx = list(set(self.train_other_idx) - set(dict_users[i]))

            return dict_users

        if self.args.dataset == 'mnist':
            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.dataset_train = datasets.MNIST(
                './data/mnist/', train=True, download=True, transform=trans_mnist)
            self.dataset_test = datasets.MNIST(
                './data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            # if self.args.iid:
            #     dict_users = mnist_iid(self.dataset_train, self.args.num_users)
            # else:
            #     dict_users = mnist_noniid(self.dataset_train, self.args.num_users)

            # dict_users = self.split_data(self.dataset_train, self.args.num_users, int(max(self.dataset_test.targets)+1), self.args.q)
        elif self.args.dataset == 'fashion_mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
            self.dataset_train = datasets.FashionMNIST(
                './data/', train=True, download=True, transform=trans_mnist)
            self.dataset_test = datasets.FashionMNIST(
                './data/', train=False, download=True, transform=trans_mnist)
            # sample users
            # if self.args.iid:
            #     dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
            # else:
            #     dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
            # dict_users = self.split_data(self.dataset_train, self.args.num_users, int(max(self.dataset_test.targets)+1), self.args.q)
        elif self.args.dataset == 'cifar':
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])

            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset_train = datasets.CIFAR10(
                './data/cifar', train=True, download=True, transform=transform_train)
            self.dataset_test = datasets.CIFAR10(
                './data/cifar', train=False, download=True, transform=transform_test)
            # if self.args.iid:
            #     dict_users = cifar_iid(self.dataset_train, self.args.num_users)
            # else:
            #     dict_users = cifar_noniid(self.dataset_train, self.args.num_users)
            # dict_users = self.split_data(self.dataset_train, self.args.num_users, int(max(self.dataset_test.targets)+1), self.args.q)
        else:
            exit('Error: unrecognized dataset')
        
        if self.args.heter_method == "normal":
            dict_users = self.split_data_normal(self.dataset_train, self.args.num_users, int(max(self.dataset_test.targets)+1), self.args.alpha)
        else:
            self.classes_dict = self.build_classes_dict()
            dict_users = self.split_data_dirichlet(self.args.num_users, self.args.alpha)

        self.args.num_classes = int(max(self.dataset_test.targets)+1)

        # only used in semantic attack (cifar dataset)
        if self.args.dataset == "cifar" and self.args.attack == 'semantic':
            poisoned_data_num = int(len(self.dataset_train)/self.args.num_users)
            self.poisoned_data_for_train = self.poison_dataset(poisoned_data_num)
        
        # if self.args.attack == 'edges' and self.args.poison_type == 'southwest':
        #     with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
        #         self.saved_southwest_dataset_train = pickle.load(train_f)

        #     with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
        #         self.saved_southwest_dataset_test = pickle.load(test_f)
            
        #     # print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
        #     #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
        #     self.sampled_targets_array_train = 9 * np.ones((self.saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
        #     # print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
        #     #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
        #     self.sampled_targets_array_test = 9 * np.ones((self.saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck

        return dict_users

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.dataset_train):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def split_data_normal(self, dataset, num_clients, num_classes, q):
        dict_users = {i: [] for i in range(num_clients)}
        group_indices = {i: [] for i in range(num_classes)}
        class_indices = {i: [] for i in range(num_classes)}
        clients_per_group = int(num_clients / num_classes) + 1

        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        for label, indices in class_indices.items():
            for idx in indices:
                if np.random.rand() < q:
                    group_indices[label].append(idx)
                else:
                    client_group = np.random.choice([i for i in range(num_classes) if i != label])
                    group_indices[client_group].append(idx)

        # TODO:目前是均匀分的组，即恶意客户端是均匀分布在不同组中        
        for client, client_indices in dict_users.items():
            group = client % num_classes
            data_num = int(len(group_indices[group])/clients_per_group)
            client_indices.extend(list(np.random.choice(group_indices[group], data_num, replace=False)))
            group_indices[group] = list(set(group_indices[group]) - set(client_indices))

        return dict_users

    def split_data_dirichlet(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0])  # for cifar: 5000
        per_participant_list = {i: [] for i in range(no_participants)}
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def poison_dataset(self, poisoned_data_num):
        range_no_id = list(range(50000))
        for image in self.args.poison_images:
            if image in range_no_id:
                range_no_id.remove(image)
        
        poisoned_data_indices = np.random.choice(range_no_id, poisoned_data_num, replace=False)

        return poisoned_data_indices

    def edges_poisoned_data(self):
        if self.args.dataset == "cifar":
            if self.args.poison_type == "southwest":
                with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
        
                # print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
                #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
                sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
                
                # print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
                #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
                sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck

                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
                saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
                sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
                # print("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
                
                poisoned_trainset = copy.deepcopy(self.dataset_train)
                num_sampled_data_points = 400 # M
                samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
                poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
                poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
                # print("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
                # keep a copy of clean data
                clean_trainset = copy.deepcopy(poisoned_trainset)

                poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
                poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

                self.poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=self.args.local_bs, shuffle=True)
                self.clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=self.args.local_bs, shuffle=True)

                poisoned_testset = copy.deepcopy(self.dataset_test)
                poisoned_testset.data = saved_southwest_dataset_test
                poisoned_testset.targets = sampled_targets_array_test

                self.vanilla_test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.bs, shuffle=False)
                self.targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=self.args.bs, shuffle=False)

                self.num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]
        
        return self.poisoned_train_loader, self.vanilla_test_loader, self.targetted_task_test_loader, self.num_dps_poisoned_dataset, self.clean_train_loader