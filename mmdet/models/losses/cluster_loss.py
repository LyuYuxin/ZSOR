from addict.addict import Dict
import torch
from torch._C import _init_names
import torch.nn as nn
import random
from collections import deque
import numpy as np
from mmdet.models.utils import Store
from mmdet.models.utils.events import get_event_storage

class Clustering(nn.Module):
    '''
        关于聚类的方法，维护bank，计算聚类损失
    '''
    def __init__(self, 
    feature_store:Store, 
    feats_vector, 
    gt_targets, 
    num_classes, 
    clustering_start_iter, 
    clustering_update_mu_iter, 
    clustering_momentum, 
    clustering_margin, 
    device
    ):
        super(Clustering, self).__init__()
        # 存放每个类的特征向量的bank
        self.feature_store = feature_store
        # bbox head的特征向量，应当代表当前batch所有与gt对应的正样本
        self.feats_vector = feats_vector
        # feat_vectors的标签，val从0-97， 99
        self.gt_targets = gt_targets
        # 输入bank的已知类的类别数
        self.num_classes = num_classes
        # 开始计算类中心的迭代次数，原论文为1000
        self.clustering_start_iter = clustering_start_iter
        # 更新类中心得迭代次数， 原论文为3000
        self.clustering_update_mu_iter = clustering_update_mu_iter
        # 类中心的更新动量，原论文为0.99
        self.clustering_momentum = clustering_momentum
        # 聚类损失的边界
        self.clustering_margin = clustering_margin

        # 在类中新建的tensor指定到的设备
        self.device = device

        self.prototypes = [None for _ in range(num_classes + 1)] # 类中心的一个张量列表 加上未知类

    def get_feats_prototype(self):
        # 得到类中心向量的list of tensor
        storage = get_event_storage() # 便于取当前迭代次数

        # 如果当前迭代次数等于起始计算类中心向量的迭代次数
        if storage.iter == self.clustering_start_iter: 
            items = self.feature_store.retrieve(-1) # list of ech class feature vector
            for idx, item in items:
                if len(item) == 0:
                    self.prototypes[idx] = None
                else:
                    self.prototypes[idx] = torch.tensor(item, device = self.device).mean(dim=0)

        # 如果当前迭代次数大于起始并且需要更新类中心
        elif storage.iter > self.clustering_start_iter and storage.iter % self.clustering_update_mu_iter == 0:
            items = self.feature_store.retrieve(-1) # list of ech class feature vector
            new_prototypes = [None for _ in range(self.num_classes)]
            for idx, item in items:
                if len(item) == 0:
                    new_prototypes[idx] = None
                else:
                    new_prototypes[idx] = torch.tensor(item, device = self.device).mean(dim=0)
            for i, mean_vector in enumerate(self.prototypes):
                # 更新类中心向量
                    if (mean_vector) is not None and new_prototypes[i] is not None:
                        self.prototypes[i] = self.clustering_momentum * mean_vector + (1 - self.clustering_momentum) * new_prototypes[i]
        return self.prototypes
            
    def update_feature_bank(self, vector_feats, gt_labels):
        '''
        gt_labels: 每个sample对应gt的标签。0-97为正样本，98为背景，99为未知类
        '''
        # 判断向量与标签是否匹配
        assert len(vector_feats) == len(gt_labels)

        # 先处理已知类的向量，放入bank中
        pos_idx = torch.nonzero(gt_labels < 98).squeeze()
        pos_vectors = vector_feats[pos_idx]
        pos_targets = gt_labels[pos_idx]
        self.feature_store.add(pos_vectors, pos_targets)

        # 处理未知类的向量
        unknown_idx = torch.nonzero(gt_labels == 99).squeeze()
        unknown_vectors = vector_feats[unknown_idx]
        unknown_targets = torch.full(unknown_idx.shape, fill_value=98, device = self.device)# 适配feats bank 下标
        self.feature_store.add(unknown_vectors, unknown_targets.detach())

    def cal_clustering_loss(self, input_vectors, input_labels, loss_first:bool):
        # 计算最后的聚类loss
        c_loss = 0
        # 先计算loss，再更新bank
        if loss_first:
            c_loss = self.clstr_distance_loss(input_vectors, input_labels)
            self.update_feature_bank(input_vectors, input_labels)
        # 先更新bank再计算loss
        elif loss_first is False:
            self.update_feature_bank(input_vectors, input_labels)
            c_loss = self.clstr_distance_loss(input_vectors, input_labels)
        # # 再总的lossdict上加
        # if 'loss_clustering' in loss_dict.keys():
        #     loss_dict['loss_clustering'] += c_loss
        # else:
        #     loss_dict['loss_clustering'] = c_loss
        return c_loss

    def clstr_distance_loss(self, input_vectors, input_labels, margin = 2):
        all_mean_vectors = self.get_feats_prototype() #取类中心list
        mask = input_labels != self.num_classes # 滤除背景标签
        # 得到已知类和未知类的特征向量和标签
        object_feats_vectors = input_vectors[mask]
        object_feats_labels = input_labels[mask]
        # 将未知类标签-1方便与特征中心的下标进行比对
        object_feats_labels = [label - 1 for label in object_feats_labels if label == 99]

        # 将某个类中心没有的放入0
        for item in all_mean_vectors:
            if item != None:
                length = item.shape
                break

        for i, item in enumerate(all_mean_vectors):
            if item == None:
                all_mean_vectors[i] = torch.zeros((length), device = self.device)

        # 得到滤除背景后的特征向量与类中心的距离
        distances = torch.cdist(object_feats_vectors, torch.stack(all_mean_vectors).cuda(), p = margin)
        labels = []
        # 对于输入的特征向量需要根据其类别标签与类中心下标是否对应来构建损失函数计算的目标标签
        for index, feature in enumerate(object_feats_vectors):
            for cls_index, mu in enumerate(all_mean_vectors):
                if mu is not None and feature is not None:
                    if  object_feats_labels[index] ==  cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)
        # 这里num_classes是已知类的个数
        loss = nn.HingeEmbeddingLoss(self.clustering_margin)(distances, torch.tensor(labels).reshape((-1, self.num_classes + 1)).cuda())
        return loss

# if __name__ == "__main__":
#     get_event_storage()