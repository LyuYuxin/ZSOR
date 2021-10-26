import random
from collections import deque
import numpy as np
import torch
class Store:
    def __init__(self, total_num_classes, items_per_class, momentum = 0.9, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]
        self.momentum = momentum# 更新动量

        for i in range(self.total_num_classes):
            self.store[i].append(torch.randn((1024,), device='cuda', requires_grad=False) )

        self.prototypes = torch.empty((self.total_num_classes, 1024), device='cuda')
        
        for idx, feats in enumerate(self.store):
            self.prototypes[idx] =  sum(feats) / len(self.store[idx])# 计算已知类 和未知类

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def get_feats_prototype(self, update=False):
        if update:
            new_prototypes = torch.empty((self.total_num_classes, 1024), device='cuda')
            for idx, feats in enumerate(self.store):
                new_prototypes[idx] =  sum(feats) / len(self.store[idx])   # tag 
            
            self.prototypes = self.prototypes * self.momentum + new_prototypes * (1 - self.momentum)

        #return feats mean list
        return self.prototypes# cuda tensor list

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])


    def update_feats_bank(self, gt_targets):
        '''
        gt_targets: 每个sample对应gt的标签。0-97为正样本，98为背景，99为未知类
        '''
        assert len(gt_targets) == len(self.feat_vectors)
        pos_idx = torch.nonzero(gt_targets < 98).squeeze()
        pos_vectors = self.feat_vectors[pos_idx]
        pos_targets = gt_targets[pos_idx]
        self.feats_bank.add(pos_vectors, pos_targets)

        unknown_idx = torch.nonzero(gt_targets == 99).squeeze()
        unknown_vectors = self.feat_vectors[unknown_idx]
        unknown_targets = torch.full(unknown_idx.shape, fill_value=98, device='cuda')# 适配feats bank 下标
        self.feats_bank.add(unknown_vectors, unknown_targets.detach())



if __name__ == "__main__":
    store = Store(10, 3)
    store.add(('a', 'b', 'c', 'd', 'e', 'f'), (1, 1, 9, 1, 0, 1))
    store.add(('h',), (4,))
    # print(store.retrieve(1))
    # print(store.retrieve(3))
    # print(store.retrieve(9))
    print(store.retrieve(-1))
    # print(len(store))
    # store.reset()
    # print(len(store))

    print(store)