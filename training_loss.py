import pickle as pickle
import os
import pandas as pd
import sklearn
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class F1Loss(nn.Module):
    def __init__(self, classes=30, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


def read_dict_label_to_num():
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    return dict_label_to_num

class CustomTrainer(Trainer):
    """ ????????? ????????? ???????????? ?????????, ?????? ??????????????? trainer??? overiding?????? ?????? """
    def __init__(self, loss_name, original_dataset, train_dataloader, device, *args, **kwargs):
    # def __init__(self, loss_name, original_dataset, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.device = device
        self.train_dataloader = train_dataloader

        weight, rootweight = list(), list()
        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
        for name in dict_label_to_num.keys():
            # print( len(original_dataset[original_dataset['label'] == name]) )
            weight.append(1/len(original_dataset[original_dataset['label'] == name]))
            rootweight.append(1/math.sqrt(len(original_dataset[original_dataset['label'] == name])))
        self.weight, self.rootweight = torch.Tensor(weight).cuda(), torch.Tensor(rootweight).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        model.cuda()
        outputs = model(**inputs)
        logits = outputs.get("logits") 

        if self.loss_name == 'ce':
            custom_loss = torch.nn.CrossEntropyLoss()
        elif self.loss_name == 'weightedce':
            custom_loss = torch.nn.CrossEntropyLoss(weight = (self.weight))
        elif self.loss_name == 'rootweightedce':
            custom_loss = torch.nn.CrossEntropyLoss(weight = (self.rootweight))    
        elif self.loss_name == 'focal':
            custom_loss = FocalLoss()
        elif self.loss_name == 'f1':
            custom_loss = F1Loss()
        elif self.loss_name == 'default':
            custom_loss = LabelSmoother(epsilon=self.args.label_smoothing_factor)
            loss = custom_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss 
        loss = custom_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataloader == "sequential":
            return SequentialSampler(self.train_dataset)  
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """ ???????????? ???????????? ???????????? ?????? SequentialSampler??? DataLoader ?????? """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )