import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from data_aug import *


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset, filter, marking_mode):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sentences = []
  filtered_sentence = sentence_filter(dataset['sentence'], filter) # sentence filter
  for sub,obj,sentence in zip(dataset['subject_entity'], dataset['object_entity'], filtered_sentence):
    sub = eval(sub)
    obj = eval(obj)
    subject_entity.append(sub['word'])
    object_entity.append(obj['word'])
    sentences.append(sentence_marking(sentence, sub, obj, marking_mode)) # sentence_marking
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, train=True, filter=False, marking_mode="normal"):
  """ 
  csv 파일을 경로에 맡게 불러 옵니다. 
  train_test_split: choice_train_test_split, stratified_choice_train_test_split 
  sentence_filter: True, False
  marking_mode: normal, entity, typed_entity, typed_entity_punc
  """
  pd_dataset = pd.read_csv(dataset_dir)
  # train_test split
  if train:
    pd_train, pd_eval = stratified_choice_train_test_split(pd_dataset, test_size=0.2) 
    train_dataset = preprocessing_dataset(pd_train, filter, marking_mode)
    eval_dataset = preprocessing_dataset(pd_eval, filter, marking_mode)
    return train_dataset, eval_dataset
  else:
    test_dataset = preprocessing_dataset(pd_dataset, filter, marking_mode)
    return test_dataset

def load_aug_data(dataset_dir, aug_dir, train=True, filter=False, marking_mode="normal"):
    """ augmentation 적용된 csv파일 로드 """
    pd_dataset = pd.read_csv(dataset_dir)

    # train_test split
    if train:
        pd_train, pd_eval = stratified_choice_train_test_split(pd_dataset, test_size=0.2) 
        # train_dataset = preprocessing_dataset(pd_train, filter, marking_mode)  # 기존
        train_dataset = pd.read_csv(aug_dir)  # data augmentation 적용된 csv파일
        eval_dataset = preprocessing_dataset(pd_eval, filter, marking_mode)
        return train_dataset, eval_dataset
    else:
        test_dataset = preprocessing_dataset(pd_dataset, filter, marking_mode)
        return test_dataset

## sentence 전처리
def sentence_filter(sentence, filter=False):
    if filter:
        senten = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]', repl='', regex=True) # 한자 포함
    else:
        senten = sentence
    return senten

def sentence_marking(sentence, sub, obj, marking_mode):
    if marking_mode == "entity":              # [sub]aaa[/sub]bbb[obj]ccc[/obj]
      sentence = sentence.replace(sub['word'], f"[sub] {sub['word']} [/sub]")
      sentence = sentence.replace(obj['word'], f"[obj] {obj['word']} [/obj]")
    elif marking_mode =="typed_entity":       # [S:type]aaa[/S:type]bbb[O:type]ccc[/O:type]
      sentence = sentence.replace(sub['word'], f"<S:{sub['type']}> {sub['word']} </S:{sub['type']}>")
      sentence = sentence.replace(obj['word'], f"<O:{obj['type']}> {obj['word']} </O:{obj['type']}>")
    elif marking_mode == "typed_entity_punc": # @*aaa*@ bbb #&ccc&#
      sentence = sentence.replace(sub['word'], f"@ * {sub['type']} * {sub['word']} @")
      sentence = sentence.replace(obj['word'], f"# ^ {obj['type']} ^ {obj['word']} #")
    else: 
      sentence=sentence
    return sentence

### train_test_split ()
def choice_train_test_split(X, test_size=0.2, shuffle=True, random_state=42):
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num
    if shuffle:
        np.random.seed(random_state)
        train_idx = np.random.choice(X.shape[0], train_num, replace=False)
        test_idx = np.setdiff1d(range(X.shape[0]), train_idx)
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X.iloc[:train_num]
        X_test = X.iloc[train_num:]
    return X_train, X_test
  
def stratified_choice_train_test_split(X, test_size=0.2, random_state=42):
  """ 라벨별로 일정 비율로 추출합니다 (dict_label_to_num.pkl 경로 확인 필수)"""
  split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
  group = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in X['label'].values:
    group.append(dict_label_to_num[v])
  for train_idx, test_idx in split.split(X, group):
      X_train = X.iloc[train_idx]
      X_test = X.iloc[test_idx]
  return X_train, X_test

### 데이터셋 토크나이즈
def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

if __name__ == '__main__':
  load_data("../dataset/train/train.csv", train=True, filter=False ,marking_mode="typed_entity_punc")
  # sen = sentence_filter(pd.Series(["◆▶ ♧'문찬국'(文讚國, 1995~) ☆ §", "애플은 옳고 그름에 대한 감각이 없으며 진실을 외면했다라며 비난했다."]), True)
