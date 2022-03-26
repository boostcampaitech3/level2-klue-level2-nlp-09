import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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

def preprocessing_dataset(dataset, marking_mode):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sentences = []
  filtered_sentence = sentence_filter(dataset['sentence'], hanza=False)
  for sub,obj,sentence in zip(dataset['subject_entity'], dataset['object_entity'], filtered_sentence):
    sub =eval(sub)
    obj =eval(obj)
    subject_entity.append(sub['word'])
    object_entity.append(obj['word'])
    #marking 모드 선택, 모드 추가 시 이 코드에 elif 추가 + json 파일에 special token 추가해야함
    if marking_mode == "entity":
      sentence = sentence.replace(sub['word'], f"[sub] {sub['word']} [/sub]")
      sentence = sentence.replace(obj['word'], f"[obj] {obj['word']} [/obj]")
    elif marking_mode =="typed_entity":
      sentence = sentence.replace(sub['word'], f"<S:{sub['type']}> {sub['word']} </S:{sub['type']}>")
      sentence = sentence.replace(obj['word'], f"<O:{obj['type']}> {obj['word']} </O:{obj['type']}>")
    sentences.append(sentence)
  # sentence preprocessing들어가야 한다.
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, test_size, shuffle, marking_mode="normal"):
  """ csv 파일을 경로에 맡게 불러 옵니다. 현재 Default split 방법은 Train_test_split 이며, 
  변경하려면 아래 코드에서 choice_train_test_split 함수를 stratified_choice_train_test_split 함수로 변경해주셔야합니다 """
  pd_dataset = pd.read_csv(dataset_dir)
  # train_test split
  pd_train, pd_eval = stratified_choice_train_test_split(pd_dataset,test_size)
  train_dataset = preprocessing_dataset(pd_train, marking_mode)
  eval_dataset = preprocessing_dataset(pd_eval, marking_mode)
  return train_dataset, eval_dataset

def load_test_data(dataset_dir, marking_mode="normal"):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, marking_mode)
  return dataset

def choice_train_test_split(X, test_size=0.2, shuffle=True, random_state=15):
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num
    if shuffle:
        np.random.seed(random_state)
        train_idx = np.random.choice(X.shape[0], train_num, replace=False)
        #-- way 1: using np.setdiff1d() # trainset의 idndes를 제외한 나머지 index 추출 # 차집합 연산
        test_idx = np.setdiff1d(range(X.shape[0]), train_idx)
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X.iloc[:train_num]
        X_test = X.iloc[train_num:]
    return X_train, X_test


def sentence_filter(sentence, hanza=False):
    if hanza:
        series = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]|[一-龥]+, ', repl='', regex=True) # 한자, 공백
    else:
        series = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]', repl='', regex=True)
    return series
  
def stratified_choice_train_test_split(X, test_size=0.2, random_state=15):
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
  load_data("../dataset/train/train.csv", test_size=0.2, shuffle=True)