import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np


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

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  # sentence preprocessing들어가야 한다.
  filtered_sentence = sentence_filter(dataset['sentence'], hanza=True)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':filtered_sentence ,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, test_size, shuffle):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  # train_test split
  pd_train, pd_eval = choice_train_test_split(pd_dataset, test_size, shuffle)
  train_dataset = preprocessing_dataset(pd_train)
  eval_dataset = preprocessing_dataset(pd_eval)
  return train_dataset, eval_dataset

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
        X_train = X.iloc[:train_idx]
        X_test = X.iloc[test_idx:]
    return X_train, X_test

def sentence_filter(sentence, hanza=False):
    if hanza:
        series = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]|[一-龥]+, ', repl='', regex=True) # 한자, 공백
    else:
        series = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]', repl='', regex=True)
    return series

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