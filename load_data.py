import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from swap_entity import *
from aeda import *


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
  subject_type = []
  object_type = []
  sentences = []
  filtered_sentence = sentence_filter(dataset['sentence'], filter) # sentence filter
  for sub,obj,sentence in zip(dataset['subject_entity'], dataset['object_entity'], filtered_sentence):
    sub = eval(sub)
    obj = eval(obj)
    subject_entity.append(sub['word'])
    object_entity.append(obj['word'])
    subject_type.append(sub['type'])
    object_type.append(obj['type'])
    sentences.append(sentence_marking(sentence, sub, obj, marking_mode)) # sentence_marking
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_entity,'object_entity':object_entity,
  'subject_type':subject_type,'object_type':object_type, 'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, train=True, filter=False ,marking_mode="normal"):
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

def load_aug_data(dataset_dir, train=True, filter=False, marking_mode="normal", aug_type=False, save=False):
  """ 
  csv 파일을 경로에 맡게 불러 옵니다. 
  train_test_split: choice_train_test_split, stratified_choice_train_test_split 
  sentence_filter: True, False
  marking_mode: normal, entity, typed_entity, typed_entity_punc
  aug_type: swap, aeda
  """
  pd_dataset = pd.read_csv(dataset_dir)
  if train:
    ### augmentation 적용 안하는 경우
    pd_train, pd_eval = stratified_choice_train_test_split(pd_dataset, test_size=0.2) 
    aug_dataset = preprocessing_dataset(pd_train, filter, marking_mode)
    eval_dataset = preprocessing_dataset(pd_eval, filter, marking_mode)

    if aug_type == "swap":
      train_dataset = preprocessing_swap(pd_train, False)
      aug_dataset = apply_swap(train_dataset)  # swap entity
      aug_dataset = preprocessing_dataset(aug_dataset, False, marking_mode=marking_mode)  # sentence_marking
      # aug_dataset = swap_dataset

      print("현재 사용중인 marking_mode: ", marking_mode)
      print('원본 데이터 개수: ', len(train_dataset))
      print('swap으로 증강한 데이터 개수: ', len(aug_dataset) - len(train_dataset))
      print('원본+swap 데이터 개수: ', len(aug_dataset))
      print('############################  Done with Swap Entity Augmentation ############################')

    elif aug_type == "aeda":
      train_dataset = preprocessing_swap(pd_train, False)
      swap_dataset = apply_swap(train_dataset)  # swap entity
      swap_dataset = preprocessing_dataset(swap_dataset, False, marking_mode=marking_mode)  # sentence_marking
      aug_dataset = aeda(swap_dataset)  # aeda

      print("현재 사용중인 marking_mode: ", marking_mode)
      print('원본 데이터 개수: ', len(train_dataset))
      print('swap으로 증강한 데이터 개수: ', len(aug_dataset) - len(swap_dataset))
      print('원본+swap 데이터 개수: ', len(swap_dataset))
      print("aeda로 증강한 데이터 개수: ", len(aug_dataset) - len(train_dataset))
      print("aug 이후 데이터 개수: ", len(aug_dataset))
      print('############################  Done with Swap Entity & AEDA Augmentation ############################')

    if save:
      save_dir = "final_aug_dataset_" + marking_mode + ".csv"
      aug_dataset.to_csv(save_dir, index=False, encoding="utf-8-sig")
    return aug_dataset, eval_dataset
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
def tokenized_dataset(dataset, tokenizer, type):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
    if type == "multi":
      temp = f"{e01}[SEP]{e02} 어떤 관계일까?"
    elif type =="entity":
      temp = f"[sub]{e01}[/sub] [obj]{e02}[/obj] 어떤 관계일까?"
    elif type == "typed_entity":
      temp = f"<S:{t01}> {e01} </S:{t01}> <O:{t02}> {e02} </O:{t02}> 어떤 관계일까?"
    elif type == "typed_entity_punc":
      temp = f"@ * {t01} * {e01} @ # ^ {t02} ^ {e02} # 어떤 관계일까?" 
    else:
      temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity, # for single_sentence
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256, # 512로 변경 가능/ amp 미사용에 batch_size=32로 설정하면 256으로 줄여야 OOM 발생X
      add_special_tokens=True,
      )
  return tokenized_sentences

if __name__ == '__main__':
  # load_data("../dataset/train/train.csv", train=True, filter=False ,marking_mode="typed_entity_punc")
  dataset_dir = "../dataset/train/train.csv"
  load_aug_data(dataset_dir, train=True, filter=False, marking_mode="typed_entity_punc")
  # sen = sentence_filter(pd.Series(["◆▶ ♧'문찬국'(文讚國, 1995~) ☆ §", "애플은 옳고 그름에 대한 감각이 없으며 진실을 외면했다라며 비난했다."]), True)
