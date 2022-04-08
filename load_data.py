import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class RE_Dataset(torch.utils.data.Dataset):
  """
  Dataset 구성을 위한 class.
  __get_item__ 시 반환하는 형태 (학습으로 들어가는 데이터의 형태): 
  {'input_ids': tensor, 'token_type_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
  """

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
  """ 
  처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
  """
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
  """ 라벨별로 test_size 비율로 추출합니다 (dict_label_to_num.pkl 경로 확인 필수)"""
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

def load_data(dataset_dir, train=True, filter=False ,marking_mode="normal", split_function = stratified_choice_train_test_split):
  """ 
  csv 파일을 경로에 맡게 불러 옵니다. 

  parameters
  - train: boolean
  True 일때 train, val data 스플릿해서 load (default)
  False 일때 스플릿 없이 data load

  - filter: boolean
  True 일때 일부 특수문자 제거
  False 일때는 그대로 (default)

  - marking_mode: str ["normal", "entity", "typed_entity", "typed_entity_punc"]
  entity에 마킹해주는 함수 각 marking_mode를 거치면 아래와 같이 문자들이 처리됨
  "normal": "나는 부스트캠프의 김태일이다." (default)
  "entity": "나는 [sub]부스트캠프[/sub]의 [obj]김태일[/obj]이다."
  "typed_entity": "나는 <S:ORG>부스트캠프</S:ORG>의 <O:PER>김태일</O:PER>이다."
  "typed_entity_punc": "나는 @ * ORG * 부스트캠프 @의 # ^ PER ^ 김태일 #이다."

  - split_function: split function [load_data.stratified_choice_train_test_split, load_data.choice_train_test_split]
  choice_train_test_split: 분포 상관 없는 일반적인 스플릿 
  stratified_choice_train_test_split: 각 분포별로 동일한 비율로 스플릿 (default)
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
    elif marking_mode == "normal": 
      pass
    else:
      raise Exception(f"unexpected marking_mode: {marking_mode}")
    return sentence


### 데이터셋 토크나이즈
def tokenized_dataset(dataset, tokenizer, tokenize_type = "normal"):
  """
  tokenizer에 따라 sentence를 tokenizing 합니다.

  - tokenize_type : str ["multi", "entity", "typed_entity", "typed_entity_punc", "normal"]

  각 tokenize_type를 거치면 아래와 같이 문자들이 처리됨

  원문: "나는 부스트캠프의 김태일이다." 부스트캠프, 김태일

  * "normal": "[CLS] 부스트캠프 [SEP] 김태일 [SEP] 나는 부스트캠프의 김태일이다." 
  * "multi": "부스트캠프 김태일 어떤 관계일까? 나는 부스트캠프의 김태일이다."
  * "entity": "[CLS] [sub]부스트캠프[/sub] [obj]김태일[/obj] 어떤 관계일까? [SEP] 나는 [sub]부스트캠프[/sub]의 [obj]김태일[/obj]이다."
  * "typed_entity": "[CLS] <S:ORG>부스트캠프</S:ORG> <O:PER>김태일</O:PER> 어떤 관계일까? [SEP] 나는 <S:ORG>부스트캠프</S:ORG>의 <O:PER>김태일</O:PER>이다."
  * "typed_entity_punc": "[CLS] @ * ORG * 부스트캠프 @ # ^ PER ^ 김태일 # 어떤 관계일까? [SEP] @ * ORG * 부스트캠프 @의 # ^ PER ^ 김태일 #이다."
  """
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
    if tokenize_type == "multi":
      temp = f"{e01}[SEP]{e02} 어떤 관계일까?"
    elif tokenize_type =="entity":
      temp = f"[sub]{e01}[/sub] [obj]{e02}[/obj] 어떤 관계일까?"
    elif tokenize_type == "typed_entity":
      temp = f"<S:{t01}> {e01} </S:{t01}> <O:{t02}> {e02} </O:{t02}> 어떤 관계일까?"
    elif tokenize_type == "typed_entity_punc":
      temp = f"@ * {t01} * {e01} @ # ^ {t02} ^ {e02} # 어떤 관계일까?" 
    elif tokenize_type == "normal":
      temp = e01 + '[SEP]' + e02
    else:
      raise Exception(f"unexpected tokenize_type: {tokenize_type}")
    concat_entity.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512,
      add_special_tokens=True,
      )
  return tokenized_sentences

if __name__ == '__main__':
  load_data("../dataset/train/train.csv", train=True, filter=False ,marking_mode="typed_entity_punc")
  # sen = sentence_filter(pd.Series(["◆▶ ♧'문찬국'(文讚國, 1995~) ☆ §", "애플은 옳고 그름에 대한 감각이 없으며 진실을 외면했다라며 비난했다."]), True)
