import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
import seaborn as sns
from load_data import *

## sentence 전처리
def sentence_filter(sentence, filter=False):
    if filter:
        senten = sentence.str.replace(pat='[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]', repl='', regex=True) # 한자 포함
    else:
        senten = sentence
    return senten

def swap_entity(sent, sub_s, sub_e, obj_s, obj_e):
    new_sent = ''
    if sub_s < obj_s:
        # sub가 obj보다 먼저 등장하는 경우:
        new_sent = sent[:sub_s] + sent[obj_s:obj_e+1] + sent[sub_e+1:obj_s] + sent[sub_s:sub_e+1] + sent[obj_e+1:]

    else:
        # sub가 obj보다 나중에 등장하는 경우:
        new_sent = sent[:obj_s] + sent[sub_s:sub_e+1] + sent[obj_e+1:sub_s] + sent[obj_s:obj_e+1] + sent[sub_e+1:]
    return new_sent

def preprocessing_swap(dataset, filter):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subs = []
    objs = []
    sub_s = []
    sub_e = []
    obj_s = []
    obj_e = []
    
    filtered_sentence = sentence_filter(dataset['sentence'], filter) # sentence filter
    for sub,obj,sentence in zip(dataset['subject_entity'], dataset['object_entity'], filtered_sentence):
        sub = eval(sub)
        obj = eval(obj)
        subs.append(sub['word'])
        objs.append(obj['word'])
        sub_s.append(sub['start_idx'])
        sub_e.append(sub['end_idx'])
        obj_s.append(obj['start_idx'])
        obj_e.append(obj['end_idx'])
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'], 
                                'subject_entity':dataset['subject_entity'],'object_entity':dataset['object_entity'], 'label':dataset['label'], 
                                'sub':subs, 'obj':objs, 'sub_s':sub_s, 'sub_e':sub_e, 'obj_s':obj_s, 'obj_e':obj_e})
    return out_dataset

# swap해도 라벨이 동일한 클래스에 대해서만 swap 수행
def apply_swap(train_df):
    same_df1 = train_df[train_df['label'] == 'org:alternate_names']
    same_df2 = train_df[train_df['label'] == 'per:alternate_names']
    # same_df3 = train_df[train_df['label'] == 'per:sibilings']  # 0개
    same_df4 = train_df[train_df['label'] == 'per:spouse']
    same_df5 = train_df[train_df['label'] == 'per:other_family']
    # same_df6 = train_df[train_df['label'] == 'per:colleages']  # 0개

    same_df1['sentence'] = same_df1.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)
    same_df2['sentence'] = same_df2.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)
    # same_df3['sentence'] = same_df3.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)
    same_df4['sentence'] = same_df4.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)
    same_df5['sentence'] = same_df5.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)
    # same_df6['sentence'] = same_df6.apply(lambda x: swap_entity(x['sentence'], x['sub_s'], x['sub_e'], x['obj_s'], x['obj_e']), axis=1)

    res_df = pd.concat([same_df1, same_df2, same_df4, same_df5])
    swap_df = res_df[["id", "sentence", "subject_entity", "object_entity", "label"]]
    train_df = train_df[["id", "sentence", "subject_entity", "object_entity", "label"]]
    total_df = pd.concat([swap_df, train_df])

    return total_df



def main(dataset_dir):
    # dataset_dir = "/opt/ml/dataset/train/train_df_latest.csv"
    pd_dataset = pd.read_csv(dataset_dir)
    train_df, val_df = stratified_choice_train_test_split(pd_dataset, test_size=0.2) 
    train_df = preprocessing_swap(train_df, False)

    ### swap해도 라벨이 동일한 클래스에 대해서만 swap 수행
    swap_dataset = apply_swap(train_df)
    print('원본 데이터 개수: ', len(train_df))
    print('swap으로 증강한 데이터 개수: ', len(swap_dataset) - len(train_df))
    print('원본+swap 데이터 개수: ', len(swap_dataset))
    
    # swap_df.to_csv("only_swap.csv", index=False, encoding="utf-8-sig")
    swap_dataset.to_csv("swap_and_original.csv", index=False, encoding="utf-8-sig")


# if __name__ == "__main__":
#     dataset_dir = "/opt/ml/dataset/train/train_df_latest.csv"
#     main(dataset_dir)
