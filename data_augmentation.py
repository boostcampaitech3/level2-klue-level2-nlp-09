import os
import random
import numpy as np
import pandas as pd
import pickle as pickle
from load_data import *


random.seed(42)
PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3

# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    """ 랜덤한 위치에 랜덤하게 문장부호 추가 """
    words = sentence.split()
    # words = okt.morphs(sentence)  # okt 형태소 분석기 사용
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))  # punc를 몇 개 추가할지 선택(min=1, max=문장길이/3) - 논문 기반
    qs = random.sample(range(0, len(words)), q)  # 1 ~ 문장길이 개의 위치 punc 추가할 위치 선택

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])  # 랜덤하게 추가할 punc 선택
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line

def iter_punc(num_iter, sentence, punc_ratio=PUNC_RATIO):
    """ num_iter 만큼 aeda 수행 """
    new_lines = [sentence]
    for i in range(num_iter):
        new_line = insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO)
        new_lines.append(new_line)
        
    return new_lines

def count_label(train_dataset):
    """  라벨별 개수 구하기 """
    num_dict = {}
    labels = train_dataset['label'].unique()

    for i in range(len(labels)):
        df = train_dataset[train_dataset['label'] == labels[i]]
        num_dict[labels[i]] = len(df)
    
    nums = sorted(list(num_dict.values()))  # 클래스별 데이터 개수 정렬
    quarter1_idx = len(nums[:-1]) // 4  # 가장 많은 no_relation은 제외
    median_idx = quarter1_idx * 2
    quarter3_idx = quarter1_idx * 3

    quarter1 = nums[:-1][quarter1_idx]
    median = nums[:-1][median_idx]
    quarter3 = nums[:-1][quarter3_idx]
    avg = np.average(nums[:-1])

    return num_dict, quarter1, median, quarter3

def get_num_iter(train_dataset, label):
    """ 문장부호 추가 증강을 반복할 횟수 계산 """
    num_iter = 0
    iter1 = ['per:product', 'per:place_of_residence', 'per:origin']
    iter2 = ['org:members', 'per:other_family', 'org:political/religious_affiliation', 'per:place_of_birth']

    if label == iter1:
        num_iter = 4
    elif label in iter2:
        num_iter = 2
    
    return num_iter
    
def aeda(dataset, undersamp=False):
    """ 문장부호 추가 증강 수행 후 csv파일로 저장 """
    total_id = []
    total_sent = []
    total_sub = []
    total_obj = []
    total_label = []
    total_sub_type = []
    total_obj_type = []

    num_dict, quarter1, median, quarter3 = count_label(dataset)

    for i in range(len(dataset)):
        label = dataset['label'].iloc[i]
        num_iter = get_num_iter(dataset, label)
                                
        new_id = [dataset['id'].iloc[i]] * (num_iter+1)
        new_sentence = iter_punc(num_iter, list(dataset['sentence'])[i], PUNC_RATIO)
        new_sub = [dataset['subject_entity'].iloc[i]] * (num_iter+1)
        new_obj = [dataset['object_entity'].iloc[i]] * (num_iter+1)
        new_label = [label] * (num_iter+1)
        new_sub_type = [dataset['subject_type'].iloc[i]] * (num_iter+1)
        new_obj_type = [dataset['object_type'].iloc[i]] * (num_iter+1)
        
        total_id += new_id
        total_sent += new_sentence
        total_sub += new_sub
        total_obj += new_obj
        total_label += new_label
        total_sub_type += new_sub_type
        total_obj_type += new_obj_type

    aug_df = pd.DataFrame({'id':total_id, 'sentence':total_sent, 'subject_entity':total_sub,
                            'object_entity':total_obj, 'label':total_label, 
                            'subject_type':total_sub_type,'object_type':total_obj_type,})
    if undersamp:
        # undersampling 'no_relation'
        no_relation_df = aug_df[aug_df['label'] == 'no_relation']
        print("no_relation undersampling 이전: ", len(no_relation_df))
        
        no_relation_df = no_relation_df.sample(frac=0.7, random_state=42).reset_index(drop=True) 
        print("no_relation undersampling 이후: ", len(no_relation_df))

        res_df = aug_df[aug_df['label'] != 'no_relation']
        aug_df = pd.concat([no_relation_df, res_df])

    return aug_df



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


def apply_swap(train_df):
    """ swap해도 라벨이 동일한 클래스에 대해서만 swap 수행 """
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



def main_aeda(dataset_dir, marking_mode="normal", undersamp=False):
    save_dir = "aeda_augmentation.csv"
    train_dataset, eval_dataset = load_aug_data(dataset_dir, train=True, filter=False, marking_mode=marking_mode)  # 전처리 완료된 dataframe 사용
    aug_dataset = aeda(train_dataset, undersamp=undersamp)
    aug_dataset.to_csv(save_dir, index=False, encoding="utf-8-sig")
    
    print("현재 사용중인 marking_mode: ", marking_mode)
    print("aug 이전 데이터 개수: ", len(train_dataset))
    print("문장부호 추가로 증강한 데이터 개수: ", len(aug_dataset) - len(train_dataset))
    print("aug 이후 데이터 개수: ", len(aug_dataset))
    print('@@@@@@@@@@@@@@@@ Done @@@@@@@@@@@@@@@@')

def main_swap(dataset_dir):
    pd_dataset = pd.read_csv(dataset_dir)
    train_df, val_df = stratified_choice_train_test_split(pd_dataset, test_size=0.2) 
    train_df = preprocessing_swap(train_df, False)
    swap_dataset = apply_swap(train_df)
    swap_dataset.to_csv("swap_and_original.csv", index=False, encoding="utf-8-sig")

    print('원본 데이터 개수: ', len(train_df))
    print('swap으로 증강한 데이터 개수: ', len(swap_dataset) - len(train_df))
    print('원본+swap 데이터 개수: ', len(swap_dataset))


if __name__ == "__main__":
    # 본인에게 맞는 원본 train dataset csv경로를 넣어주세요.
    dataset_dir = "train_df_v1.csv"
    main_aeda(dataset_dir, marking_mode="normal", undersamp=True)  # aeda usage
    # main_swap(dataset_dir)  # swap entity usage