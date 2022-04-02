import pickle as pickle
import os
import pandas as pd
import numpy as np
from load_data import *
import random


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

    # 분포 고르게 다시 조정
    iter0 = ['org:alternate_names', 'org:top_members/employees', 'per:employee_of']
    iter1 = ['org:member_of', 'per:alternate_names', 'per:spouse', 'per:title']
    iter2 = ['org:place_of_headquarters', 'per:date_of_birth', 'per:origin']
    iter4 = ['org:founded', 'org:members', 'org:product', 'per:children', 'per:colleagues', 'per:date_of_death', 'per:parents']
    iter8 = ['org:dissolved', 'org:founded_by', 'org:number_of_employees/members', 'org:political/religious_affiliation',
            'per:other_family', 'per:place_of_birth', 'per:place_of_death', 'per:place_of_residence', 'per:product',
            'per:religion', 'per:schools_attended', 'per:siblings']
    
    # under sampling 수행X
    if label == 'no_relation':
        num_iter = 0
    elif label in iter0:
        num_iter = 0
    elif label in iter1:
        num_iter = 1
    elif label in iter2:
        num_iter = 2
    elif label in iter4:
        num_iter = 4
    elif label in iter8:
        num_iter = 8
    
    return num_iter
    

def aeda(train_dataset):
    """ 문장부호 추가 증강 수행 후 csv파일로 저장 """
    total_id = []
    total_sent = []
    total_sub = []
    total_obj = []
    total_label = []

    num_dict, quarter1, median, quarter3 = count_label(train_dataset)
    # print('train dataset 개수:', len(train_dataset))
    # # print('eval dataset 개수:', len(eval_dataset))
    # print('class 개수: ', len(num_dict))
    # print("=============================")
    # print('quarter1:', quarter1)
    # print('median:', median)
    # print('quarter3:', quarter3)

    for i in range(len(train_dataset)):
        label = train_dataset['label'].iloc[i]
        num_iter = get_num_iter(train_dataset, label)
                                
        new_id = [train_dataset['id'].iloc[i]] * (num_iter+1)
        new_sentence = iter_punc(num_iter, list(train_dataset['sentence'])[i], PUNC_RATIO)
        new_sub = [train_dataset['subject_entity'].iloc[i]] * (num_iter+1)
        new_obj = [train_dataset['object_entity'].iloc[i]] * (num_iter+1)
        new_label = [label] * (num_iter+1)
        
        total_id += new_id
        total_sent += new_sentence
        total_sub += new_sub
        total_obj += new_obj
        total_label += new_label

    aug_df = pd.DataFrame()
    aug_df['id'] = total_id
    aug_df['sentence'] = total_sent
    aug_df['subject_entity'] = total_sub
    aug_df['object_entity'] = total_obj
    aug_df['label'] = total_label

    return aug_df
        

# if __name__ == "__main__":
#     marking_mode = "normal"

#     # 본인에게 맞는 원본 train dataset csv경로를 넣어주세요.
#     dataset_dir = "/opt/ml/dataset/train/train_df_latest.csv"
#     aug_dir = "swap_and_original.csv"
#     save_dir = "aug_with_undersam_normal.csv"
#     # save_dir = "aug_without_undersam_normal.csv"

#     # train_dataset, eval_dataset = load_data2(dataset_dir, train=True, filter=False, marking_mode=marking_mode)  # 전처리 완료된 dataframe 사용
#     train_dataset, eval_dataset = load_aug_data2(dataset_dir, aug_dir, train=True, filter=False, marking_mode=marking_mode)  # 전처리 완료된 dataframe 사용

#     # aug_dataset = aeda(train_dataset)
#     # aug_dataset.to_csv("aug_" + marking_mode + ".csv", index=False, encoding="utf-8-sig")
#     # print("aug 후 데이터 개수: ", len(aug_dataset))
#     # print('Done!!!!!!!!!!!!!')
    
    
#     aug_dataset = aeda(train_dataset)
#     aug_dataset.to_csv(save_dir, index=False, encoding="utf-8-sig")
    
#     print("현재 사용중인 marking_mode: ", marking_mode)
#     print("aug 이전 데이터 개수: ", len(train_dataset))
#     print("문장부호 추가로 증강한 데이터 개수: ", len(aug_dataset) - len(train_dataset))
#     print("aug 이후 데이터 개수: ", len(aug_dataset))
#     print('@@@@@@@@@@@@@@@@ Done @@@@@@@@@@@@@@@@')