from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import json
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load_parameter: tokenizer, sentence preprocessing
  tokenize_function_list = {"default": tokenized_dataset}
  with open("config.json","r") as js:
    config = json.load(js)
    load_model = config['model_name']        # model
    filter = config['sentence_filter']       # sentence_filter
    marking_mode = config['marking_mode']    # marking_mode
    tokenize_mode = config['tokenize_mode'] # tokenize_function
  print("####################################################################################################################\n",
        f"Model_name: {load_model}, Filter: {filter}, Marking_mode: {marking_mode}, Tokenized_function: {tokenize_mode}\n",
        "####################################################################################################################\n")

  # load tokenizer
  Tokenizer_NAME = load_model
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  # add_vocab
  with open("marking_mode_tokens.json","r") as json_file:
    mode2special_token = json.load(json_file)
  if marking_mode != "normal" and  marking_mode != "typed_entity_punc":
    tokenizer.add_special_tokens({"additional_special_tokens":mode2special_token[marking_mode]})


  ## load_test_datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_dataset = load_data(test_dataset_dir, train=False, filter=filter, marking_mode=marking_mode)

  test_id = test_dataset['id'] 
  test_label = list(map(int,test_dataset['label'].values))
  
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, tokenize_mode)
  Re_test_dataset = RE_Dataset(tokenized_test ,test_label)


  n_iter = 0
  for idx in range(len(args.model_dir)):
    n_iter += 1
    print('------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------')
    print(f'inference #{n_iter}')
    print('------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------')

    ## load my model
    MODEL_NAME = args.model_dir[idx] # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir[idx])
    model.parameters
    model.to(device)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    # print('pred_answer')
    # print(pred_answer)
    # print('output_prob')
    # print(output_prob)
    # print(output_prob.shape)

    if idx == 0:
      kfold_pred = np.array(output_prob)/len(args.model_dir)
    else:
      kfold_pred += np.array(output_prob)/len(args.model_dir)
  
  pred_answer = []
  for logit in kfold_pred:
    result = np.argmax(logit, axis=-1)
    pred_answer.append(result)
    print(result, end = ' | ')

  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  ################### 아래 directory와 columns의 형태는 지켜주시기 바랍니다.######################################
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':kfold_pred.tolist()})
  output.to_csv('./prediction/submission.csv', index=False) 
  ###################  최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장. ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default=["./last_model_kfold_#1","./last_model_kfold_#2","./last_model_kfold_#3","./last_model_kfold_#4","./last_model_kfold_#5" ])
  args = parser.parse_args()
  print(args)
  main(args)
  
