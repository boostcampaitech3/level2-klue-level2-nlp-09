import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from load_data import *
import wandb
import json
import random
from test_recording import *

def seed_everything(seed: int = 42):
    """Random seed(Reproducibility)"""
    random.seed(seed)                              
    np.random.seed(seed)                           
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)                        
    torch.cuda.manual_seed(seed)  # type: ignore    
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  return num_label

def train():
  # load_parameter: tokenizer, sentence preprocessing
  with open("config.json","r") as js:
    config = json.load(js)
    load_model = config['model_name']        # model
    filter = config['sentence_filter']       # sentence_filter
    marking_mode = config['marking_mode']    # marking_mode
    tokenize_mode = config['tokenize_mode'] # tokenize_function
    wandb_name = config['test_name']
  
  # load model and tokenizer  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = load_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  print("#################################################################################################################### \n",
        f"Model_name: {MODEL_NAME}, Filter: {filter}, Marking_mode: {marking_mode}, Tokenized_function: {tokenize_mode}\n",
        "#################################################################################################################### \n")

  # load dataset
  dataset_dir = "../dataset/train/train.csv"
  train_dataset, dev_dataset = load_data(dataset_dir, train=True, filter=filter, marking_mode=marking_mode)
  # train_dataset, dev_dataset = load_data(dataset_dir, train=True, filter=filter, marking_mode=marking_mode, aug_type="swap", save=True)  # augmentation 사용시
  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)
  
  # add vocab (special tokens)
  with open("marking_mode_tokens.json","r") as json_file:
    mode2special_token = json.load(json_file)
  add_token_num = 0
  if marking_mode != "normal" and  marking_mode != "typed_entity_punc":
    add_token_num += tokenizer.add_special_tokens({"additional_special_tokens":mode2special_token[marking_mode]})
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, tokenize_mode)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, tokenize_mode)
  # print(tokenizer.decode(tokenized_train['input_ids'][0]))

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  #resize models vocab_size(add add_token_num) 
  model.resize_token_embeddings(tokenizer.vocab_size + add_token_num)
  # print(model.config)
  model.parameters
  model.to(device)
  
  project = "KLUE-test"  # W&B Projects
  entity_name = "level2-nlp-09"
  display_name = "wandb-test"  # Model_name displayed in W&B Projects
  wandb.init(project=project, entity=entity_name, name=display_name)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=5,              # total number of training epochs
    learning_rate=3e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    # added max_length in load_data.py
    warmup_ratio = 0.1,  # defalut 0
    adam_epsilon = 1e-6, # default 1e-8
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model = 'micro f1 score',
    report_to="wandb",  # enable logging to W&B
    fp16 = True,        # whether to use 16bit (mixed) precision training
    fp16_opt_level = 'O1' # choose AMP optimization level (AMP Option:'O1' , 'O2')(FP32: 'O0')
  )
  # save test result 
  save_record(config, training_args)
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)] #EarlyStopping callbacks
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
def main():
  train()

if __name__ == '__main__':
  seed_everything(42)
  main()