import json
import os
from transformers import TrainingArguments


def save_record(config, training_args):
    """
    실험 결과 저장을 위한 함수입니다. config와 training_arguments를 입력으로 받아 json파일로 저장합니다.
    주요 파라미터 이외의 새로운 파라미터들을 추가할 수 있습니다. 
    """
    # data setting
    args_dict = args2dict(training_args)
    data_log = {
    "Test_name":  config['test_name'],
    "load_model":  config['model_name'],
    "filter":  config['sentence_filter'],
    "marking_mode":  config['marking_mode'],
    "tokenized":  config['tokenized_function'],
    "epoch": args_dict['save_total_limit'], 
    "learning_rate": args_dict['learning_rate'], 
    "train_batch_size": args_dict['per_device_train_batch_size']
    }

    # path setting
    directory = './test_recording/'
    if not os.path.exists(directory): os.makedirs(directory)

    idx =str(len(os.listdir(directory)))
    filename = f"{idx}_{config['test_name']}.json"
    with open(os.path.join(directory, filename),'w') as f:
        json.dump(data_log, f, ensure_ascii=False, indent=4)
    print(f"{filename} saved in {directory}")

def args2dict(training_args):
    temp = str(training_args)
    temp = temp.replace(',', "")
    temp_ls = temp.split("\n")[1:-1]
    args_dict={}
    for param in temp_ls:
        key, value = param.split("=")
        args_dict[key] = value
    return args_dict

if __name__ == '__main__':
    with open("config.json","r") as js:
        config = json.load(js)
    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
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
    report_to="wandb",  # enable logging to W&B
    fp16 = True,        # whether to use 16bit (mixed) precision training
    fp16_opt_level = 'O1' # choose AMP optimization level (AMP Option:'O1' , 'O2')(FP32: 'O0')
    )
    save_record(config, training_args)