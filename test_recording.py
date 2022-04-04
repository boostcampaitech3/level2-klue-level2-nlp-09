import json
import os

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
    "tokenized":  config['tokenize_mode'], 
    "train_batch_size": args_dict['per_device_train_batch_size'],
    "fp16": args_dict['fp16'],
    "fp16_opt_level": args_dict['fp16_opt_level'],
    "epoch": args_dict['num_train_epochs'],
    "warmup_ratio": args_dict['warmup_ratio'],
    "adam_epsilon": args_dict['adam_epsilon'],
    "learning_rate": args_dict['learning_rate'],
    "max_seq_length": 512
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