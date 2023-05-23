from dataclasses import dataclass
from statistics import mean
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
import random
from cot_reward import ZeroShotCoTReward

# Reference: Dataset codes are modified from https://github.com/kojima-takeshi188/zero_shot_cot and https://github.com/mingkaid/rl-prompt
DATASET_NAMES = ["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"]

def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)

def data_reader(dataset):

    if dataset == "aqua":
        dataset_path = "./dataset/AQuA/test.json"
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif dataset == "gsm8k":
        dataset_path = "./dataset/grade-school-math/test.jsonl"
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset == "commonsensqa":
        dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif dataset == "addsub":
        dataset_path = "./dataset/AddSub/AddSub.json"
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset == "multiarith":
        dataset_path = "./dataset/MultiArith/MultiArith.json"
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset == "strategyqa":
        dataset_path = "./dataset/StrategyQA/task.json"
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset == "svamp":
        dataset_path = "./dataset/SVAMP/SVAMP.json"
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset == "singleeq":
        dataset_path = "./dataset/SingleEq/questions.json"
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset == "bigbench_date":
        dataset_path = "./dataset/Bigbench_Date/task.json"
        direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif dataset == "object_tracking":
        dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif dataset == "coin_flip":
        dataset_path = "./dataset/coin_flip/coin_flip.json"
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset == "last_letters":
        dataset_path = "./dataset/last_letters/last_letters.json"
        direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if dataset == "aqua":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif dataset == "gsm8k":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif dataset == "commonsensqa":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif dataset in ("addsub", "multiarith", "singleeq"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif dataset == "strategyqa":
      with open(dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif dataset == "svamp":
      with open(dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif dataset in ("bigbench_date", "object_tracking"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)            
          
    elif dataset in ("coin_flip", "last_letters"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("Dataset : {}".format(dataset))
    print("Data size : {}".format(len(answers)))
    print("Average num of words for each sample : {}".format(q_len_mean))
    
    return list(zip(questions, answers)), direct_answer_trigger

class ZeroShotCoTDataset(Dataset):
    def __init__(
        self, 
        question_label_pair: List[tuple], 
    ):
        self.question_label_pair = question_label_pair

    def __len__(self):
        return len(self.question_label_pair)

    def __getitem__(self, idx):
        item = {'source_texts': self.question_label_pair[idx][0],
                'class_labels': self.question_label_pair[idx][1]}
        return item


def make_zero_shot_chain_of_thought_dataset(
        config: "DictConfig") -> Tuple[ZeroShotCoTDataset]:
    question_answer_pairs, direct_answer_trigger = data_reader(config.dataset) 
    random.seed(config.dataset_seed)
    random.shuffle(question_answer_pairs)
    train_size, dev_size = config.train_set_size, config.val_set_size
    data_dict = {}
    data_dict['train'] = ZeroShotCoTDataset(question_answer_pairs[:train_size])
    data_dict['dev'] = ZeroShotCoTDataset(question_answer_pairs[train_size:dev_size])
    data_dict['test'] = ZeroShotCoTDataset(question_answer_pairs[train_size+dev_size:])

    return (data_dict['train'], data_dict['dev'], data_dict['test'], direct_answer_trigger)

@dataclass
class ZeroShotCoTDatasetConfig:
    batch_size: 20
    dataset: str = "aqua"
    dataset_seed: Optional[int] = 42 
    base_path: str = ''
    train_dev_split_size: tuple = (20, 60)


def make_chain_of_thought_reward(config: "DictConfig") -> ZeroShotCoTReward:
    return ZeroShotCoTReward(
                            model_name = config.model_name,
                            compute_zscore = config.compute_zscore,
                            incorrect_coeff = config.incorrect_coeff,
                            correct_coeff = config.correct_coeff,
                            openai_key = config.openai_key,
                            max_length = config.max_length,
                            temperature = config.temperature,
                            api_time_interval = config.api_time_interval,
                            method = config.method,
                            )


@dataclass
class ZeroShotCoTRewardConfig:
    model_name: str = 'gpt3'
    compute_zscore: bool = True
    incorrect_coeff: float = 180.0
    correct_coeff: float = 200.0
    openai_key: str = "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"
    max_length: int = 256
    temperature: int = 0
    api_time_interval: float = 0
    method: str = "zero-shot-cot"
    
