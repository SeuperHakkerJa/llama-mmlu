import numpy as np
import torch
import pandas as pd
# from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import copy
import os
from testing import load_model
from tqdm import tqdm
import gc


choices = ["A", "B", "C", "D"]
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def get_ans(base_model, tokenizer, text, top_n=1):
    inputs = tokenizer(text, return_tensors='pt')
    logits = base_model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda()).logits[0, -1]
    
    # Create a list of tuples having (logit, 'option') format
    options_list = [(logits[tokenizer('\nA').input_ids[-1]], 'A'), (logits[tokenizer('\nB').input_ids[-1]], 'B'), (logits[tokenizer('\nC').input_ids[-1]], 'C'), (logits[tokenizer('\nD').input_ids[-1]], 'D'), (logits[tokenizer('\nE').input_ids[-1]], 'E')] 
    options_list = sorted(options_list, reverse=True)
    ans_list = []
    del logits, inputs
    gc.collect(2)
    torch.cuda.empty_cache()
    # giving the top n most likely result. top_n default set to 1
    for i in range(top_n):
        ans_list.append(options_list[i][1])
    del options_list
    return ans_list


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(n_shot, subject, base_model, tokenizer , dev_data, test_data):

    # create no answer prompt for test_data
    # _test_data = copy.deepcopy(test_data).map(format_prompt_no_answer)
    # _dev_data = copy.deepcopy(dev_data)

    test_data = pd.DataFrame(test_data)
    dev_data = pd.DataFrame(dev_data)

    # number of correct answers
    cors = []

    for i in tqdm(range(test_data.shape[0])):
        prompt_end = format_example(test_data, i, include_answer=False)
        train_prompt = gen_prompt(dev_data, subject, n_shot)
        
        prompt = train_prompt + prompt_end

        # get ground truth label
        gt_label = test_data.iloc[i, test_data.shape[1]-1]
        answer_list = get_ans(base_model, tokenizer, prompt)[0] 
        answer = answer_list[0]

        cor = answer == gt_label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    return acc, cors, None


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_name = 'NousResearch/Nous-Hermes-Llama2-13b'
    base_model, tokenizer = load_model(model_name, bnb_config)


    n_shot = 5
    data_dir = 'data-original'
    save_dir = 'llama2-mmlu-results'


    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])
    subjects = subjects[:10]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(subjects)
    all_cors = []
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:n_shot]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        acc, cors, probs = eval(n_shot, subject, base_model, tokenizer, dev_df, test_df)
        all_cors.append(cors)
        print(subject, 'accuracy:', acc)

        del dev_df
        del test_df
        del cors, acc
        gc.collect(2)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()