from filters import tf_idf_filter, lexical_overlap_filter, similarity_filter_for_step_inference
import argparse 
from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nlpturk
import torch
import json
import random
from tqdm import tqdm
from simcse import SimCSE
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')
def main():
    def find_steps_of_the_goal(goal, step2goal):
        goals_array = np.array(list(step2goal.values()))
        indexes = np.where(goals_array == goal)[0]
        return [list(step2goal.keys())[i] for i in indexes]

    def safe_execute_tfidf_filter(step, step_list):
        try:
            return tf_idf_filter(step, step_list)
        except:
            return False
    def safe_execute_lex_filter(positive_candidate, negative_candidates):
        try:
            return lexical_overlap_filter(positive_candidate, negative_candidates)
        except:
            return False
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, help="Path of tokenizer")
    parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
    parser.add_argument("--step_inference_path", type=str, help="Path of step inference data")
    parser.add_argument("--step2goal_path", type=str, help="Path of goal inference data")
    parser.add_argument("--wikihow_path", type=str, help="Path of goal inference data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    step_inference = json.load(open(args.step_inference_path))
    step2goal = json.load(open(args.step2goal_path))
    wikihow = json.load(open(args.wikihow_path))

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    simcse = SimCSE(args.simcse_path)

    filtered_data = {}

    step_inference_keys = list(step_inference.keys())
    tokenized_step_inference_keys = tokenizer(step_inference_keys)["input_ids"]
    step_inference_keys = [step_inference_keys[i] for i in tqdm(np.arange(0, len(step_inference_keys))) if len(tokenized_step_inference_keys[i]) > 10]
    print(len(step_inference_keys))
    step_inference_keys = [key for key in tqdm(step_inference_keys) if similarity_filter_for_step_inference(step2goal[key], step_inference[key]["negative_candidates"], wikihow, simcse)]
    print(len(step_inference_keys))
    step_inference_keys = [key for key in tqdm(step_inference_keys) if safe_execute_lex_filter(key, step_inference[key]["negative_candidates"])]
    print(len(step_inference_keys))
    #step_inference_keys = [key for key in tqdm(step_inference_keys) if safe_execute_tfidf_filter(key, find_steps_of_the_goal(step2goal[key], step2goal))]
    #print(len(step_inference_keys))

    for i in tqdm(np.arange(0, len(step_inference_keys))):
        key = step_inference_keys[i]
        steps = step_inference[key]["negative_candidates"] + [key]
        np.random.shuffle(steps)
        filtered_data[key] = {"goal":step2goal[key], "step0":steps[0], "step1":steps[1], "step2":steps[2], "step3":steps[3], "label":steps.index(key)}
        
    for key in tqdm(list(filtered_data.keys())):
        if random.randint(0, 100) < 15:
            random_index_to_change = random.randint(0, 3)
            new_positive_candidate = [filtered_data[key]["step0"], filtered_data[key]["step1"], filtered_data[key]["step2"], filtered_data[key]["step3"]][random_index_to_change]
            filtered_data[key]["label"] = random_index_to_change

            new_goal = step2goal[new_positive_candidate]
            filtered_data[key]["goal"] = new_goal

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=3, ensure_ascii=False)
if __name__ == "__main__":
    main()