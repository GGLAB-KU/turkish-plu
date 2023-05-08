from filters import tf_idf_filter, lexical_overlap_filter, length_filter, similarity_filter_for_goal_inference
import argparse 
from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nlpturk
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from simcse import SimCSE

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('turkish'))

def main():
    def find_steps_of_the_goal(goal, step2goal):
        goals_array = np.array(list(step2goal.values()))
        indexes = np.where(goals_array == goal)[0]
        return [list(step2goal.keys())[i] for i in indexes]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, help="Path of tokenizer")
    parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
    parser.add_argument("--goal_inference_path", type=str, help="Path of goal inference data")
    parser.add_argument("--step2goal_path", type=str, help="Path of goal inference data")
    parser.add_argument("--wikihow_path", type=str, help="Path of goal inference data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    goal_inference = json.load(open(args.goal_inference_path))
    step2goal = json.load(open(args.step2goal_path))
    wikihow = json.load(open(args.wikihow_path))

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    simcse = SimCSE(args.simcse_path)

    goal_inference_data = {}
    for key in goal_inference.keys():
        for step in goal_inference[key]["steps"]:
            goals = goal_inference[key]["negative_candidates"] + [key]
            random.shuffle(goals)
            goal_inference_data[len(goal_inference_data)] = {"step":step, "goal0":goals[0], "goal1":goals[1], "goal2":goals[2], "goal3":goals[3], "label":goals.index(key)}

    keys_to_delete = []

    for key in tqdm(list(goal_inference_data.keys())):
        try:
            length = length_filter(goal_inference_data[key]["step"], tokenizer, threshold=5)

            if length:
                positive_candidate = [goal_inference_data[key]["goal0"], goal_inference_data[key]["goal1"], goal_inference_data[key]["goal2"], goal_inference_data[key]["goal3"]][goal_inference_data[key]["label"]] 
                negative_candidates = [goal_inference_data[key]["goal0"], goal_inference_data[key]["goal1"], goal_inference_data[key]["goal2"], goal_inference_data[key]["goal3"]]
                negative_candidates.remove(positive_candidate)

                lexical_overlap = lexical_overlap_filter(positive_candidate, negative_candidates)
                if lexical_overlap:

                    tf_idf = tf_idf_filter(goal_inference_data[key]["step"], step2goal, wikihow)
                    if tf_idf:

                        similarity = similarity_filter_for_goal_inference(goal_inference_data[key]["step"], negative_candidates, wikihow, simcse)
                        if similarity:
                            pass
                        
                        else:
                            keys_to_delete.append(key)
                    else:
                        keys_to_delete.append(key)
                else:
                    keys_to_delete.append(key)
            else:
                keys_to_delete.append(key)

            torch.cuda.empty_cache()
        except:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del goal_inference_data[key]

    for key in tqdm(list(goal_inference_data.keys())):
        if random.randint(0, 100) < 15:
            random_index_to_change = random.randint(0, 3)
            new_positive_candidate = [goal_inference_data[key]["goal0"], goal_inference_data[key]["goal1"], goal_inference_data[key]["goal2"], goal_inference_data[key]["goal3"]][random_index_to_change]
            goal_inference_data[key]["label"] = random_index_to_change

            steps_of_the_new_positive_candidate = find_steps_of_the_goal(new_positive_candidate, step2goal)
            new_step = steps_of_the_new_positive_candidate[random.randint(0, len(steps_of_the_new_positive_candidate)-1)]
            goal_inference_data[key]["step"] = new_step

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(goal_inference_data, f, indent=3, ensure_ascii=False)
if __name__ == "__main__":
    main()