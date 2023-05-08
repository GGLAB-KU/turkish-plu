from filters import tf_idf_filter, lexical_overlap_filter, length_filter, similarity_filter_for_next_event_prediction
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, help="Path of tokenizer")
    parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
    parser.add_argument("--next_event_prediction_path", type=str, help="Path of goal inference data")
    parser.add_argument("--step2goal_path", type=str, help="Path of goal inference data")
    parser.add_argument("--wikihow_path", type=str, help="Path of goal inference data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    next_event_prediction = pd.read_csv(args.next_event_prediction_path)
    step2goal = json.load(open(args.step2goal_path))
    wikihow = json.load(open(args.wikihow_path))

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    simcse = SimCSE(args.simcse_path)

    previous_steps = next_event_prediction["sent1"].tolist()
    ending0s = next_event_prediction["ending0"].tolist()
    ending1s = next_event_prediction["ending1"].tolist()
    ending2s = next_event_prediction["ending2"].tolist()
    ending3s = next_event_prediction["ending3"].tolist()
    labels = next_event_prediction["label"].tolist()

    next_event_prediction_data = {}
    for step in tqdm(previous_steps):
        next_event_prediction_data[len(next_event_prediction_data)] = {"step":step, "ending0":ending0s[previous_steps.index(step)], "ending1":ending1s[previous_steps.index(step)], "ending2":ending2s[previous_steps.index(step)], "ending3":ending3s[previous_steps.index(step)], "label":labels[previous_steps.index(step)]}

    keys_to_delete = []
    for key in tqdm(list(next_event_prediction_data.keys())):
        try:
            length = length_filter(next_event_prediction_data[key]["step"], tokenizer, threshold=5)

            if length:
                positive_candidate = [next_event_prediction_data[key]["ending0"], next_event_prediction_data[key]["ending1"], next_event_prediction_data[key]["ending2"], next_event_prediction_data[key]["ending3"]][next_event_prediction_data[key]["label"]] 
                negative_candidates = [next_event_prediction_data[key]["ending0"], next_event_prediction_data[key]["ending1"], next_event_prediction_data[key]["ending2"], next_event_prediction_data[key]["ending3"]]
                negative_candidates.remove(positive_candidate)

                lexical_overlap = lexical_overlap_filter(positive_candidate, negative_candidates)
                if lexical_overlap:

                    tf_idf = tf_idf_filter(next_event_prediction_data[key]["step"], step2goal, wikihow)
                    if tf_idf:

                        similarity = similarity_filter_for_next_event_prediction(positive_candidate, negative_candidates, simcse)
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
        del next_event_prediction_data[key]

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(next_event_prediction_data, f, indent=3, ensure_ascii=False)
if __name__ == "__main__":
    main()