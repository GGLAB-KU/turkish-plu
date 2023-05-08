import json
import numpy as np
import faiss
from tqdm import tqdm
import argparse
import pickle
from filters import tf_idf_filter, length_filter, similarity_filter_for_next_event_prediction
from transformers import BertTokenizerFast
import pandas as pd
from simcse import SimCSE
import warnings
warnings.filterwarnings('ignore')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, help="Path of model")
  parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
  parser.add_argument("--wikihow", type=str, help="Path of step2goal data")
  parser.add_argument("--step2goal", type=str, help="Path of step2goal data")
  parser.add_argument("--step_embeddings_path", type=str, help="Path of step_embeddings")
  parser.add_argument("--save_path", type=str, help="Path of step2goal data")
  args = parser.parse_args()

  tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
  simcse = SimCSE(args.simcse_path)

  with open(args.step_embeddings_path, 'rb') as pkl:
    step_embeddings = pickle.load(pkl)

  wikihow = json.load(open(args.wikihow, "r", encoding="utf-8"))  
  step2goal = json.load(open(args.step2goal, "r", encoding="utf-8"))
  steps = list(step2goal.keys())

  ndarray = np.array(step_embeddings)
  ndarray = ndarray.astype("float32")

  d = ndarray.shape[1]

  index = faiss.IndexFlatL2(d)
  index.add(ndarray)

  dataset_dict = {}
  ordered_wikihow = [element for element in wikihow if element["is_ordered"] == 1]
  for element in tqdm(ordered_wikihow):
    caption = element["caption"]
    for step in caption:
      step_ind = caption.index(step)
      if (step_ind != (len(caption) - 1)):
        try:
          if (length_filter(step, tokenizer, threshold=10)) and (tf_idf_filter(step, caption)):
            positive_candidate = caption[step_ind+1]
            step_embedding = step_embeddings[steps.index(positive_candidate)].reshape(1, 768)

            k = 4
            D, I = index.search(step_embedding, k)

            negative_candidates = [steps[I[0][i]] for i in range(k)]
            if positive_candidate in negative_candidates:
              negative_candidates.remove(positive_candidate)
            negative_candidates = negative_candidates[:3]
            if similarity_filter_for_next_event_prediction(positive_candidate, negative_candidates, simcse):
              ending_candidates =(negative_candidates + [positive_candidate])
              np.random.shuffle(ending_candidates)
              dataset_dict[len(dataset_dict)] = ["xxx", len(dataset_dict), "xxx", step2goal[step], step, "xxx", ending_candidates[0], ending_candidates[1], ending_candidates[2], ending_candidates[3], ending_candidates.index(positive_candidate)] 
        except:
          pass
  dataset = pd.DataFrame.from_dict(dataset_dict, orient="index", columns=["video-id", "fold-ind", "startphrase", "sent1", "sent2", "gold-source", "ending0", "ending1", "ending2", "ending3", "label"])
  dataset = dataset.sample(frac=1)
  dataset.to_csv(args.save_path)
if __name__ == "__main__":
    main()