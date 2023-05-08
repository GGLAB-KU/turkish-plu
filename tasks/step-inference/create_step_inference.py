import nlpturk
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import json
import faiss
from sentence_transformers import util
import argparse
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2goal", type=str, help="Path of step2goal data")
    parser.add_argument("--step_embeddings_path", type=str, help="Path of step_embeddings")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    def find_steps_of_the_goal(goal, step2goal):
        goals_array = np.array(list(step2goal.values()))
        indexes = np.where(goals_array == goal)[0]
        return [list(step2goal.keys())[i] for i in indexes]

    with open(args.step_embeddings_path, 'rb') as pkl:
        step_embeddings = pickle.load(pkl)
    
    step2goal = json.load(open(args.step2goal, "r", encoding="utf-8"))
    steps = list(step2goal.keys())

    new_ndarray = np.array(step_embeddings)
    new_ndarray = new_ndarray.astype("float32")

    d = new_ndarray.shape[1]
    nlist = 50
    
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(new_ndarray)
    index.ntotal

    step_inference = {}
    for step in tqdm(steps):
        step_embedding = step_embeddings[steps.index(step)].reshape(1, 768)

        goal = step2goal[step]
        goal_steps = find_steps_of_the_goal(goal, step2goal)

        k = 7
        D, I = index.search(step_embedding, k)
        negative_candidates = []
        negative_candidates_similarities = [] 

        for i in range(k):
            retrieved_step_embedding = step_embeddings[steps.index(steps[I[0][i]])].reshape(1, 768)
            cosine_scores = util.cos_sim(step_embedding, retrieved_step_embedding)
            if (step != steps[I[0][i]]) and (steps[I[0][i]] not in goal_steps):
                negative_candidates.append(steps[I[0][i]])
                negative_candidates_similarities.append(float(cosine_scores[0][0]))

        try:
            negative_candidates_similarities, negative_candidates = zip(*sorted(zip(negative_candidates_similarities, negative_candidates), reverse=True))
            negative_candidates = list(negative_candidates)[:3]
            negative_candidates_similarities = list(negative_candidates_similarities)[:3]

            step_inference[step] = {"negative_candidates":negative_candidates, "negative_candidates_similarities":negative_candidates_similarities}
        except:
            pass
    
        del step_embedding
        torch.cuda.empty_cache()
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(step_inference, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main()