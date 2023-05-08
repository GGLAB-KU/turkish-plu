import nlpturk
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import json
import faiss
from sentence_transformers import util
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path of model")
    parser.add_argument("--step2goal", type=str, help="Path of step2goal data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    def get_verb_token_indexes(text, tokenizer):
        doc = nlpturk(text)
        tokens = tokenizer.tokenize(text)
        verb_token_indexes = []
        for token in doc:
            if (token.pos == "VERB") or (token.pos == "NOUN") or (token.pos == "PROPN"):
                verb_tokens = tokenizer.tokenize(token.text)
                for token in verb_tokens:
                    if token in tokens:
                        verb_token_indexes.append(tokens.index(token))
        return verb_token_indexes

    def get_mean_of_the_verb_tokens(text, verb_token_indexes, model, tokenizer):
        tokenizer_result = tokenizer(text, return_attention_mask=True, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').to(device)
        input_ids = tokenizer_result.input_ids
        attention_mask = tokenizer_result.attention_mask

        model_result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeddings = model_result.last_hidden_state
        new_embeddings = torch.empty((1, 0, 768), dtype=torch.int64, device=device)
        new_attention_mask = torch.empty((1, 0), dtype=torch.int64, device=device)

        for ind in verb_token_indexes:
            new_embeddings = torch.cat((new_embeddings, token_embeddings[:, ind+1, :].reshape(1, 1, 768)), 1)
            new_attention_mask = torch.cat((new_attention_mask, attention_mask[:, ind+1].reshape((1, 1))), 1)

        return (new_embeddings.sum(axis=1) / new_attention_mask.sum(axis=-1).unsqueeze(-1)).cpu().detach().numpy()

    def find_steps_of_the_goal(goal, step2goal):
        goals_array = np.array(list(step2goal.values()))
        indexes = np.where(goals_array == goal)[0]
        return [list(step2goal.keys())[i] for i in indexes]

    model_name = args.model_path

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    step2goal = json.load(open(args.step2goal))
    steps = list(step2goal.keys())
    goals = list(set(list(step2goal.values())))

    embeddings = []
    for goal in tqdm(goals):
        emb = get_mean_of_the_verb_tokens(goal, get_verb_token_indexes(goal, tokenizer), model, tokenizer)
        embeddings.append(emb)
        torch.cuda.empty_cache()
    new_ndarray=np.array(embeddings)
    new_ndarray = new_ndarray.astype("float32")
    new_ndarray = new_ndarray.reshape((len(embeddings), 768))

    d = new_ndarray.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(new_ndarray)
    index.ntotal

    goal_inference = {}
    for goal in tqdm(goals):
        steps = find_steps_of_the_goal(goal, step2goal)
        goal_verb_token_indexes = get_verb_token_indexes(goal, tokenizer)
        goal_embedding = get_mean_of_the_verb_tokens(goal, goal_verb_token_indexes, model, tokenizer)

        k = 10
        D, I = index.search(goal_embedding, k)
        negative_candidates = []
        negative_candidates_similarities = []

        for i in range(k):
            if get_verb_token_indexes(goals[I[0][i]], tokenizer) != []:
                retrieved_goal_embedding = get_mean_of_the_verb_tokens(goals[I[0][i]], get_verb_token_indexes(goals[I[0][i]], tokenizer), model, tokenizer)
                cosine_scores = util.cos_sim(goal_embedding, retrieved_goal_embedding)
                if (goal != goals[I[0][i]]):
                    negative_candidates.append(goals[I[0][i]])
                    negative_candidates_similarities.append(float(cosine_scores[0][0]))
                torch.cuda.empty_cache()
            else:
                pass

        negative_candidates_similarities, negative_candidates = zip(*sorted(zip(negative_candidates_similarities, negative_candidates), reverse=True))
        negative_candidates = list(negative_candidates)[:3]
        negative_candidates_similarities = list(negative_candidates_similarities)[:3]

        goal_inference[goal] = {"negative_candidates":negative_candidates, "steps":steps, "negative_candidates_similarities":negative_candidates_similarities}
        torch.cuda.empty_cache()
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(goal_inference, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main()