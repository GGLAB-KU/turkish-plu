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

        verb_token_indexes = [token_index for token_index in verb_token_indexes if token_index < 511]

        for ind in verb_token_indexes:
            new_embeddings = torch.cat((new_embeddings, token_embeddings[:, ind+1, :].reshape(1, 1, 768)), 1)
            new_attention_mask = torch.cat((new_attention_mask, attention_mask[:, ind+1].reshape((1, 1))), 1)

        return (new_embeddings.sum(axis=1) / new_attention_mask.sum(axis=-1).unsqueeze(-1)).cpu().detach().numpy()

    model_name = args.model_path

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    step2goal = json.load(open(args.step2goal))
    steps = list(step2goal.keys())

    embeddings = []
    for step in tqdm(steps):
        embedding = get_mean_of_the_verb_tokens(step, get_verb_token_indexes(step, tokenizer), model, tokenizer)
        embeddings.append(embedding)
        del embedding
        torch.cuda.empty_cache()

    new_ndarray = np.array(embeddings)
    new_ndarray = new_ndarray.astype("float32")
    new_ndarray = new_ndarray.reshape((len(embeddings), 768))

    with open(args.save_path, 'wb') as pkl:
        pickle.dump(new_ndarray, pkl)

if __name__ == "__main__":
    main()