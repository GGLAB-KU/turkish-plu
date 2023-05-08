import json
import faiss
import argparse
from simcse import SimCSE
from sentence_transformers import SentenceTransformer, util
from beautifultable import BeautifulTable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path of SimCSE checkpoint folder")
    parser.add_argument("--titles_file", type=str, help="Path of titles file")
    parser.add_argument("--test_data", type=str, help="Path of test data")
    args = parser.parse_args()

    titles = [title.replace("\n", "").replace("How to", "").replace("- wikiHow", "").replace("?", "").strip() for title in open(args.titles_file, "r", encoding="utf-8").readlines()]
    titles = list(set([title for title in titles if "https" not in title]))

    step_goal_matches_test = json.load(open(args.test_data, "r", encoding="utf-8"))
    steps = list(step_goal_matches_test.keys())

    if "simcse" in args.model_name_or_path.lower():
        model = SimCSE(args.model_name_or_path)
        model.build_index(titles)

        retrieval_dict = {}
        for key in step_goal_matches_test.keys():
            print(list(step_goal_matches_test.keys()).index(key))
                
            step = key
            gold_goal = step_goal_matches_test[key]

            results = model.search(step, top_k=30, threshold=0.01)
            retrieved_goals = [result[0] for result in results]
            retrieved_goals_similarity = [float(result[1]) for result in results]

            if gold_goal in retrieved_goals:
                gold_goal_index = retrieved_goals.index(gold_goal)
            else:
                gold_goal_index = -1

            retrieval_dict[key] = {"corresponding_goal":None, "gold_goal":gold_goal, "retrieved_goals":retrieved_goals, "retrieved_goals_similarity":retrieved_goals_similarity, "retrieved_goal_rank":gold_goal_index}
        
        top1 = 0
        top10 = 0
        top30 = 0
        table = BeautifulTable()
        table.columns.header = ["R@1", "R@10", "R@30"]
        table.rows.header = [args.model_name_or_path]

        for key in retrieval_dict.keys():
            gold_goal_index = retrieval_dict[key]["retrieved_goal_rank"]
            if gold_goal_index == 0:
                top1+=1
            elif 1 <= gold_goal_index < 10:
                top10+=1
            elif 10 <= gold_goal_index < 30:
                top30+=1

        top30+=top1+top10
        top10+=top1

        table.rows.append([top1/len(step_goal_matches_test.keys()), top10/len(step_goal_matches_test.keys()), top30/len(step_goal_matches_test.keys())])
        print(table)

    else:
        model = SentenceTransformer(args.model_name_or_path)

        title_embeddings = model.encode(titles)
        print("Title embeddings completed.")

        d = title_embeddings.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(title_embeddings)

        retrieval_dict = {}
        for key in step_goal_matches_test.keys():
            print(list(step_goal_matches_test.keys()).index(key))
            
            step = key
            gold_goal = step_goal_matches_test[key]
            step_embedding = model.encode(step).reshape((1, title_embeddings.shape[1]))

            k = 30
            D, I = index.search(step_embedding, k)

            retrieved_goals = []
            retrieved_goals_similarity = []
            for i in range(len(I[0])):
                retrieved_goal_embedding = model.encode(titles[I[0][i]])
                cosine_scores = util.cos_sim(step_embedding, retrieved_goal_embedding)

                retrieved_goals.append(titles[I[0][i]])
                retrieved_goals_similarity.append(float(cosine_scores[0][0]))

            if gold_goal in retrieved_goals:
                gold_goal_index = retrieved_goals.index(gold_goal)
            else:
                gold_goal_index = -1

            retrieval_dict[key] = {"corresponding_goal":None, "gold_goal":gold_goal, "retrieved_goals":retrieved_goals, "retrieved_goals_similarity":retrieved_goals_similarity, "retrieved_goal_rank":gold_goal_index}

        top1 = 0
        top10 = 0
        top30 = 0
        table = BeautifulTable()
        table.columns.header = ["R@1", "R@10", "R@30"]
        table.rows.header = [args.model_name_or_path]

        for key in retrieval_dict.keys():
            gold_goal_index = retrieval_dict[key]["retrieved_goal_rank"]
            if gold_goal_index == 0:
                top1+=1
            elif 1 <= gold_goal_index < 10:
                top10+=1
            elif 10 <= gold_goal_index < 30:
                top30+=1

        top30+=top1+top10
        top10+=top1

        table.rows.append([top1/len(step_goal_matches_test.keys()), top10/len(step_goal_matches_test.keys()), top30/len(step_goal_matches_test.keys())])
        print(table)

if __name__ == "__main__":
    main()