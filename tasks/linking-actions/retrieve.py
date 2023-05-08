import json
import argparse
from simcse import SimCSE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path of SimCSE checkpoint folder")
    parser.add_argument("--retrieve_for", type=str, help="Retrieve for training the reranker or reranking the retrieved matches")
    parser.add_argument("--gold_matches_train", type=str, help="Path of gold step and goal matches train")
    parser.add_argument("--gold_matches_test", type=str, help="Path of gold step and goal matches test")
    parser.add_argument("--titles_file", type=str, help="Path of titles file")
    parser.add_argument("--steps_file", type=str, help="Path of steps file")
    parser.add_argument("--step2goal", type=str, help="Path of step2goal file")
    args = parser.parse_args()

    if args.retrieve_for == "train":
        gold_matches_train = json.load(open(args.gold_matches_train, "r", encoding="utf-8"))
        gold_matches_test = json.load(open(args.gold_matches_test, "r", encoding="utf-8"))

        gold_matches_all = {}
        gold_matches_all.update(gold_matches_train)
        gold_matches_all.update(gold_matches_test)

        gold_matches_train_candidates = {}
        gold_matches_test_candidates = {}
        gold_matches_all_step_and_goal_scors = {}

        model = SimCSE(args.model_name_or_path)

        model.build_index(list(set(list(gold_matches_train.values()))))
        for key in gold_matches_train.keys():
            print(list(gold_matches_train.keys()).index(key))
                
            step = key
            gold_goal = gold_matches_train[key]

            results = model.search(step, top_k=30, threshold=0.01)
            retrieved_goals = [result[0] for result in results]
            retrieved_goals_similarity = [float(result[1]) for result in results]

            if gold_goal in retrieved_goals:
                gold_goal_index = retrieved_goals.index(gold_goal)
            else:
                gold_goal_index = -1

            gold_matches_train_candidates[key] = {"corresponding_goal":None, "gold_goal":gold_goal, "retrieved_goals":retrieved_goals, "retrieved_goals_similarity":retrieved_goals_similarity, "retrieved_goal_rank":gold_goal_index}

        model.build_index(list(set(list(gold_matches_test.values()))))
        for key in gold_matches_test.keys():
            print(list(gold_matches_test.keys()).index(key))
                
            step = key
            gold_goal = gold_matches_test[key]

            results = model.search(step, top_k=30, threshold=0.01)
            retrieved_goals = [result[0] for result in results]
            retrieved_goals_similarity = [float(result[1]) for result in results]

            if gold_goal in retrieved_goals:
                gold_goal_index = retrieved_goals.index(gold_goal)
            else:
                gold_goal_index = -1

            gold_matches_test_candidates[key] = {"corresponding_goal":None, "gold_goal":gold_goal, "retrieved_goals":retrieved_goals, "retrieved_goals_similarity":retrieved_goals_similarity, "retrieved_goal_rank":gold_goal_index}

        for key in gold_matches_all.keys():
            step = key
            goal = gold_matches_all[key]
            similarity = float(model.similarity(step, goal))

            gold_matches_all_step_and_goal_scors[step + " || " + goal] = similarity

        with open("gold.rerank.org.t30.train.json", "w", encoding="utf-8") as f:
            json.dump(gold_matches_train_candidates, f, ensure_ascii=False, indent=3)

        with open("gold.rerank.org.t30.test.json", "w", encoding="utf-8") as f:
            json.dump(gold_matches_test_candidates, f, ensure_ascii=False, indent=3)

        with open("gold.para.base.all.score.json", "w", encoding="utf-8") as f:
            json.dump(gold_matches_all_step_and_goal_scors, f, ensure_ascii=False, indent=3)

    elif args.retrieve_for == "rerank":
        titles = [title.replace("\n", "").replace("How to", "").replace("- wikiHow", "").strip() for title in open(args.titles_file, "r", encoding="utf-8").readlines()]
        titles = list(set([title for title in titles if "https" not in title]))

        steps = [step.replace("\n", "").strip() for step in open(args.steps_file, "r", encoding="utf-8").readlines()]
        steps = list(set([step for step in steps if "https" not in step]))

        step2goal = json.load(open(args.step2goal, "r", encoding="utf-8"))

        all_wikihow_step_t30goals = {}

        model = SimCSE(args.model_name_or_path)

        model.build_index(titles)
        for step in steps:
            try:
                print(steps.index(step))
                    
                results = model.search(step, top_k=30, threshold=0.01)
                retrieved_goals = [result[0] for result in results]
                retrieved_goals_similarity = [float(result[1]) for result in results]

                all_wikihow_step_t30goals[step] = {"corresponding_goal":step2goal[step], "gold_goal":None, "retrieved_goals":retrieved_goals, "retrieved_goals_similarity":retrieved_goals_similarity, "retrieved_description_similarity": [], "weighted_sum_similarity": [], "retrieved_goal_rank":-1}
            except:
                pass
        with open("all_wikihow_step_t30goals.json", "w", encoding="utf-8") as f:
            json.dump(all_wikihow_step_t30goals, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main()