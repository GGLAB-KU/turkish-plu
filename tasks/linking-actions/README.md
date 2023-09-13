# Linking Actions

This folder contains information regarding the linking actions task.

### Requirements

Implemented with Python 3.7, using the following libraries:

```
tqdm
torch==1.7.0
transformers
numpy
```

### Data

Find the necessary data files in the ```data``` folder.

Linking actions task follows the data format formulated by [Show Me More Details: Discovering Hierarchies of Procedures from Semi-structured Web Data](https://github.com/shuyanzhou/wikihow_hierarchy).

```data/wikihow.json```: wikiHow dump in a single .json file.
```data/hyperlinks.json```: Ground-truth data for goal - step matches, obtained from wikiHow.
```data/step2goal.json```: Each step from the wikiHow dumped, matched with the goal of its tutorial.

Please refer to [Show Me More Details: Discovering Hierarchies of Procedures from Semi-structured Web Data](https://github.com/shuyanzhou/wikihow_hierarchy) to obtain data files used in training and testing.

### Training and Testing

Training and testing scripts are provided.

For training, simply run the following command:

```
python3 reranking/train.py \
  --train_null \
  --add_goal \
  --use_para_score \
  --model_name MODEL_NAME \
  --context_length 1 \
  --train_file ./data/gold.rerank.org.t30.train.json \
  --dev_file ./data/gold.rerank.org.t30.dev.json \
  --gold_step_goal_para_score ./data/gold.para.base.all.score \
  --save_path OUTPUT_PATH \
  --neg_num 29 --bs 1 \
  --mega_bs 4 --val_bs 1 \
  --min_save_ep 0 \
  --epochs 5
```

For testing, simply run the following command:

```
python3 reranking/inference.py \
  --model_path MODEL_PATH \
  --test_path ./data/all_wikihow_step_t30goals.json \
  --save_path ./data/all.result \
  --no_label
```