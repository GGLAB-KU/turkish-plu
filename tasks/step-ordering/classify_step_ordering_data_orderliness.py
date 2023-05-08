import json
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, help="Path of model")
  parser.add_argument("--data", type=str)
  args = parser.parse_args()

  tokenizer= AutoTokenizer.from_pretrained(args.model_name)
  model= AutoModelForSequenceClassification.from_pretrained(args.model_name)
  classification_pipeline=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

  data = pd.read_csv(args.data)
  step1_list = data.sent1.tolist()
  step2_list = data.sent2.tolist()
  text_bodies = [" ".join([step1, step2_list[step1_list.index(step1)]]).strip() for step1 in step1_list]
  is_ordered_list = []

  batch_size = 512
  batched_data = [text_bodies[i*batch_size:(i+1)*batch_size] for i in range(int(len(text_bodies)/batch_size)+1)]
  for batched_element in tqdm(batched_data):
    is_ordereds = classification_pipeline(batched_element, truncation=True)
    is_ordereds = [1 if is_ordered["label"] == "ORDERED" else 0 for is_ordered in is_ordereds]
    is_ordered_list += is_ordereds

  data["is_ordered"] = is_ordered_list
  data = data[data['is_ordered'] == 1]
  #data.drop(["is_ordered"], axis=1, inplace=True)
  
  train_df = data.iloc[:len(data)-20000]
  val_df = data.iloc[len(data)-20000:len(data)-10000]
  test_df = data.iloc[len(data)-10000:]

  train_df.to_csv("step-ordering-train.csv")
  val_df.to_csv("step-ordering-val.csv")
  test_df.to_csv("step-ordering-test.csv")
  data.to_csv("step-ordering.csv")

if __name__ == "__main__":
    main()