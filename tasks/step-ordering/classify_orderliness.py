import json
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, help="Path of model")
  parser.add_argument("--data", type=str)
  parser.add_argument('--output_file', type=str)
  args = parser.parse_args()

  tokenizer= AutoTokenizer.from_pretrained(args.model_name)
  model= AutoModelForSequenceClassification.from_pretrained(args.model_name)
  classification_pipeline=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

  wikihow = json.load(open(args.data, "r", encoding="utf-8"))
  batch_size = 128
  batched_wikihow = [wikihow[i*batch_size:(i+1)*batch_size] for i in range(int(len(wikihow)/batch_size)+1)]
  for batched_element in tqdm(batched_wikihow):
    texts = [" ".join(element["caption"]).strip() for element in batched_element]
    is_ordereds = classification_pipeline(texts, truncation=True)
    is_ordereds = [1 if is_ordered["label"] == "ORDERED" else 0 for is_ordered in is_ordereds]
    for element in batched_element:
      element["is_ordered"] = is_ordereds[batched_element.index(element)]
      wikihow[wikihow.index(element)] = element

  with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(wikihow, f, indent=3, ensure_ascii=False)
if __name__ == "__main__":
    main()