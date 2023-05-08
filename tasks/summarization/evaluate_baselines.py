from datasets import Dataset, DatasetDict, load_dataset
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, pipeline
import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Path of model")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    dataset = load_dataset("ardauzunoglu/tr-wikihow-summ")
    summ_pipeline = pipeline(task="summarization", model=args.model_name, max_length=512)

    task_evaluator = evaluator("summarization")
    eval_results = task_evaluator.compute(
        model_or_pipeline=summ_pipeline,
        data=dataset["test"],
        input_column="text",
        label_column="summary"       
    )
    print(args.model_name)
    print(eval_results)

if __name__ == "__main__":
    main()