# Out-of-Domain Evaluation

This folder contains information regarding the out-of-domain evaluation of language models across different PLU tasks, as described in Appendix F.

### Data

OOD evaluation is performed with only next event prediction and step inference tasks. Find the test splits of next event prediction and step inference datasets under the ```data``` folder.

### How to Run

Simply use the following command:

```
python3 test_multiple_choice.py \
  --model_name_or_path MODEL_NAME \
  --tokenizer_name TOKENIZER_OF_THE_MODEL \
  --per_device_test_batch_size 8 \
  --test_file data/TASK-NAME-test.csv \
```

where the model and the data must be for separate tasks.