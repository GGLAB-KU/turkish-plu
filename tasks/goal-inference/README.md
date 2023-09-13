# Goal Inference

This folder contains information regarding the goal inference task.

### Requirements

Implemented with Python 3.7, using the following libraries:

```
transformers==4.27.0
evaluate==0.4.0
accelerate==0.15.0
torch==1.13.1
```

### Data

Find the train, validation, and test splits in the ```data``` folder. 

Goal inference data follows the format of the SWAG dataset, as can be seen in the example below:

```
video-id,fold-ind,startphrase,sent1,sent2,gold-source,ending0,ending1,ending2,ending3,label
xxx,259773,xxx,xxx,Modemi koyacağın yere karar ver.,xxx,Modem Kurmak,Anten Ayarlamak,VPN Kurmak,VPN Kullanmak,0
```

### Training and Testing

Training and testing scripts are provided.

For training, Hugging Face's Accelerate library is used. Therefore, run the following command for training:

```
accelerate config

accelerate test

accelerate launch train.py \
  --model_name_or_path MODEL_NAME \
  --train_file data/goal-inference-train.csv \
  --validation_file data/goal-inference-val.csv \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --output_dir OUTPUT_MODEL_NAME \
```

For testing, simply run the following command:

```
python3 test.py \
  --model_name_or_path MODEL_NAME \
  --per_device_test_batch_size 8 \
  --test_file data/goal-inference-test.csv \
```