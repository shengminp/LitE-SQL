# FlexSQL
Official code for "**FlexSQL: A Lightweight and Flexible Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction**"
![Method-Inference-final](https://github.com/user-attachments/assets/16d9c03d-926d-40d3-aa98-0e7bb4a337f0)


## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#hammer_and_wrench-getting-started)
- :rocket: [Running FlexSQL](#rocket-running-tinythinker)
  - :mag: [Schema Retriever](#mag-schema-retriever)
  - :dart: [SQL Generator](#dart-SQL-generator)
    - :memo: [Prepare Training Dataset](#memo-prepare-training-dataset)
    - :gear: [Train SQL Generator](#gear-train-SQL-generator)
    - :hourglass_flowing_sand: [Inference](#hourglass_flowing_sand-inference)
- :page_facing_up: [License](#page_facing_up-license)

## :hammer_and_wrench: Getting Started

To set up the environment, use the following commands:

```
git clone https://github.com/shengminp/LitE-SQL.git
cd LitE-SQL
conda env create -f environment.yml
```

After installation, your project directory structure should look like this:
    
    .
    ├── datasets
    ├── schema_retriever
    ├── sql_generator
    ├── environment.yml
    └── README.md

Specifically, the directory structure of datasets is as follow:

    .
    └── datasets
         ├── bird
         |    |── train
         |    |    ├── train_databases
         |    |    ├── sft
         |    |    ├── rft
         |    |    ├── BIRD-train-more-schema.json
         |    |    ├── train_tables.json
         |    |    ├── train.json
         |    |    └── train_gold.sql
         |    └── dev_20240627
         |         ├── dev_databases
         |         ├── retrieved
         |         ├── dev.json
         |         ├── dev_tables.json
         |         ├── dev_tied_append.json
         |         └── dev.sql
         └── spider
              |── train
              |    ├── database
              |    ├── sft
              |    ├── rft
              |    ├── SPIDER-train-more-schema.json
              |    ├── train_gold.sql
              |    ├── train_others.json
              |    ├── tables.json
              |    └── train_spider.sql
              |── dev
              |    ├── database
              |    ├── retrieved
              |    ├── dev.json
              |    ├── tables.json
              |    └── dev_gold.sql
              └── test
                   ├── dev_database
                   ├── retrieved
                   ├── test.json
                   ├── test_tables.json
                   └── test_gold.sql

## :rocket: Running LitE-SQL

### :mag: Schema Retriever
The directory structure of schema retriever is as follow:
### :dart: SQL Generator
The directory structure of SQL generator is as follow:

    .
    └──sql_generator
        ├── datasets
        │    └── Prepare.ipynb
        ├── models
        ├── results
        └── scripts
             ├── utils
             │    ├── spider_tool
             |    |     ├── __init__.py
             |    |     ├── evaluation.py
             |    |     ├── exec_eval.py
             |    |     ├── parse.py
             |    |     └── process_sql.py
             │    ├── __init__.py
             │    ├── config.py
             │    ├── data.py
             │    ├── metric.py
             |    └── trainer.py
             ├── rft.py
             ├── sft.py
             └── generate.py



#### :memo: Prepare Training Dataset
1. Download the dataset from its official site and place it under ```./datasets```.
2. Run ```./datasets/Prepare.ipynb``` to prepare dataset for suervised fine-tuning.

#### :gear: Train SQL Generator
**Phase-1: Supervised Fine-tuning**

In this phase, a Qwen2.5-Coder model is fine-tuned. Run the following command:
```
python sft.py \
    --base_model $model_name \
    --data_name $data_name \
    --training_phase sft \
```
- **$model_name:** Model name from Huggingface ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$data_name:** Dataset name ```(bird, spider)```.

**Phase-2: Reinforcement Fine-tuning**

In this phase, we fine tune the model with self-generated data. Run the following command:
```
python rft.py \
    --base_model $base_model \
    --tune_model $checkpoint \
    --ref_model $checkpoint \
    --data_name $data_name \
    --training_phase rft \
    --rft_iter $rft_iter \
```
- **$base_model:** Model name from Huggingface ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$tune_model:** Path of checkpoint from sft pahse.
- **$ref_model:** Same as $tune_model.
- **$data_name:** Dataset name ```(bird, spider)```.

#### :hourglass_flowing_sand: Inference
Use the following command to do inferences after sft phase:
```
python generate.py \
    --base_model $base_model \
    --data_name $data_name \
    --training_phase sft \
    --checkpoint_dir $checkpoint_path\
    --schema_file $schema_file_path\
    --generation_mode $generation_mode \
    --generation_type $generation_type \
    --generation_file $generation_file
```
- **$base_name:** Model name ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$data_name:** Dataset name ```(bird, spider)```.
- **$checkpoint_dir:** Path of checkpoint.
- **$schema_file:** Path of schema file
- **$generation_mode:** Generation mode ```(generate, revise)```.
- **$generation_type:** Generation type ```(greedy, random)```.
- **$generation_file:** Generation file ```(test, augment)```.

Use the following command to generate data for rft phase:
```
python generate.py \
    --base_model $base_model \
    --data_name $data_name \
    --training_phase sft \
    --checkpoint_dir $checkpoint_path\
    --schema_file $schema_file_path\
    --generation_mode $generation_mode \
    --generation_type $generation_type \
    --generation_file $generation_file \
    --augment_times $augment_times \
    --generate_times $generate_times \
    --rft_iter 1 \
```
- **$augment_times:** The times to augment the training data.
- **$generate_times:** The times to generate new data.

Use the following command to do inferences after rft phase:
```
python generate.py \
    --base_model $base_model \
    --data_name $data_name \
    --training_phase rft \
    --checkpoint_dir $checkpoint_path\
    --schema_file $schema_file_path\
    --generation_mode $generation_mode \
    --generation_type $generation_type \
    --generation_file $generation_file \
    --revise_times $revise_times \
    --rft_iter 1 \
```
- **$revise_times:** The times to revise the generated SQL.

## :page_facing_up: License

This project is licensed under the [MIT](LICENSE) © Shengmin Piao & Jieun Lee
