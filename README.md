# FlexSQL
Official code for "**FlexSQL: A Lightweight and Flexible Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction**"
![Method-Inference-final](https://github.com/user-attachments/assets/16d9c03d-926d-40d3-aa98-0e7bb4a337f0)


## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#hammer_and_wrench-getting-started)
- :rocket: [Running FlexSQL](#rocket-running-tinythinker)
  - :dart: [Schema Retriever](#dart-train-tinythinker)
  - :dart: [SQL Generator](#dart-train-tinythinker)
    - :memo: [Prepare Training Dataset](#prepare-training-dataset)
    - :memo: [Train SQL Generator](#memo-train-SQL-generator)
    - :hourglass_flowing_sand: [Inference](#hourglass_flowing_sand-inference)
- :page_facing_up: [License](#page_facing_up-license)

## :hammer_and_wrench: Getting Started

To set up the environment, use the following commands:

```
git clone https://github.com/shengminp/FlexSQL.git
cd FlexSQL
conda env create -f environment.yml
```

After installation, your project directory structure should look like this:
    
    .
    ├──schema_retriever
    ├──sql_generator
    └── README.md

## :rocket: Running FlexSQL

### :dart: Schema Retriever
### :dart: SQL Generator
The directory structure of SQL generator is as follow:

    .
    └──sql_generator
        ├── datasets
        │    ├── bird
        |    |    |── train
        |    |    |    ├──train_databases
        |    |    |    ├──sft
        |    |    |    ├──rft
        |    |    |    ├──BIRD-train-more-schema.json
        |    |    |    ...
        |    |    └── dev_20240627
        |    |        ├──dev_databases
        |    |        ├──retrieved
        |    |        ...
        │    ├── spider
        |    │    ...
        │    |    └── same as aboe
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

#### :memo: Train SQL Generator
**Phase-1: Supervised Fine-tuning**

In this phase, a Qwen2.5-Coder model is fine-tuned. Run the following command:
```
python finetune.py \
    --base_model $model_name \
    --data_name $data_name \
    --training_phase sft \
```
- **$model_name:** Model name from Huggingface ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$data_name:** Dataset name ```(bird, spider)```.

**Phase-2: Reinforcement Fine-tuning**

In this phase, we fine tune the model with self-generated data. Run the following command:
```
python dpo.py \
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

### :hourglass_flowing_sand: Inference
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
- **$base_name:** Model name ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$data_name:** Dataset name ```(bird, spider)```.
- **$checkpoint_dir:** Path of checkpoint.
- **$schema_file:** Path of schema file
- **$generation_mode:** Generation mode ```(generate, revise)```.
- **$generation_type:** Generation type ```(greedy, random)```.
- **$generation_file:** Generation file ```(test, augment)```.
- **$augment_times:** The times to augment the training data```.
- **$generate_times:** The times to generate new data```.

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
- **$base_name:** Model name ```(Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct)```.
- **$data_name:** Dataset name ```(bird, spider)```.
- **$checkpoint_dir:** Path of checkpoint.
- **$schema_file:** Path of schema file
- **$generation_mode:** Generation mode ```(generate, revise)```.
- **$generation_type:** Generation type ```(greedy, random)```.
- **$generation_file:** Generation file ```(test, augment)```.
- **$revise_times:** The times to revise the generated SQL```.

## :page_facing_up: License

This project is licensed under the [MIT](LICENSE) © Shengmin Piao & Jieun Lee
