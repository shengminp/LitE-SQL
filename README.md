# FlexSQL
Official code for "**FlexSQL: A Lightweight and Flexible Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction**"
![Method-Inference-final](https://github.com/user-attachments/assets/16d9c03d-926d-40d3-aa98-0e7bb4a337f0)


## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#hammer_and_wrench-getting-started)
- :rocket: [Running FlexSQL](#rocket-running-tinythinker)
  - :memo: [Prepare Training Dataset](#memo-prompt-engineering)
  - :dart: [Train FlexSQL](#dart-train-tinythinker)
  - :hourglass_flowing_sand: [Inference](#hourglass_flowing_sand-inference)
- :page_facing_up: [License](#page_facing_up-license)

## :hammer_and_wrench: Getting Started

To set up the environment, use the following commands:

```
git clone https://github.com/shengminp/TinyThinker.git
cd TinyThinker
conda env create -f environment.yml
```

After installation, your project directory structure should look like this:
    
    .
    ├──sql_generator
    |    ├── datasets
    |    │    ├── bird
    |    |    |    |── train
    |    |    |    |    ├──train_databases
    |    |    |    |    ├──sft
    |    |    |    |    ├──rft
    |    |    |    |    ├──BIRD-train-more-schema.json
    |    |    |    |    ...
    |    |    |    └── dev_20240627
    |    |    |        ├──dev_databases
    |    |    |        ├──retrieved
    |    |    |        ...
    |    │    ├── spider
    |    |    │    ...
    |    │    |    └── same as aboe
    |    │    └── Prepare.ipynb
    |    ├── models
    |    ├── results
    |    └── scripts
    |         ├── utils
    |         │    ├── spider_tool
    |         |    |     ├── __init__.py
    |         |    |     ├── evaluation.py
    |         |    |     ├── exec_eval.py
    |         |    |     ├── parse.py
    |         |    |     └── process_sql.py
    |         │    ├── __init__.py
    |         │    ├── config.py
    |         │    ├── data.py
    |         │    ├── metric.py
    |         |    └── trainer.py
    |         ├── rft.py
    |         ├── sft.py
    |         └── generate.py
    └── README.md

## :rocket: Running Train FlexSQL

### :memo: Prepare Training Dataset
1. Download the dataset from its official site and place it under ```./datasets/DATASET/original```.
2. Run ```./datasets/Prompt Engineering.ipynb``` to prepare prompts for the dataset.
3. Execute ```./datasets/openai_request.py``` to generate responses from OpenAI:
   - If the generated responses contain errors, repeat steps 2 and 3 as needed (see details in ```Prompt Engineering.ipynb```).
4. Once completed, the prepared dataset will be located at ```./datasets/DATASET/final```.
5. Use ```Prepare Ablation.ipynb``` to prepare the ablation study dataset, with results saved at ```./datasets/DATASET/final```.
   
### :dart: Train FlexSQL
**Phase-1: Supervised Fine-tuning**

In this phase, a T5 model is fine-tuned using a three-stage process. Run the following command:
```
python finetune.py \
    --base_model $model_name \
    --data_name $data_name \
    --training_type sft \
    --stage_type $stage_type \
    --per_gpu_batch_size $batch_size \
    --num_train_epochs $num_epochs \
    --learning_rate $lr \
    --interval $interval
```
- **$model_name:** Model name from Huggingface ```(google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b)```.
- **$data_name:** Dataset name ```(csqa, obqa, strategyqa)```.
- **$stage_type:** Training stage ```(summarize, recall_summarize, analyze_summarize, recall_analyze_summarize)```.
- **$interval:** Interval value between stages.

**Phase-2: Reinforcement Fine-tuning**

In this phase, we refine the reasoning through self-generated data using DPO. Run:
```
python dpo.py \
    --base_model $base_model \
    --ref_model $ref_model \
    --data_name $data_name \
    --training_type dpo \
    --stage_type $stage_type \
    --dpo_iter $dpo_iter \
    --per_gpu_batch_size $per_gpu_batch_size \
    --learning_rate $learning_rate
```
- **$base_name:** Model name ```(google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b)```.
- **$ref_model:** Path to the reference model checkpoint.
- **$data_name:** Dataset name ```(csqa, obqa, strategyqa)```.
- **$stage_type:** Training stage ```(recall, analyze, recall_analyze)```.
- **$dpo_iter:** Current iteration timestamp.

For iterative DPO, use the ```run_dpo.sh``` script.

### :hourglass_flowing_sand: Inference
Use the following command to generate inferences:
```
python generate.py \
    --base_model $base_model \
    --data_name $data_name \
    --training_type $type_name \
    --stage_type $stage_type \
    --dpo_iter $dpo_iter \
    --checkpoint_dir $checkpoint_path\
    --per_gpu_batch_size $batch_size \
    --generation_type $generation_type \
    --generation_file $generation_file \
    --generation_times $generation_times
```
- **$base_name:** Model name ```(google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b)```.
- **$data_name:** Dataset name ```(csqa, obqa, strategyqa)```.
- **$training_type:** Phase type ```(sft, dpo)```.
- **$stage_type:** Inference stage ```(recall, analyze, summarize, recall_summarize, analyze_summarize, recall_analyze, recall_analyze_summarize)```.
- **$dpo_iter:** Iteration timestamp.
- **$generation_type:** Generation type ```(greedy, random)```.
- **$generation_file:** File type ```(test, dpo)```.
- **$generation_times:** Number of generation attempts.

## :page_facing_up: License

This project is licensed under the [MIT](LICENSE) © Shengmin Piao & Jieun Lee
