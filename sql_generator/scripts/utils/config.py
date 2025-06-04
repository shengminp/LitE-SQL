import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import GenerationConfig
from trl.trainer.dpo_trainer import DPOConfig

SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../../'))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../../../../../'))


@dataclass
class BaseConfiguration:
    base_model: str
    data_name: str
    training_phase: str
    per_gpu_batch_size: int
    gradient_accumulation_steps: int = 0
    per_device_eval_batch_size: int = 0
    steps: int = 100
    warmup_ratio: float = 0.1
    checkpoint_dir: str = None
    num_train_epochs: float = 0.0
    learning_rate: float = 0.0
    lora_r: int = 0
    lr_scheduler_type: str = "cosine"
    all_strategy: str = "steps"
    metric_name: str = "accuracy"
    log_level: str = "info"
    eval_mode: str = "generate"
    output_dir: str = "models/"
    dataset_dir: str = "datasets/"
    resume_from_checkpoint: str = None

    def __post_init__(self):
        self.dataset_dir = os.path.abspath(os.path.join(DATA_PATH, self.dataset_dir, self.data_name))
        self.output_dir = os.path.abspath(os.path.join(ROOT_PATH, self.output_dir, self.data_name, self.base_model))


@dataclass
class SFTConfiguration(BaseConfiguration):
    def __post_init__(self):
        super().__post_init__()
        self.generation_file = None
        ##########################################
        ######## LORA & SFT Configuration ########
        self.max_steps = 2000
        self.steps = 100
        self.warmup_ratio = 0.05
        
        if "1.5B" in self.base_model:
            self.per_gpu_batch_size = 4
            self.gradient_accumulation_steps = 1
            self.per_device_eval_batch_size = 256
            self.lora_r = 64
            self.learning_rate = 0.0001
        elif "3B" in self.base_model:
            self.per_gpu_batch_size = 2
            self.gradient_accumulation_steps = 2
            self.per_device_eval_batch_size = 128
            self.lora_r = 128
            self.learning_rate = 0.0001
        elif "7B" in self.base_model:
            self.per_gpu_batch_size = 2
            self.gradient_accumulation_steps = 2
            self.per_device_eval_batch_size = 64
            self.lora_r = 128
            self.learning_rate = 0.00005

        self.lora_alpha = self.lora_r * 2
        self.lora_dropout = 0.05
        self.lora_inference_mode = False
        self.lora_bias = "none"
        self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"
        ]
        ######## LORA & SFT Configuration ########
        ##########################################
        if self.data_name == "bird":
            self.max_new_tokens = 300
            self.db_dir = os.path.join(self.dataset_dir, "dev_20240627/dev_databases")
            self.dataset_path = {
                "train": os.path.join(self.dataset_dir, "train/sft/sql_train.json"),
                "valid": os.path.join(self.dataset_dir, "dev_20240627/retrieved/queried_schemas_top25.json")
            }
        elif self.data_name == "spider":
            self.max_new_tokens = 256
            self.db_dir = os.path.join(self.dataset_dir, "dev/database")
            self.dataset_path = {
                "train": os.path.join(self.dataset_dir, "train/sft/sft_train.json"),
                "valid": os.path.join(self.dataset_dir, "dev/retrieved/queried_schemas_top20.json")
            }
        self.all_strategy_steps = self.steps
        self.output_dir = os.path.join(self.output_dir, f"{self.training_phase}")


@dataclass
class SFTTrainingArguments(Seq2SeqTrainingArguments):
    data_name : Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"},
    )
    eval_mode : Optional[str] = field(
        default=None,
        metadata={"help": "dummy flag for evaluation"},
    )
    db_dir : Optional[str] = field(
        default=None,
        metadata={"help": "Directory path of DB"},
    )
    sql_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


@dataclass
class RFTConfiguration(BaseConfiguration):
    tune_model: str = None
    ref_model: str = None
    rft_iter: int = 0
    max_prompt_length: int = 4000
    max_length: int = 5000

    def __post_init__(self):
        super().__post_init__()
        self.num_train_epochs = 2
        self.steps = 100
        if "1.5B" in self.base_model:
            self.per_gpu_batch_size = 4
            self.gradient_accumulation_steps = 1
            self.per_device_eval_batch_size = 256
            self.lora_r = 64
            self.learning_rate = 0.000005
        elif "3B" in self.base_model:
            self.per_gpu_batch_size = 2
            self.gradient_accumulation_steps = 2
            self.per_device_eval_batch_size = 128
            self.lora_r = 128
            self.learning_rate = 0.000005
        elif "7B" in self.base_model:
            self.per_gpu_batch_size = 1
            self.gradient_accumulation_steps = 4
            self.per_device_eval_batch_size = 64
            self.lora_r = 128
            self.learning_rate = 0.000005
        self.warmup_ratio = 0.1
        self.beta = 0.1

        self.lora_alpha = self.lora_r * 2
        self.lora_dropout = 0.05
        self.lora_inference_mode = False
        self.lora_bias = "none"
        self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"
        ]
        ######## LORA Configuration ########
        ####################################

        if self.data_name == "bird":
            self.max_new_tokens = 300
            self.db_dir = os.path.join(self.dataset_dir, "dev_20240627/dev_databases")
            self.dataset_path = {
                "train": os.path.join(self.dataset_dir, f"train/rft/{self.base_model}/rft_train_merge.json"),
                "valid": os.path.join(self.dataset_dir, "dev_20240627/retrieved/queried_schemas_top25.json")
            }
        elif self.data_name == "spider":
            self.max_new_tokens = 256
            self.db_dir = os.path.join(self.dataset_dir, "dev/database")
            self.dataset_path = {
                "train": os.path.join(self.dataset_dir, f"train/rft/{self.base_model}/rft_train_merge.json"),
                "valid": os.path.join(self.dataset_dir, "dev/retrieved/queried_schemas_top20.json")
            }
        
        self.eval_mode = "revise"
        self.all_strategy_steps = self.steps

        if self.tune_model:
            if self.rft_iter == 1:
                tune_dir = os.path.join(self.output_dir, "sft")
            elif self.rft_iter > 1:
                tune_dir = os.path.join(self.output_dir, "rft", f"iter_{self.rft_iter-1}")
            self.tune_model = os.path.join(tune_dir, self.tune_model)
        
        if self.ref_model:
            if self.rft_iter == 1:
                ref_dir = os.path.join(self.output_dir, "sft")
            elif self.rft_iter > 1:
                ref_dir = os.path.join(self.output_dir, "rft", f"iter_{self.rft_iter-1}")
            self.ref_model = os.path.join(ref_dir, self.ref_model)
        
        self.output_dir = os.path.join(self.output_dir, self.training_phase, f"iter_{self.rft_iter}")


@dataclass
class RFTTrainingArguments(DPOConfig):
    data_name : Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"},
    )
    eval_mode : Optional[str] = field(
        default=None,
        metadata={"help": "dummy flag for evaluation"},
    )
    db_dir : Optional[str] = field(
        default=None,
        metadata={"help": "Directory path of DB"},
    )
    predict_with_generate: bool = field(
        default=False, 
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    sql_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


@dataclass
class GenerationConfiguration(BaseConfiguration):
    generate_dir: str = "results/"
    generation_mode: str = "generate"
    generation_type: str = None
    generation_file: str = None
    schema_file: str = None
    revise_times: int = 0
    augment_times: int = 1
    generate_times: int = 1
    cache_type: str = "offloaded"
    temperature: float = 0.7
    top_k: int = 50
    rft_iter: int = 0

    def __post_init__(self):
        super().__post_init__()
        if "1.5B" in self.base_model:
            self.per_device_eval_batch_size = 128
        elif "3B" in self.base_model:
            self.per_device_eval_batch_size = 128
        elif "7B" in self.base_model:
            self.per_device_eval_batch_size = 64

        if self.generation_file == "test":
            if self.data_name == "bird":
                self.max_new_tokens = 300
                self.db_dir = os.path.join(self.dataset_dir, "dev_20240627/dev_databases")
                self.dataset_path = {'test': os.path.join(self.dataset_dir, f"dev_20240627/retrieved/{self.schema_file}.json")}
            elif self.data_name == "spider":
                self.max_new_tokens = 256
                self.db_dir = os.path.join(self.dataset_dir, "test/test_database")
                self.dataset_path = {'test': os.path.join(self.dataset_dir, f"test/retrieved/{self.schema_file}.json")}

            self.generate_dir = os.path.abspath(os.path.join(ROOT_PATH, self.generate_dir, self.data_name, self.base_model, self.training_phase))
            suffix = f'{self.generation_file}_{self.generation_type}_{self.schema_file}_{self.revise_times}'
            if self.rft_iter > 0:
                self.generate_path = os.path.join(self.generate_dir, f'iter_{self.rft_iter}', f'{suffix}.json')
            else:
                self.generate_path = os.path.join(self.generate_dir, f'{suffix}.json')
        
        elif self.generation_file == "augment":
            if self.data_name == "bird":
                self.num_augment_info = 25
                self.max_new_tokens = 300
                self.dataset_path = {'test': os.path.join(self.dataset_dir, "train/sft/sql_train.json")}
                self.db_dir = os.path.join(self.dataset_dir, "train/train_databases")
            elif self.data_name == "spider":
                self.num_augment_info = 20
                self.max_new_tokens = 200
                self.db_dir = os.path.join(self.dataset_dir, "train/database")
                
            self.train_info_path = os.path.join(self.dataset_dir, f'train/{self.data_name.upper()}-train-more-schema.json')
            self.generate_path = os.path.join(self.dataset_dir, f'train/rft/{self.base_model}/rft_train_{self.rft_iter}.json')
            self.merge_path = os.path.join(self.dataset_dir, f'train/rft/{self.base_model}/rft_train_merge.json')

        if self.checkpoint_dir:
            self.output_dir = os.path.join(self.output_dir, self.training_phase)
            if self.training_phase == "sft" and self.rft_iter == 0:
                self.checkpoint_dir = os.path.join(self.output_dir, self.checkpoint_dir)
            elif self.rft_iter <= 1:
                if self.generation_file == "augment":
                    self.checkpoint_dir = os.path.join(self.output_dir, self.checkpoint_dir)
                else:
                    self.checkpoint_dir = os.path.join(self.output_dir, f"iter_{self.rft_iter}", self.checkpoint_dir)
            elif self.rft_iter > 1:
                if self.generation_file == "augment":
                    self.checkpoint_dir = os.path.join(self.output_dir, f"iter_{self.rft_iter-1}", self.checkpoint_dir)
                else:
                    self.checkpoint_dir = os.path.join(self.output_dir, f"iter_{self.rft_iter}", self.checkpoint_dir)

        self._create_generate_dir()
    
    def _create_generate_dir(self):
        dir_path = os.path.dirname(self.generate_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class GenerationArguments(Seq2SeqTrainingArguments):
    data_name : Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"},
    )
    eval_mode : Optional[str] = field(
        default=None,
        metadata={"help": "dummy flag for evaluation"},
    )
    db_dir : Optional[str] = field(
        default=None,
        metadata={"help": "Directory path of DB"},
    )
    sql_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
