import os
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
import sys
import torch
import logging
import datasets
import argparse
import transformers
import pandas as pd
import torch.distributed as dist
from transformers import (
    set_seed,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.trainer_utils import get_last_checkpoint
from torch.distributed import is_initialized, get_rank
from peft import (
    LoraConfig,
    TaskType,
    PeftConfig,
    PeftModel,
)

from utils import (
    SDPOTrainer,
    RFTConfiguration,
    RFTTrainingArguments,
    load_and_prepare_dpo_data,
    post_process,
    post_process_spider,
    collect_result_parallel
)

logger = logging.getLogger(__name__)

def setup_args(config):
    sql_generation_config = GenerationConfig(
        max_length=None,
        max_new_tokens=config.max_new_tokens,
        do_sample=False
    )

    training_args = RFTTrainingArguments(
        output_dir=config.output_dir,
        eval_strategy=config.all_strategy,
        per_device_train_batch_size=config.per_gpu_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        log_level=config.log_level,
        logging_strategy=config.all_strategy,
        logging_first_step=True,
        logging_steps=10,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_strategy=config.all_strategy,
        save_steps=config.all_strategy_steps,
        save_total_limit=5,
        save_only_model=False,
        eval_steps=config.all_strategy_steps,
        metric_for_best_model=config.metric_name,
        load_best_model_at_end=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        bf16=True,
        label_names=["labels"],
        data_name=config.data_name,
        eval_mode=config.eval_mode,
        db_dir=config.db_dir,
        sql_generation_config=sql_generation_config,
        # for dpo
        model_adapter_name="base",
        ref_adapter_name="reference",
        beta=config.beta,
        rpo_alpha=1.0,
        predict_with_generate=True,
        max_prompt_length=config.max_prompt_length,
        max_length=config.max_length,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    return training_args

def setup_model_and_tokenizer(config, world_rank):
    """
    Setup model and tokenizer based on training configuration.
    """
    if world_rank == 0:
        peft_config = PeftConfig.from_pretrained(config.tune_model)
        if not os.path.isdir(os.path.join(config.tune_model, 'merge')):
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                tie_word_embeddings=False,
            )

            peft_model = PeftModel.from_pretrained(
                model=base_model,
                model_id=config.tune_model,
                config=peft_config,
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                is_trainable=True
            )

            peft_model = peft_model.merge_and_unload(progressbar=True)
            peft_model.save_pretrained(f"{config.tune_model}/merge")
            del peft_model
            del base_model
            torch.cuda.empty_cache()
    dist.barrier()

    model = AutoModelForCausalLM.from_pretrained(
        f"{config.tune_model}/merge",
        #torch_dtype="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=config.lora_inference_mode,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules = config.lora_target_modules
    )
    model.add_adapter(
        lora_config,
        adapter_name="base",
    )
    model.set_adapter("base")

    model.add_adapter(
        lora_config,
        adapter_name="reference",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        padding_side="left"
    )
    return model, tokenizer

        
def run_rft(training_args, model, tokenizer, datasets):
    """
    Main training loop.
    """
    def _compute_metric(total_preds, revise_dataset, correct_df, args, compute_result=False):
        if is_initialized() and get_rank() != 0:
            return {'accuracy': 0.0}
        
        total_results = post_process('rft', total_preds.predictions, total_preds.label_ids, tokenizer)
        sql_gt = total_results['label']
        sql_pred = total_results['prediction']
        if args.data_name == "spider":
            gt_pred_pairs = [post_process_spider(gt, pred) for gt, pred in zip(sql_gt, sql_pred)]
            sql_gt, sql_pred = map(list, zip(*gt_pred_pairs))

        revise_df = pd.DataFrame({
            'db_id': revise_dataset['db_id'],
            'question': revise_dataset['question'],
            'table': revise_dataset['table'],
            'column': revise_dataset['column'],
            'sql_gt': sql_gt,
            'sql_pred': sql_pred,
        })

        if "evidence" in revise_dataset.column_names:
            revise_df["evidence"] = revise_dataset["evidence"]
    
        sql_df = pd.concat([correct_df, revise_df], ignore_index=True)
        sql_df = collect_result_parallel(args, "train", sql_df)
        results = sql_df['res'].tolist()

        metrics = {'accuracy': sum(results) / len(results)}
        return metrics

    trainer = SDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['valid'],
        processing_class=tokenizer,
        compute_metrics=_compute_metric
    )
 
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for SDPO training")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--tune_model', type=str, default=None, help='Tuning model')
    parser.add_argument('--ref_model', type=str, default=None, help='Reference model')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_phase', type=str, required=True, help='Training phase')
    parser.add_argument('--per_gpu_batch_size', type=int, required=False, help='Batch size per GPU')
    parser.add_argument('--num_train_epochs', type=float, required=False, help='Number of epoches for training')
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate')
    parser.add_argument('--rft_iter', type=int, default=0, help='Iter for prepare RFT')

    return parser.parse_args()

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_rank = dist.get_rank()
    args = parse_args()
    config = RFTConfiguration(**vars(args))
    training_args = setup_args(config)
    model, tokenizer = setup_model_and_tokenizer(config, world_rank)
    data = load_and_prepare_dpo_data(config, tokenizer, training_args)
    run_rft(training_args, model, tokenizer, data)