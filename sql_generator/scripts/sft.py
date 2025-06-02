import os
import torch
import torch.distributed as dist
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
import sys
import logging
import datasets
import argparse
import transformers
from transformers import (
    set_seed,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint
from torch.distributed import is_initialized, get_rank
from peft import LoraConfig, TaskType, get_peft_model

from utils import (
    SFTTrainer,
    SFTConfiguration,
    SFTTrainingArguments,
    load_and_prepare_data,
    post_process,
    post_process_spider,
    collect_result_parallel,
    build_sql_dataframe
)

logger = logging.getLogger(__name__)

def setup_args(config):
    sql_generation_config = GenerationConfig(
        max_length=None,
        max_new_tokens=config.max_new_tokens,
        do_sample=False
    )

    training_args = SFTTrainingArguments(
        output_dir=config.output_dir,
        eval_strategy=config.all_strategy,
        per_device_train_batch_size=config.per_gpu_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
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
        predict_with_generate=True,
        data_name=config.data_name,
        eval_mode=config.eval_mode,
        db_dir=config.db_dir,
        sql_generation_config=sql_generation_config
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
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    
    return training_args


def setup_model_and_tokenizer(config):
    """
    Setup model and tokenizer based on training configuration.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
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
        target_modules = config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        padding_side="left",
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

        
def run_sft(training_args, model, tokenizer, tokenized_datasets):
    """
    Main training loop.
    """
    def _compute_metric(total_preds, eval_dataset, args, compute_result=False):
        if is_initialized() and get_rank() != 0:
            return {'accuracy': 0.0}
    
        total_results = post_process('sft', total_preds.predictions, total_preds.label_ids, tokenizer)
        sql_gt = total_results['label']
        sql_pred = total_results['prediction']
        if args.data_name == "bird":
            sql_df = build_sql_dataframe(
                args.data_name,
                eval_dataset,
                sql_gt,
                sql_pred
            )
        if args.data_name == "spider":
            gt_pred_pairs = [post_process_spider(gt, pred) for gt, pred in zip(sql_gt, sql_pred)]
            post_sql_gt, post_sql_pred = map(list, zip(*gt_pred_pairs))
            sql_df = build_sql_dataframe(
                args.data_name,
                eval_dataset,
                (sql_gt, post_sql_gt),
                (sql_pred, post_sql_pred)
            )
        sql_df = collect_result_parallel(args, "train", sql_df)
        results = sql_df['res'].tolist()
        metrics = {
            'accuracy': sum(results) / len(results),
            'sql_gt': sql_gt[:10],
            'sql_pred': sql_pred[:10]
        }
        return metrics

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
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
    parser = argparse.ArgumentParser(description="Configuration for Generation")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_phase', type=str, required=True, help='Training pahse')
    parser.add_argument('--per_gpu_batch_size', type=int, required=False, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, required=False, help='Batch size per GPU for eval')
    parser.add_argument('--num_train_epochs', type=float, required=False, help='Number of epoches for training')
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate')
    parser.add_argument('--lora_r', type=int, required=False, help='LoRA rank')

    return parser.parse_args()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    
    args = parse_args()
    config = SFTConfiguration(**vars(args))
    training_args = setup_args(config)
    model, tokenizer = setup_model_and_tokenizer(config)
    data = load_and_prepare_data(config, tokenizer, training_args)
    run_sft(training_args, model, tokenizer, data)