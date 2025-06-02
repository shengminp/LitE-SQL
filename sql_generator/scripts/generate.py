import os
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
import sys
import json
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
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datetime import timedelta
from accelerate.utils import broadcast_object_list
from peft import (
    PeftConfig,
    PeftModel
)
from utils import (
    SFTTrainer,
    GenerationConfiguration,
    GenerationArguments,
    load_and_prepare_data,
    load_and_prepare_revise_data,
    post_process,
    post_process_spider,
    post_process_original,
    collect_result_parallel,
    build_sql_dataframe
)

logger = logging.getLogger(__name__)


def setup_args(config):
    if config.generation_type == "greedy":
        sql_generation_config = GenerationConfig(
            max_length=None,
            max_new_tokens=config.max_new_tokens,
            do_sample=False
        )
    else:
        sql_generation_config = GenerationConfig(
            max_length=None,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_k=config.top_k
        )

    training_args = GenerationArguments(
        output_dir=config.output_dir,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        log_level=config.log_level,
        logging_strategy=config.all_strategy,
        metric_for_best_model=config.metric_name,
        disable_tqdm=False,
        remove_unused_columns=True,
        predict_with_generate=True,
        data_name=config.data_name,
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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    
    return training_args


def setup_model_and_tokenizer(config, world_rank):
    """
    Setup model and tokenizer based on training configuration.
    """
    if config.training_phase == "original":
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
    else:
        peft_config = PeftConfig.from_pretrained(config.checkpoint_dir)
        if world_rank == 0:
            if not os.path.isdir(os.path.join(config.checkpoint_dir, 'merge')):
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    torch_dtype="auto",
                    attn_implementation="flash_attention_2",
                    tie_word_embeddings=False,
                )

                peft_model = PeftModel.from_pretrained(
                    model=base_model,
                    model_id=config.checkpoint_dir,
                    config=peft_config,
                    torch_dtype="auto",
                    attn_implementation="flash_attention_2",
                )

                peft_model = peft_model.merge_and_unload(progressbar=True)
                peft_model.save_pretrained(f"{config.checkpoint_dir}/merge")
                del peft_model
                del base_model
                torch.cuda.empty_cache()
        dist.barrier()

        model = AutoModelForCausalLM.from_pretrained(
            f"{config.checkpoint_dir}/merge",
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        padding_side="left"
    )
    model.eval()
    return model, tokenizer


def generate_data(trainer, config, tokenizer, test_dataset, training_args):
    logger.info(f'============ CHECK IMPORTANT INFO =================')
    logger.info(f">> Load model from {config.checkpoint_dir} <<")
    logger.info(f">> Save result to {config.generate_path} <<")

    torch.cuda.empty_cache()
    test_preds = trainer.predict(test_dataset)

    if trainer.is_world_process_zero():
        if config.training_phase == "original":
            total_results = post_process_original(test_preds.predictions, test_preds.label_ids, tokenizer)
        else:
            total_results = post_process('sft', test_preds.predictions, test_preds.label_ids, tokenizer)

        sql_gt = total_results['label']
        sql_pred = total_results['prediction']

        if config.data_name == "bird":
            sql_df = build_sql_dataframe(
                config.data_name,
                test_dataset,
                sql_gt,
                sql_pred,
                extra_fields=['difficulty']
            )
        elif config.data_name == "spider":
            if config.generation_file == "augment":
                sql_df = build_sql_dataframe(
                    config.data_name,
                    test_dataset,
                    ("", sql_gt),
                    ("", [[pred] for pred in sql_pred]),
                    extra_fields=['difficulty']
                )
            else:
                gt_pred_pairs = [post_process_spider(gt, pred) for gt, pred in zip(sql_gt, sql_pred)]
                post_sql_gt, post_sql_pred = map(list, zip(*gt_pred_pairs))
                sql_df = build_sql_dataframe(
                    config.data_name,
                    test_dataset,
                    (sql_gt, post_sql_gt),
                    (sql_pred, post_sql_pred),
                    extra_fields=['difficulty']
                )
        
        return sql_df
    else:
        return None


def revise_data(trainer, config, tokenizer, sql_df, training_args):
    if 'question_id' not in sql_df.columns:
        sql_df['question_id'] = [i for i in range(len(sql_df))]

    history_df = sql_df.copy()
    history_df["history_pred"] = [[] for _ in range(len(history_df))]
    history_df["history_error"] = [[] for _ in range(len(history_df))]

    for rev in range(config.revise_times):
        logger.info(f"============ Revise for {rev+1} time ============")
        if trainer.is_world_process_zero():
            result_df = collect_result_parallel(training_args, "test", sql_df if rev == 0 else revise_df)
            
            result_df = result_df.set_index(['question_id'])
            history_df = history_df.set_index(['question_id'])

            for ques in result_df.index:
                if ques in history_df.index:
                    pred_list = history_df.at[ques, 'history_pred']
                    err_list = history_df.at[ques, 'history_error']

                    if config.data_name == "bird":
                        pred_list.append(result_df.at[ques, 'sql_pred'])
                    elif config.data_name == "spider":
                        pred_list.append(result_df.at[ques, 'pre_sql_pred'])
                    err_list.append(result_df.at[ques, 'error_msg'])

                    history_df.at[ques, 'history_pred'] = pred_list
                    history_df.at[ques, 'history_error'] = err_list

            result_df = result_df.reset_index()
            history_df = history_df.reset_index()
        else:
            result_df = None
            history_df = None

        res_df_list = [result_df] if trainer.is_world_process_zero() else [None]
        his_df_list = [history_df] if trainer.is_world_process_zero() else [None]
        res_df_list = broadcast_object_list(res_df_list)
        his_df_list = broadcast_object_list(his_df_list)
        result_df = res_df_list[0]
        history_df = his_df_list[0]

        incorrect_df = result_df[result_df['res'] == 0]
        if incorrect_df.empty:
            logger.info(f"No incorrect predictions found; Break at iter_{rev+1}.")
            revise_df = result_df.copy()
            break
        else:
            revise_dataset = load_and_prepare_revise_data(incorrect_df, tokenizer, training_args)

            torch.cuda.empty_cache()

            revise_preds = trainer.predict(revise_dataset)

            if trainer.is_world_process_zero():
                total_results = post_process('rft', revise_preds.predictions, revise_preds.label_ids, tokenizer)
                sql_gt = total_results['label']
                sql_pred = total_results['prediction']

                if config.data_name == "bird":
                    revise_df = build_sql_dataframe(
                        config.data_name,
                        revise_dataset,
                        sql_gt,
                        sql_pred,
                        extra_fields=['difficulty']
                    )
                elif config.data_name == "spider":
                    gt_pred_pairs = [post_process_spider(gt, pred) for gt, pred in zip(sql_gt, sql_pred)]
                    post_sql_gt, post_sql_pred = map(list, zip(*gt_pred_pairs))
                    revise_df = build_sql_dataframe(
                        config.data_name,
                        revise_dataset,
                        (sql_gt, post_sql_gt),
                        (sql_pred, post_sql_pred),
                        extra_fields=['difficulty']
                    )
            else:
                revise_df = None

            dist.barrier()
            revise_df_list = [revise_df] if trainer.is_world_process_zero() else [None]
            revise_df_list = broadcast_object_list(revise_df_list)
            revise_df = revise_df_list[0]
            dist.barrier()
            
    revise_df = revise_df.set_index(['question_id'])
    history_df = history_df.set_index(['question_id'])

    for ques in revise_df.index:
        if ques in history_df.index:
            pred_list = history_df.at[ques, 'history_pred']
            if not isinstance(pred_list, list):
                pred_list = []
            if config.data_name == "bird":
                pred_list.append(revise_df.at[ques, 'sql_pred'])
            elif config.data_name == "spider":
                pred_list.append(revise_df.at[ques, 'pre_sql_pred'])
            history_df.at[ques, 'history_pred'] = pred_list

    revise_df = revise_df.reset_index()
    history_df = history_df.reset_index()

    return history_df


def augment_data(trainer, config, sql_df):
    if trainer.is_world_process_zero():
        with open(f"{config.generate_path}_tmp", 'w') as f:
            json.dump(sql_df.to_dict(orient='records'), f, ensure_ascii=True, indent=4)
    
    dist.barrier()
    if trainer.is_world_process_zero():
        sql_df = collect_result_parallel(training_args, "train", sql_df)
        drop_cols = ["question_id", "error", "timeout"]
        if config.data_name == "spider":
            drop_cols += ['pre_sql_gt', 'pre_sql_pred']
        if config.generation_mode == "revise":
            drop_cols += ["history_pred", "history_error"]
        sql_df = sql_df.drop(drop_cols, axis=1)
    else:
        sql_df = None
    dist.barrier()

    if trainer.is_world_process_zero():
        with open(f"{config.generate_path}", 'w') as f:
            json.dump(sql_df.to_dict(orient='records'), f, ensure_ascii=True, indent=4)
        # Merge the current df with previous df
        augmented_df = []
        for _, group in sql_df.groupby('question'):
            correct_df = group[group['res'] == 1]
            wrong_df = group[group['res'] == 0]

            if wrong_df.empty:
                continue
            elif len(wrong_df) == len(group) or len(wrong_df) == 1:
                selected = wrong_df.copy()
            else:
                if len(wrong_df) > len(correct_df):
                    n = len(wrong_df) - len(correct_df)
                    if n > len(correct_df):
                        n = len(correct_df)  
                    sampled_wrong = wrong_df.sample(n=n).copy()
                    correct_preds = correct_df['sql_pred'].values[:n]
                    sampled_wrong['sql_gt'] = correct_preds
                    selected = sampled_wrong
                else:
                    n = len(wrong_df) - 1
                    wrong_copy = wrong_df.copy()
                    if n > 0:
                        sampled_correct  = correct_df.sample(n=n)
                        correct_preds = sampled_correct['sql_pred'].values
                        wrong_copy.loc[wrong_copy.index[:n], 'sql_gt'] = correct_preds
                    selected = wrong_copy

            augmented_df.append(selected)
        sql_df_augmented = pd.concat(augmented_df, ignore_index=True)
        sql_df_augmented = sql_df_augmented.drop(["res"], axis=1)
        if config.data_name == "spider":
            sql_df_augmented['sql_gt'] = sql_df_augmented['sql_gt'].apply(lambda x: x[0] if isinstance(x, list) else x)
            sql_df_augmented['sql_pred'] = sql_df_augmented['sql_pred'].apply(lambda x: x[0])
        sql_df_augmented =  sql_df_augmented[sql_df_augmented['sql_gt'] != sql_df_augmented['sql_pred']]
        sql_dict_augment = sql_df_augmented.to_dict(orient='records')
        for item in sql_dict_augment:
            item['column'] = json.loads(item['column'])
        sql_dict_augment_no_duplicate = set(json.dumps(item, ensure_ascii=True, sort_keys=True) for item in sql_dict_augment)
        sql_dict_augment_no_duplicate = [json.loads(item) for item in sql_dict_augment_no_duplicate]
        with open(f"{config.merge_path}", 'w') as f:
            json.dump(sql_dict_augment_no_duplicate, f, ensure_ascii=True, indent=4)


def run_generation(training_args, config, model, tokenizer, tokenized_datasets):
    """
    Main training loop.
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        train_dataset=None,
        eval_dataset=None,
        processing_class=tokenizer,
        compute_metrics=None
    )

    total_sql_df_list = []
    for _ in range(config.generate_times):
        sql_df = generate_data(trainer, config, tokenizer, tokenized_datasets["test"], training_args)

        if config.generation_mode == "revise":
            dist.barrier()
            sql_df_list = [sql_df] if trainer.is_world_process_zero() else [None]
            sql_df_list = broadcast_object_list(sql_df_list)
            sql_df = sql_df_list[0]
            dist.barrier()
            sql_df = revise_data(trainer, config, tokenizer, sql_df, training_args)
        
        if trainer.is_world_process_zero():
            total_sql_df_list.append(sql_df)

    if trainer.is_world_process_zero():
        sql_df = pd.concat(total_sql_df_list)

    if config.generation_file == "augment":
        augment_data(trainer, config, sql_df)
    else: 
        if trainer.is_world_process_zero():
            with open(f"{config.generate_path}", 'w') as f:
                json.dump(sql_df.to_dict(orient='records'), f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Generation")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_phase', type=str, required=True, help='Training phase')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--per_gpu_batch_size', type=int, required=False, help='Batch size per GPU')

    parser.add_argument('--generation_mode', type=str, required=True, help='simple generate/+revise')
    parser.add_argument('--generation_type', type=str, required=True, help='Generation type')
    parser.add_argument('--generation_file', type=str, required=True, help='Generation file')
    parser.add_argument('--schema_file', type=str, required=True, help='Schema file')

    parser.add_argument('--revise_times', type=int, default=0, help='Revise times (used during revision)')

    parser.add_argument('--augment_times', type=int, default=1, help='Augment times')
    parser.add_argument('--generate_times', type=int, default=1, help='Generate times')
    parser.add_argument('--rft_iter', type=int, default=0, help='Iter for prepare RFT')

    return parser.parse_args()


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=3)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    world_rank = dist.get_rank()
    args = parse_args()
    config = GenerationConfiguration(**vars(args))
    training_args = setup_args(config)
    model, tokenizer = setup_model_and_tokenizer(config, world_rank)
    data = load_and_prepare_data(config, tokenizer, training_args)
    run_generation(training_args, config, model, tokenizer, data)