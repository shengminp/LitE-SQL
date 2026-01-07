import re
import copy
import json
import random
import pandas as pd
from collections import defaultdict
from datasets import Dataset

def build_line_style_schema(table_list, column_list):
    schema_info = ["\n### Database\n-- Tables and Columns"]
    for table_name, column_item in zip(table_list, json.loads(column_list)):
        if column_item == "*":
            schema_info.append(f"{table_name}.{column_item}")
            continue
        for column_name, column_info in column_item.items():
            if column_name == "*":
                schema_info.append(f"{table_name}.{column_name}")
                continue
            column_data = {
                "type": column_info.get("type").upper(),
                "primary_key": bool(column_info.get("primary_key", False)),
            }
            if column_info.get("values"):
                value_list = []
                for v in column_info["values"]:
                    if type(v) is str and len(v) < 30:
                        value_list.append(v)
                    elif type(v) is int or type(v) is float:
                        value_list.append(v)
                if len(value_list) > 0:
                    column_data['values'] = value_list
            
            if column_info.get("description") and column_info["description"] != "" and len(column_info["description"]) < 60:
                column_data['description'] = column_info["description"].strip()

            if column_info.get("comment") and column_info["comment"] != "" and len(column_info["comment"]) < 30:
                column_data['comment'] = column_info["comment"].strip()

            schema_info.append(f"{table_name}.{column_name} = {json.dumps(column_data)}")
    schema_info = "\n".join(schema_info)
    return schema_info


def merge_line_style_prompt(question, table_list, column_list, foreign_key, evidence):
    schema_text = build_line_style_schema(table_list, column_list)
    parts = [f"### Question\n{question.strip()}", schema_text]

    if foreign_key:
        fk_text = " | ".join(foreign_key)
        parts.append(f"\n-- Foreign Keys\n{fk_text}")

    if evidence:
        parts.append(f"\n-- Evidence\n{evidence.strip()}")

    return "\n".join(parts)

def format_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_sql(sql):
    # merge multiple space to single space
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # replace "" to ''
    sql = re.sub(r'"([^"]*)"', r"'\1'", sql)

    # Save the content part
    literals = re.findall(r"'[^']*'", sql)
    placeholders = {}
    for i, lit in enumerate(literals):
        key = f"__lit_{i}__"
        placeholders[key] = lit
        sql = sql.replace(lit, key)
    
    # lower the whole sql
    #sql = sql.lower()
    
    # recovore the content part
    for key, lit in placeholders.items():
        sql = sql.replace(key, lit)
    
    # clean the space in  ()
    sql = re.sub(r'\(\s*', '(', sql)
    sql = re.sub(r'\s*\)', ')', sql)
    
    # add ;
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


def deduplicate_foreign_keys(fk_list: list[str]) -> list[str]:
    seen = set()
    result = []

    for fk in fk_list:
        parts = [p.strip() for p in fk.split('=')]
        if len(parts) != 2:
            continue

        key = frozenset(parts)
        if key not in seen:
            seen.add(key)
            result.append(f"{parts[0]} = {parts[1]}")
    
    return result


def get_foreign_keys_relations(schema: dict) -> list:
    foreign_keys = []

    for table_name, columns in schema.items():
        if type(columns) is list:
            columns = {list(col.keys())[0]: list(col.values())[0] for col in columns}
        if type(columns) is str:
                continue
        for column_name, infos in columns.items():
            if column_name == "*":
                continue
            if infos.get("foreign_key", ""):
                foreign_keys.append(f"{table_name}.{column_name} = {infos['foreign_key']}")

    foreign_keys = deduplicate_foreign_keys(foreign_keys)
    
    return foreign_keys
   

def augment_data(config):
    train_df = pd.read_json(config.train_info_path, encoding="utf-8")

    all_augmented_data = []
    id_count = 0
    for train_item in train_df.to_dict(orient='records'):
        best_schemas_pair = {(k, v) for k, values in train_item['related_schemas'].items() for v in values}
        full_schemas_pair = {(k, v) for k, values in train_item['full_schemas'].items() for v in values}
        remain_schemas_pair = list(full_schemas_pair - best_schemas_pair)

        for _ in range(config.augment_times):
            sample_count = max(0, config.num_augment_info - len(best_schemas_pair))
            sample_schemas_pair = set(random.sample(remain_schemas_pair, min(sample_count, len(remain_schemas_pair))))
            
            final_schemas_pair = best_schemas_pair | sample_schemas_pair
            
            final_schemas = defaultdict(dict)
            merge_key = None
            for k, v in final_schemas_pair:
                if v == "*":
                    merge_key = k
                    continue
                final_schemas[k][v] = train_item['full_schemas'][k][v]

            if merge_key is not None:
                final_schemas[merge_key].clear()
                final_schemas[merge_key] = "*"
                
            foreign_keys = get_foreign_keys_relations(final_schemas)
            
            item = {
                "question_id": id_count,
                "db_id": train_item["db_id"],
                "question": train_item['question'].strip(),
                "table": list(final_schemas.keys()),
                "column": list(final_schemas.values()),
                "sql_gt": normalize_sql(train_item['SQL'].strip()),
                "foreign_keys": foreign_keys
            }
            id_count += 1

            if "evidence" in train_item.keys():
                item["evidence"] = train_item["evidence"].strip()

            all_augmented_data.append(item)

    return pd.DataFrame(all_augmented_data)


def load_and_prepare_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for training.
    """ 
    def preprocess_function(dataset):
        formatted_inputs, prompt_lengths = [], []

        table_data = dataset.get("table")
        column_data = dataset.get("column")
        evidence_data = dataset.get("evidence", [None] * len(dataset["question"]))
        sql_preds = dataset.get("sql_pred", [None] * len(dataset["question"]))
        error_msgs = dataset.get("error_msg", [None] * len(dataset["question"]))
        has_question_id = "question_id" in dataset.keys()

        for ques, sql_gt, sql_pred, table_list, column_list, foreign_key, evidence, error_msg in zip(
            dataset["question"], dataset["sql_gt"], sql_preds, table_data, column_data, 
            dataset["foreign_keys"], evidence_data, error_msgs):

            formatted_ques = merge_line_style_prompt(ques, table_list, column_list, foreign_key, evidence)
           
            if error_msg is None:
                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": formatted_ques},
                ])
            else:
                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": formatted_ques},
                    {"role": "assistant", "content": sql_pred},
                    {"role": "user", "content": f"### Error Message\n{error_msg}"},
                ])
        
            prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\n?", "", prompt, flags=re.DOTALL).strip()

            if config.training_phase == "original":
                prompt += "\n```sql\n"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            if config.training_phase == "original":
                formatted_inputs.append(prompt)
            elif not has_question_id:
                formatted_inputs.append(f"{prompt}\n{sql_gt.strip()}<|im_end|>")
            else:
                formatted_inputs.append(prompt)

            prompt_lengths.append(len(prompt_ids))

        model_inputs = tokenizer(formatted_inputs, add_special_tokens=False)

        if config.training_phase == "original":
            labels = [f"{sql}<|im_end|>" for sql in dataset['sql_gt']]
            model_inputs["labels"] = tokenizer(labels, add_special_tokens=False)["input_ids"]
        elif not has_question_id:
            labels = copy.deepcopy(model_inputs["input_ids"])

            for i in range(len(labels)):
                labels[i][:prompt_lengths[i]] = [-100] * prompt_lengths[i]
            model_inputs["labels"] = labels
        else:
            labels = [f"{sql}<|im_end|>" for sql in dataset['sql_gt']]
            model_inputs["labels"] = tokenizer(labels, add_special_tokens=False)["input_ids"]
        
        return model_inputs
     
    def make_keep_short(max_length):
        def keep_short(example):
            return len(example["input_ids"]) <= max_length
        return keep_short
    
    if config.generation_file == "augment":
        augmented_df = augment_data(config)
        augmented_df['column'] = augmented_df['column'].apply(json.dumps)
        ds = {"test": Dataset.from_pandas(augmented_df)}
    else:
        ds = {}
        for split, path in config.dataset_path.items():
            df = pd.read_json(path)
            df['column'] = df['column'].apply(json.dumps)
            ds[split] = Dataset.from_pandas(df)

    with training_args.main_process_first():
        tokenized_datasets = {name: data.map(preprocess_function, batched=True, num_proc=4) for name, data in ds.items()}
        if config.training_phase == "sft" or config.training_phase == "original":
            if hasattr(config, "generation_file") and config.generation_file == "augment":
                tokenized_datasets['test'] = tokenized_datasets['test'].filter(make_keep_short(3000), num_proc=4)
            elif hasattr(config, "generation_file") and config.generation_file == "test":
                tokenized_datasets['test'] = tokenized_datasets['test'].filter(make_keep_short(32768), num_proc=4)
            else:
                tokenized_datasets['train'] = tokenized_datasets['train'].filter(make_keep_short(3000), num_proc=4)

    return tokenized_datasets


def load_and_prepare_dpo_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for training.
    """ 
    def preprocess_function(dataset):
        table_data = dataset.get("table")
        column_data = dataset.get("column")
        evidence_data = dataset.get("evidence", [None] * len(dataset["question"]))
        has_question_id = "question_id" in dataset.keys()

        if not has_question_id:
            prompt_list, chosen_list, reject_list = [], [], []
            for ques, sql_gt, sql_pred, table_list, column_list, foreign_key, evidence, error_msg in zip(
                dataset["question"], dataset["sql_gt"], dataset["sql_pred"],
                table_data, column_data, dataset["foreign_keys"], evidence_data, dataset["error_msg"]):

                schema_text = merge_line_style_prompt(ques, table_list, column_list, foreign_key, evidence)

                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": schema_text},
                    {"role": "assistant", "content": sql_pred},
                    {"role": "user", "content": f"### Error Message\n{error_msg}"},
                ])
                prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\n?", "", prompt, flags=re.DOTALL)

                prompt_list.append(prompt)
                chosen_list.append(f"{sql_gt}<|im_end|>")
                reject_list.append(f"{sql_pred}<|im_end|>")
                
            return {"prompt": prompt_list, "chosen": chosen_list, "rejected": reject_list}
        else:
            formatted_inputs = []
            for ques, table_list, column_list, foreign_key, evidence in zip(
                dataset["question"], table_data, column_data, dataset["foreign_keys"], evidence_data):

                schema_text = merge_line_style_prompt(ques, table_list, column_list, foreign_key, evidence)
                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": schema_text},
                ])
                prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\n?", "", prompt, flags=re.DOTALL).strip()

                formatted_inputs.append(prompt)
        
            model_inputs = tokenizer(formatted_inputs, add_special_tokens=False)

            labels = [f"{sql}<|im_end|>" for sql in dataset['sql_gt']]
            model_inputs["labels"] = tokenizer(labels, add_special_tokens=False)["input_ids"]
            
            return model_inputs
    
    def make_keep_short(max_length):
        def keep_short(example):
            return len(example["prompt"]) <= max_length
        return keep_short
    
    ds = {}
    for split, path in config.dataset_path.items():
        df = pd.read_json(path)
        df['column'] = df['column'].apply(json.dumps)
        ds[split] = Dataset.from_pandas(df)

    with training_args.main_process_first():
        clean_datasets = {name: data.map(preprocess_function, batched=True) for name, data in ds.items()}
        if config.training_phase == "rft":
            clean_datasets['train'] = clean_datasets['train'].filter(make_keep_short(config.max_prompt_length), num_proc=4)

    return clean_datasets


def load_and_prepare_revise_data(df, tokenizer, training_args):
    """
    Load and prepare dataset for revise.
    """ 
    def preprocess_function(dataset):
        table_data = dataset.get("table")
        column_data = dataset.get("column")
        evidence_data = dataset.get("evidence", [None] * len(dataset["question"]))
        pre_sql_data = dataset.get("pre_sql_pred", [None] * len(dataset["question"]))

        prompt_list = []
        for ques, sql_pred, pre_sql_pred, table_list, column_list, foreign_key, evidence, error_msg in zip(
            dataset["question"], dataset["sql_pred"], pre_sql_data, table_data, 
            column_data, dataset["foreign_keys"], evidence_data, dataset["error_msg"]):

            schema_text = merge_line_style_prompt(ques, table_list, column_list, foreign_key, evidence)
            
            if training_args.data_name == "bird":
                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": schema_text},
                    {"role": "assistant", "content": sql_pred},
                    {"role": "user", "content": f"### Error Message\n{error_msg}"},
                ])
            elif training_args.data_name == "spider":
                prompt = format_chat(tokenizer, [
                    {"role": "user", "content": schema_text},
                    {"role": "assistant", "content": pre_sql_pred.strip()},
                    {"role": "user", "content": f"### Error Message\n{error_msg}"},
                ])

            prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\n?", "", prompt, flags=re.DOTALL).strip() 
            prompt_list.append(prompt)

        model_inputs = tokenizer(prompt_list, add_special_tokens=False)

        if training_args.data_name == "bird":
            labels = [f"{sql}<|im_end|>" for sql in dataset['sql_gt']]
        elif training_args.data_name == "spider":
            labels = [f"{sql}<|im_end|>" for sql in dataset['pre_sql_gt']]
        model_inputs["labels"] = tokenizer(labels, add_special_tokens=False)["input_ids"]

        return model_inputs
    
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    with training_args.main_process_first():
        clean_datasets = ds.map(preprocess_function, batched=True)

    return clean_datasets
