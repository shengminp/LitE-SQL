import os
import re
import sqlite3
import threading
import numpy as np
import pandas as pd
from itertools import chain
from func_timeout import func_timeout, FunctionTimedOut
from .spider_tool.parse import remove_distinct, get_all_preds_for_execution
from .spider_tool.exec_eval import postprocess, replace_cur_year, result_eq


def post_process(phase, pred_ids, label_ids, tokenizer):
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    pred_tokens = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    label_tokens = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    label_sql = [label.strip() for label in label_tokens]

    pattern = re.compile(r'<\|im_start\|>assistant(.*?)<\|(im_end|endoftext)\|>', re.DOTALL)

    pred_sql = []
    for pred in pred_tokens:
        matches = list(pattern.finditer(pred))
        if phase == "sft":
            pred_sql.append(matches[0].group(1).strip() if matches else "")
        elif phase == "rft":
            if len(matches) > 1:
                pred_sql.append(matches[1].group(1).strip())
            else:
                pred_sql.append(matches[0].group(1).strip() if matches else "")

    return {'prediction': pred_sql, 'label': label_sql}


def post_process_original(pred_ids, label_ids, tokenizer):
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    pred_tokens = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    label_tokens = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    label_sql = [label.strip() for label in label_tokens]

    pattern = re.compile(
        r"```sql\s*([\s\S]*?)(?=\s*```|<\|im_end\|>|<ANSWER>|<\|endoftext\|>)",
        re.MULTILINE
    )

    pred_sql = []
    for pred in pred_tokens:
        match = pattern.search(pred)
        if match:
            block = match.group(1).strip()
            block = block.replace("\n", " ")
            block = re.sub(r'\s+', ' ', block)
            pred_sql.append(block)
        else:
            pred_sql.append("")

    return {'prediction': pred_sql, 'label': label_sql}


def post_process_spider(gt, pred):
    gt = postprocess(gt)
    gt = remove_distinct(gt)

    pred = pred.replace("value", "1")
    pred = postprocess(pred)
    pred = remove_distinct(pred)

    try:
        _, pred_value = func_timeout(1, get_all_preds_for_execution, args=(gt, pred))
        pred_list = list(chain([pred], pred_value))
    except FunctionTimedOut:
        pred_list = [pred]

    gt = replace_cur_year(gt)
    pred_list = [replace_cur_year(sql) for sql in pred_list]
    return gt, pred_list


def build_sql_dataframe(data_name, dataset, sql_gt, sql_pred, extra_fields=None):
    if data_name == "bird":
        data = {
            'db_id': dataset['db_id'],
            'question': dataset['question'],
            'table': dataset['table'],
            'column': dataset['column'],
            'foreign_keys': dataset['foreign_keys'],
            'sql_gt': sql_gt,
            'sql_pred': sql_pred
        }
    elif data_name == "spider":
        pre_sql_gt, post_sql_gt = sql_gt
        pre_sql_pred, post_sql_pred = sql_pred
        data = {
            'db_id': dataset['db_id'],
            'question': dataset['question'],
            'table': dataset['table'],
            'column': dataset['column'],
            'foreign_keys': dataset['foreign_keys'],
            'sql_gt': post_sql_gt,
            'sql_pred': post_sql_pred,
            'pre_sql_gt': pre_sql_gt,
            'pre_sql_pred': pre_sql_pred
        }

    if "evidence" in dataset.column_names:
        data["evidence"] = dataset["evidence"]
    if "question_id" in dataset.column_names:
        data["question_id"] = dataset["question_id"]

    if extra_fields:
        for f in extra_fields:
            if f in dataset.column_names:
                data[f] = dataset[f]
                
    return pd.DataFrame(data)


_thread_local = threading.local()


def get_conn(db_path):
    tl = _thread_local
    if not hasattr(tl, "conn") or tl.db_path != db_path:
        if hasattr(tl, "conn"):
            tl.conn.close()
        tl.conn = sqlite3.connect(db_path, check_same_thread=True)
        tl.db_path = db_path
    return tl.conn


def execute_sql_parallel(data_name, sql_pred, sql_label, db_path):
    conn = get_conn(db_path)
    cursor = conn.cursor()
    if sql_label is not None:
        if data_name == "bird":
            cursor.execute(sql_pred)
            res_pred = cursor.fetchall()
            cursor.execute(sql_label)
            res_label = cursor.fetchall()
            return 1 if set(res_pred) == set(res_label) else 0
        
        elif data_name == "spider":
            order_matters = 'order by' in sql_label.lower()
            for pred in sql_pred:
                pred_passes = 1
                cursor.execute(pred)
                res_pred = cursor.fetchall()
                cursor.execute(sql_label)
                res_label = cursor.fetchall()
                if not result_eq(res_label, res_pred, order_matters=order_matters):
                    pred_passes = 0
                if pred_passes == 1:
                    return {
                        "res": 1,
                    }
            return {
                "res": 0,
            }
    else:
        if data_name == "bird":
            cursor.execute(sql_pred)
            res_pred = cursor.fetchall()
            # empty resul:
            if not res_pred or not res_pred[0]:
                return 0
            else:
                first_val = res_pred[0][0]
                if first_val in (0, None, ''):
                    return 0
            
            # None result
            if all(all(c is None for c in row) for row in res_pred):
                return 0
            
            return 1
        elif data_name == "spider":
            for pred in sql_pred:
                pred_passes = 1
                cursor.execute(pred)
                res_pred = cursor.fetchall()
                # empty result
                if not res_pred or not res_pred[0]:
                    pred_passes = 0
                else:
                    first_val = res_pred[0][0]
                    if first_val in (0, None, ''):
                        pred_passes = 0
                
                # None result
                if all(all(c is None for c in row) for row in res_pred):
                    pred_passes = 0
                
                if pred_passes == 1:
                    return {
                        "res": 1,
                        "final_sql": pred
                    }
            return {
                "res": 0,
                "final_sql": sql_pred[0]
            }

from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_result_parallel(args, revise_mode, sql_df):
    records = sql_df.to_dict('records')

    def process_one(rec):
        result_msg = {
            'res': 0,
            'error': False,
            'timeout': False,
            'error_msg': None
        }
        if 'history_pred' in rec.keys():
            pred = rec['history_pred'][-1]
            if args.data_name == "spider":
                pred = [pred]
        else:
            pred = rec['sql_pred']

        db_path = os.path.join(args.db_dir, rec['db_id'], f"{rec['db_id']}.sqlite")

        try:
            if revise_mode == "train":
                gt = rec['sql_gt']
            elif revise_mode == "test":
                gt = None

            res = func_timeout(30.0, execute_sql_parallel, args=(args.data_name, pred, gt, db_path))
            if args.data_name == "bird":
                result_msg['res'] = res
                if res == 0:
                    result_msg['error_msg'] = 'runnable but wrong'
            elif args.data_name == "spider":
                result_msg['res'] = res['res']
                if res['res'] == 0:
                    result_msg['error_msg'] = 'runnable but wrong order'
        except FunctionTimedOut:
            result_msg['timeout'] = True
            if args.data_name == "bird":
                result_msg['error_msg'] = 'runnable but wrong'
            elif args.data_name == "spider":
                 result_msg['error_msg'] = 'runnable but wrong order'
        except Exception as e:
            result_msg['error'] = True
            result_msg['error_msg'] = str(e)

        return result_msg

    n_workers = int(os.cpu_count()/2)
    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        future_to_idx = {
            exe.submit(process_one, rec): idx
            for idx, rec in enumerate(records)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as err:
                results[idx] = {
                    'res': 0,
                    'error': True,
                    'timeout': False,
                    'error_msg': f"Internal error: {err}"
                }

    result_df = pd.DataFrame(results)
    out_df = sql_df.reset_index(drop=True).copy()
    for c in result_df.columns:
        out_df[c] = result_df[c]

    return out_df


def execute_sql(data_name, sql_pred, sql_label, db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if data_name == "bird":
            cursor.execute(sql_pred)
            res_pred = cursor.fetchall()
            cursor.execute(sql_label)
            res_label = cursor.fetchall()
            return 1 if set(res_pred) == set(res_label) else 0
        
        elif data_name == "spider":
            order_matters = 'order by' in sql_label.lower()
            for pred in sql_pred:
                pred_passes = 1
                cursor.execute(pred)
                res_pred = cursor.fetchall()
                cursor.execute(sql_label)
                res_label = cursor.fetchall()
                if not result_eq(res_label, res_pred, order_matters=order_matters):
                    pred_passes = 0
                if pred_passes == 1:
                    return 1
            return 0


def collect_result(args, sql_df):
    results = []
    for _, item in sql_df.iterrows():
        result_msg = {
            'res': 0,
            'error': False,
            'timeout': False,
            'error_msg': None
        }
        prediction = item['sql_pred']
        ground_truth = item['sql_gt']
        db_path = f"{args.db_dir}/{item['db_id']}/{item['db_id']}.sqlite"

        try:
            result = func_timeout(30.0, execute_sql, args=(args.data_name, prediction, ground_truth, db_path))
            result_msg['res'] = result
            if result == 0:
                result_msg['error_msg'] = (
                    'runnable but wrong' if args.data_name == "bird" 
                    else 'runnable but wrong order'
                )
        except FunctionTimedOut:
            result_msg['timeout'] = True
            result_msg['error_msg'] = 'runnable but wrong'
        except Exception as e:
            result_msg['error'] = True
            result_msg['error_msg'] = str(e)
        
        results.append(result_msg)
    
    result_df = pd.DataFrame(results)
    sql_df = sql_df.reset_index(drop=True).copy()
    for col in result_df.columns:
        sql_df[col] = result_df[col]

    return sql_df