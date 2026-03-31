import json, os, sys
import schema_retriever.utils.db_utils as db_u
# import get_related_tab_col, Questions, load_tables_description, parse_db_info, apply_original_casing, save_and_extract_schema_info

def create_documents(DB_PATH):
    table_description = db_u.load_tables_description(DB_PATH, True)
    
    docs = {table.lower(): {} for table in table_description.keys()}
    
    for table_name, columns in table_description.items():
        cols = {}
        for column_name, column_info in columns.items():
            page_content = f"<table>{table_name}</table> <column>{column_name}</column>"
            if column_info.get("column_description", "").strip():
                page_content += f" <column description>{column_info['column_description']}</column_description>"
            if column_info.get("value_description", "").strip():
                page_content += f" <value description>{column_info['value_description']}</value description>"
            cols[column_name.lower()] = page_content
        docs[table_name.lower()] = cols
    
    return docs

def get_pos_neg_samples(sample: Questions, db_schema:dict) -> (list, list):
    positive_samples = []
    negative_samples = []
    for table, columns in db_schema.items():
        for column, value in columns.items():
            if table in sample.related_schema and column in sample.related_schema[table]:
                positive_samples.append(value)
            else:
                negative_samples.append(value)
                
    return positive_samples, negative_samples

def preprocess_fine_tuning_datasets(datas, db_infos):
    datasets = [
        db_u.Questions(
            db_id=data['db_id'],
            question_id=data['question_id'],
            question=data['question'],
            evidence=data['evidence'],
            SQL=data['SQL'],
            db_info=next(
                (
                    db_info 
                    for db_info in db_infos 
                    if db_info["db_id"] == data["db_id"]), 
                None
            ), 
        )
        for data in datas
    ]
      
    for sample in datasets:
        sample.related_schema = db_u.get_related_tab_col(sample)
    
    json_dataset = []
    cnt_db = {name: 0 for name in docs.keys()}    
    for sample in datasets:
        db_schema = docs[sample.db_id]
        pos_samples, neg_samples = get_pos_neg_samples(sample, db_schema)
        json_dataset.append(
            {
                "question": sample.question,
                "evidence": sample.evidence,
                "SQL": sample.SQL,
                "positive_columns": pos_samples,
                "negative_columns": neg_samples,
            }
        )
        cnt_db[sample.db_id] += 1
        
    new_dataset = []
    for sample in json_dataset:
        new_dataset.extend(
            [{
                'question': sample['question'],
                'evidence': sample['evidence'],
                'SQL': sample['SQL'],
                'positive_columns': [pos],
                'negative_columns': sample['negative_columns']
            }
             for pos in sample['positive_columns']]
        )
            
    with open("./schema_retriever/data/fine-tuning_samples_from_BIRD_augmented_version.json", "w", encoding='utf-8') as j:
        json.dump(new_dataset, j)    

def save_db_info_with_example_values(raw_db_info, path):
    db_infos = {}

    for db_info in raw_db_infos:
        db_infos[db_info['db_id']] = db_u.parse_db_info(db_info)

    with open(path, "w") as j:
        json.dump(db_infos, j, indent=4)

def preprocess_train_dataset(train_data, db_infos):
    def get_db_dict(_sample):
        tables = [t.lower() for t in _sample.db_info['table_names_original']]
        columns = _sample.db_info['column_names_original'][1:]
    
        tc_dict = {t:[] for t in tables}
    
        for c in columns:
            tc_dict[tables[c[0]]].append(c[1].lower())

        return {k: v for k, v in tc_dict.items() if v}
    datasets = [
        db_u.Questions(
            db_id=data['db_id'],
            question=data['question'],
            evidence=data['evidence'],
            SQL=data['SQL'],
            db_info=next(
                (
                    db_info 
                    for db_info in db_infos 
                    if db_info["db_id"] == data["db_id"]), 
                None
            ), 
        )
        for data in train_data
    ]

    for sample in datasets:
        sample.related_schema = db_u.get_related_tab_col(sample)

    datasets = [sample for sample in datasets if sum(len(cols) for cols in sample.related_schema.values())]
    
    full_schema_data = []

    for sample in datasets:
        q_sample = dict()
        q_sample['db_id'] = sample.db_id
        q_sample['question'] = sample.question
        q_sample['evidence'] = sample.evidence
        q_sample['SQL'] = sample.SQL
        q_sample['related_schemas'] = db_u.apply_original_casing(sample.related_schema, sample.db_info)
        q_sample['full_schemas'] = db_u.apply_original_casing(get_db_dict(sample), sample.db_info)
        
        full_schema_data.append(q_sample)
    
    with open("./schema_retriever/data/BIRD-train.json", "w") as j:
        json.dump(full_schema_data, j, indent=4)

    db_u.save_and_extract_schema_info(
        info_path="./schema_retriever/data/BIRD_train_database_infos.json",
        data_path="./schema_retriever/data/BIRD-train.json",
        column_names=[
            'related_columns',
            'full_schemas',
        ],
        save_path="./sql_generator/bird/train/BIRD-train-more-schema.json"
    )

def main:
    DATA_PATH = "./datasets/bird/train/"
    ROOT_DB_PATH = os.path.join(DATA_PATH, 'train_databases')
  
    with open(os.path.join(DATA_PATH, 'train_tables.json'), 'r') as j:
        db_infos = json.load(j)
    with open(os.path.join(DATA_PATH, 'train.json'), 'r') as j:
        datas = json.load(j)
      
    preprocess_fine_tuning_datasets(datas, db_infos)

    save_db_info_with_example_values(db_infos, "./schema_retriever/data/BIRD_train_database_infos.json")

    preprocess_train_dataset(datas, db_infos)
    # ------------------------------------------------------------------------------

    DATA_PATH = "./datasets/bird/dev_20240627/"
    ROOT_DB_PATH = os.path.join(DATA_PATH, 'dev_databases')
    
    with open(os.path.join(DATA_PATH, 'dev_tables.json'), 'r') as j:
        db_infos = json.load(j)
    
    with open(os.path.join(DATA_PATH, 'dev.json'), 'r') as j:
        datas = json.load(j)

    save_db_info_with_example_values(db_infos, "./schema_retriever/data/BIRD_dev_database_infos.json")

if __name__ == '__main__':
    main()
