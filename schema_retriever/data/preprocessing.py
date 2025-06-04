import json, os, sys
from schema_retriever.utils.db_utils import get_related_tab_col, Questions, load_tables_description

def create_documents(DB_PATH):
    table_description = load_tables_description(DB_PATH, True)
    
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

def main:
  DATA_PATH = "BIRD/train/"
  
  with open(os.path.join(DATA_PATH, 'train_tables.json'), 'r') as j:
      db_infos = json.load(j)
  
  with open(os.path.join(DATA_PATH, 'train.json'), 'r') as j:
      datas = json.load(j)
  
  datasets = [
      Questions(
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
      sample.related_schema = get_related_tab_col(sample)
    
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
  for sample in datasets:
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
      
  with open("data/fine-tuning_samples_from_BIRD_augmented_version.json", "w", encoding='utf-8') as j:
      json.dump(new_dataset, j)
    
if __name__ == '__main__':
    main()
