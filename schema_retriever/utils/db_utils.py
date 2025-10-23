import copy
import json
import random
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Questions:
    question_id: int = None # question_id
    db_id: str = None # db_id
    question: str = None
    SQL: str = None # Golden SQL
    difficulty: str = None
    db_info: dict = None # corresponding database info.
    evidence: str = None # Hint for SQL
    related_schema: dict = None # Related Table.Column

def load_tables_description(db_directory_path: str, use_value_description: bool) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads table descriptions from CSV files in the database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        use_value_description (bool): Whether to include value descriptions.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table descriptions.
    """
    encoding_types = ['utf-8-sig', 'cp1252']
    description_path = Path(db_directory_path) / "database_description"
    
    if not description_path.exists():
        print(f"Description path does not exist: {description_path}")
        return {}
    
    table_description = {}
    for csv_file in description_path.glob("*.csv"):
        table_name = csv_file.stem.lower().strip()
        table_description[table_name] = {}
        could_read = False
        for encoding_type in encoding_types:
            try:
                table_description_df = pd.read_csv(csv_file, index_col=False, encoding=encoding_type)
                table_description_df = table_description_df.dropna(subset=['original_column_name', 'column_name'], how='all')
                for _, row in table_description_df.iterrows():
                    column_name = row['original_column_name']
                    expanded_column_name = row.get('column_name', '').strip() if pd.notna(row.get('column_name', '')) else ""
                    column_description = row.get('column_description', '').replace('\n', ' ').replace("commonsense evidence:", "").strip() if pd.notna(row.get('column_description', '')) else ""
                    data_format = row.get('data_format', '').strip() if pd.notna(row.get('data_format', '')) else ""
                    value_description = ""
                    if use_value_description and pd.notna(row.get('value_description', '')):
                        value_description = row['value_description'].replace('\n', ' ').replace("commonsense evidence:", "").strip()
                        if value_description.lower().startswith("not useful"):
                            value_description = value_description[10:].strip()
                    
                    table_description[table_name][column_name.lower().strip()] = {
                        "original_column_name": column_name,
                        "column_name": expanded_column_name,
                        "column_description": column_description,
                        "data_format": data_format,
                        "value_description": value_description
                    }
                # print(f"Loaded descriptions from {csv_file} with encoding {encoding_type}")
                could_read = True
                break
            except Exception:
                continue
    if not could_read:
        print(f"Could not read descriptions from {csv_file}")
    return table_description

def apply_original_casing(schema, db_info):
    table_name_map = {t.lower(): t for t in db_info['table_names_original']}
    column_name_map = {}

    for table_idx, col_name in db_info['column_names_original']:
        if table_idx == -1:
            continue
        table = db_info['table_names_original'][table_idx]
        key = (table.lower(), col_name.lower())
        column_name_map[key] = col_name

    result = {}
    for table_lc, columns in schema.items():
        table_original = table_name_map.get(table_lc.lower(), table_lc)
        result[table_original] = []

        for col in columns:
            key = (table_lc.lower(), col.lower())
            col_original = column_name_map.get(key, col)
            result[table_original].append(col_original)

    return result

def get_column_sample_values(db_path: str, table: str, column: str, sample_size: int = 2) -> List[Any]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = f'''
        SELECT "{column}"
        FROM "{table}"
        WHERE "{column}" IS NOT NULL
        GROUP BY "{column}"
        LIMIT 10
        '''
        cursor.execute(query)
        raw_values = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        filtered = [v for v in raw_values if v not in ("", None)]
        
        if len(filtered) <= sample_size:
            return filtered
        
        return random.sample(filtered, sample_size)

    except Exception as e:
        return [f"[Error: {e}]"]
    
def flatten_primary_keys(raw_keys):
    result = set()
    for item in raw_keys:
        if isinstance(item, list):
            result.update(item)
        else:
            result.add(item)
    return result    

def parse_db_info(db_info: dict):
    db_id = db_info['db_id']
    tables_original = db_info['table_names_original']
    columns = db_info['column_names_original']
    column_names = db_info['column_names']
    column_types = db_info['column_types']
    primary_keys = flatten_primary_keys(db_info['primary_keys'])
    foreign_keys_raw = db_info['foreign_keys']
    descriptions = load_tables_description(os.path.join(ROOT_DB_PATH, db_info['db_id']), True)

    table_columns = {table: {} for table in tables_original}
    foreign_keys = {}

    for idx, (table_idx, col_name_orig) in enumerate(columns):
        if table_idx == -1:
            continue
        table_name = tables_original[table_idx]
        col_name_disp = column_names[idx][1]
        col_type = column_types[idx]

        desc_entry = descriptions.get(table_name, {}).get(col_name_orig.lower(), {})
        description = ""
        if desc_entry.get("column_description", ""):
            description += f"{desc_entry['column_description']} "
        if desc_entry.get("value_description", ""):
            description += f"{desc_entry['value_description']}"
        
        values = get_column_sample_values(
                    db_path = os.path.join(ROOT_DB_PATH, db_id, db_id+".sqlite"),
                    table = table_name,
                    column = col_name_orig
                )
        
        col_info = {
            "type": col_type,
            "comment": col_name_disp,
            "description": description,
            "values": values,
            "primary_key": idx in primary_keys,
            "foreign_key": None
        }

        table_columns[table_name][col_name_orig] = col_info

    for source_idx, target_idx in foreign_keys_raw:
        source_table = tables_original[columns[source_idx][0]]
        source_column = columns[source_idx][1]
        target_table = tables_original[columns[target_idx][0]]
        target_column = columns[target_idx][1]

        key_from = f"{source_table}.{source_column}"
        key_to = f"{target_table}.{target_column}"

        foreign_keys[key_from] = key_to
        foreign_keys[key_to] = key_from

        if source_column in table_columns[source_table]:
            table_columns[source_table][source_column]["foreign_key"] = key_to
        
        if target_column in table_columns[target_table]:
            table_columns[target_table][target_column]["foreign_key"] = key_from

    return {
        "db_id":db_id,
        "tables":table_columns,
        "foreign_keys":foreign_keys
    }

def extract_schema_info(db_info, schemas):
    group_result = {tab: {} for tab in schemas.keys()}
    foreign_keys = []

    for table_name, column_list in schemas.items():
        for column_name in column_list:
            column_data = copy.deepcopy(db_info['tables'][table_name][column_name])
            group_result[table_name][column_name] = column_data
    return group_result

def save_and_extract_schema_info(info_path, data_path: str, column_names: list, save_path: str):
    with open(info_path, "r") as j:
        infos = json.load(j)
        
    with open(data_path, "r") as j:
        data = json.load(j)
        
    new_data = copy.deepcopy(data)
    
    for i in range(len(data)):
        for col in column_names:
            try:
                new_data[i][col] = extract_schema_info(
                    db_info=infos[data[i]['db_id']],
                    schemas=data[i][col]
                )
            except:
                print(i)
            
    with open(save_path, "w") as j:
        json.dump(new_data, j, indent=4)
