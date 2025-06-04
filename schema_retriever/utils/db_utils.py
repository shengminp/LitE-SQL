import pandas as pd
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

@dataclass
class Questions:
    db_id: str # db_id
    question_id: int # question id
    question: str
    difficulty: str
    db_info: dict # corresponding database info.
    evidence: str = None # Hint for SQL

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
