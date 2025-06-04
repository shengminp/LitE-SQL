import os, re, json
import logging
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from langchain.schema.document import Document
from langchain_chroma import Chroma

from schema_retriever.utils.utils import get_logger
import schema_retriever.utils.configs as cfg
from schema_retriever.utils.db_utils import Questions, load_tables_description, apply_original_casing, save_and_extract_schema_info
from schema_retriever.language_model.langauge_model import CustomEmbeddings

class SchemaRetriever:
    def __init__(self, args):
        self.ROOT_PATH = args.root_path # "BIRD/dev_20240627"
        self.DB_SCHEMA_INFO_PATH = args.db_schema_info_path # "dev_tables.json"
        self.DATA_PATH = args.data_path # "dev.json"
        self.DATABASE_PATH = args.database_dir_path # dev_databases
        self.EMBEDDING_MODEL_PATH = cfg.EMBEDDING_MODEL_PATH
        self.TOP_K = args.SL_K # Top k, 25
        
        self.device = torch.device("cuda")
        
        self._initialize()
        
    def _initialize(self):
        logging.info("initialize..(0/3)")
        self._get_datasets()
        logging.info("Finished load datasets (1/3)")
        self._get_embedding_model()
        logging.info("Finished load pre-trained embedding model (2/3)")
        self._create_vector_database()
        logging.info("Finished create vector database (3/3)")
        logging.info("initialize done..")
        
    def _get_datasets(self):
        with open(os.path.join(self.ROOT_PATH, self.DB_SCHEMA_INFO_PATH), "r") as j:
            self.db_infos = json.load(j)
            
        with open(os.path.join(self.ROOT_PATH, self.DATA_PATH), "r") as j:
            self.datas = json.load(j)        
        
        self.datasets = [
            Questions(
                db_id=data['db_id'],
                question_id=data['question_id'],
                question=data['question'],
                difficulty=data['difficulty'],
                evidence=data.get('evidence', ''),
                db_info=next(
                    (
                        db_info 
                        for db_info in self.db_infos 
                        if db_info["db_id"] == data["db_id"]), 
                    None
                ),
            )
            for data in self.datas
        ]
            
    def _get_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_PATH, torch_dtype="auto")
        model = AutoModel.from_pretrained(self.EMBEDDING_MODEL_PATH, torch_dtype="auto")
        model.eval()
        
        self.embedding_function = CustomEmbeddings(model, tokenizer, device=self.device)
        
    def _create_vector_database(self):
        db_names = os.listdir(os.path.join(self.ROOT_PATH, self.DATABASE_PATH))
        
        docs = []
        
        for db in db_names:
            DB_PATH = os.path.join(os.path.join(self.ROOT_PATH, self.DATABASE_PATH), db)
            table_description = load_tables_description(DB_PATH, True)
                
            for table_name, columns in table_description.items():
                for column_name, column_info in columns.items():
                    metadata = {
                        "database_name": db,
                        "table_name": table_name,
                        "original_column_name": column_name,
                        "column_name": column_info.get('column_name', ''),
                        "column_description": column_info.get('column_description', ''),
                        "value_description": column_info.get('value_description', '')
                    }
                    
                    page_content = f"<table>{table_name}</table> <column>{column_name}</column> "
                    if column_info.get("column_description", "").strip():
                        page_content += f"<column description>{column_info['column_description']}</column_description> "
                    if column_info.get("value_description", "").strip():
                        page_content += f"<value description>{column_info['value_description']}</value description>"
                    docs.append(Document(page_content=page_content, metadata=metadata))
        
        # Save vector database on root path (e.g., "BIRD/dev_20240627/")
        self.SAVE_DB_PATH = os.path.join(self.ROOT_PATH, 'vector_db')
        
        if os.path.exists(self.SAVE_DB_PATH):
            os.system(f"rm -r {self.SAVE_DB_PATH}")
            
        Chroma.from_documents(
            documents=docs, 
            embedding=self.embedding_function,
            persist_directory=self.SAVE_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def retrieve_topk_columns(self, vector_db, sample, method="similarity", topk=10):
        query = f"<question>{sample.question}</question> <evidence>{sample.evidence}</evidence>"
        
        if len(vector_db.get()['documents']) < topk:
            topk = len(vector_db.get()['documents'])
            
        kwargs = {"k": topk, "filter": {"database_name": sample.db_id}}
        
        retreiver = vector_db.as_retriever(
            search_type=method,
            search_kwargs=kwargs
        )
        
        results = retreiver.invoke(query)
        
        qtc_dict = {}
        page_contents = []
        
        for doc in results:
            if doc.metadata["table_name"] not in qtc_dict.keys():
                qtc_dict[doc.metadata["table_name"]] = []
                
            qtc_dict[doc.metadata["table_name"]].append(doc.metadata["original_column_name"])
            page_contents.append(doc.page_content)
        
        return {k: v for k, v in qtc_dict.items() if v}, page_contents
    
    def run(self) -> list:
        """Schema Retriever:

        Returns:
            list: retrieved_data = [
                {
                    'question',
                    'evidence',
                    'queried_schemas_top{self.TOP_K}',
                    '{self.TOP_K}_contents',
                },
                {}, {}, ...
            ]
        """
        # Load saved vector database
        vector_db = Chroma(
            persist_directory=os.path.join(self.SAVE_DB_PATH), 
            embedding_function=self.embedding_function
            )
        
        retrieved_data = []
        for sample in tqdm(self.datasets):
            q_sample = dict()
            
            q_sample['question_id'] = sample.question_id
            q_sample['db_id'] = sample.db_id
            q_sample['difficulty'] = sample.difficulty
            q_sample['question'] = sample.question
            q_sample['evidence'] = sample.evidence
            ## Retrieve topk columns for each question.
            q_sample[f'queried_schemas_top{self.TOP_K}'], q_sample[f'top{self.TOP_K}_contents'] = \
                self.retrieve_topk_columns(vector_db, sample, topk=self.TOP_K)

            retrieved_data.append(q_sample)
            
        for i, sample in enumerate(retrieved_data):
            retrieved_data[i][f'queried_schemas_top{self.TOP_K}'] = apply_original_casing(retrieved_data[i][f'queried_schemas_top{self.TOP_K}'], self.datasets[i].db_info)
            
        return retrieved_data

os.system('clear')

logger = get_logger()

def main():
    parser.add_argument(
        "--root_path", 
        type=str, 
        help="[PATH] Define a data sample root directory, e.g., BIRD/dev_20240627"
        )
    parser.add_argument(
        "--db_schema_info_path", 
        type=str, 
        help="[PATH] Define a database schema path, e.g., dev_tables.json"
        )
    parser.add_argument(
        "--data_path",
        type=str, 
        help="[PATH] Define a dataset path, e.g., dev.json"
        )
    parser.add_argument(
        "--database_dir_path", 
        type=str, 
        help="[PATH] Define the saved database directory, e.g., dev_databases"
        )
    # For schema linking
    parser.add_argument(
        "--SL_K", 
        type=int,
        default=25,
        help="[SCHEMA] Define k for # of retrieval columns, e.g., 25"
        )

    args = parser.parse_args()
    
    schemalinker = SchemaRetriever(args=args)
    
    logger.info("📢 Now Schema Linker running..")
    retrieved_columns = schemalinker.run()

    with open("./schema_retriever/data/BIRD-dev.json", "w") as j:
        json.dump(retrieved_columns, j, indent=4)
    
    logger.info("📢 Saving Retrieved Information..")
    save_and_extract_schema_info(
        info_path = "./schema_retriever/data/BIRD_dev_database_infos.json"),
        data_path = "./schema_retriever/data/BIRD-dev.json",
        column_names = [
            f"queried_schemas_top{args.SL_K}"
        ]
        save_path = "./sql_generator/dev_20240627/retrieved/BIRD-dev-more-schema.json"
    )

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()   
