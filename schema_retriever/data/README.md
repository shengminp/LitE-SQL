#### Preprocessing
Run the following command for preprocessing:
```
python preprocessing.py
```

After preprocessing, json files would be generated:
    
    .
    ├── BIRD_train_database_infos.json                        # For generating BIRD-train.json with example values
    ├── BIRD_dev_database_infos.json                          # For generating BIRD-dev.json with example values
    ├── BIRD-train.json                                       # For SQL Generator fine-tuning
    ├── fine-tuning_samples_from_BIRD_augmented_version.json  # For Schema Retriever fine-tuning
    └── README.md

- `BIRD-train-more-schema.json` is saved under `./sql_generator/bird/train/`
