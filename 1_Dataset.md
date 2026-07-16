---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: epoxy-inverse-design (3.12.12)
    language: python
    name: python3
---

```python
from ml_tools.ML_datasetmaster import DragonDataset as ChosenDataset
from ml_tools.utilities import load_dataframe_with_schema
from ml_tools.schema import FeatureSchema
from ml_tools.keys import TaskKeys

from paths import PM
```

```python
SCHEMA_PATH = PM.start_data
TRAIN_DATASET_INPUT_FILE = PM.imputed_file
TRAIN_DATASET_OUTPUT_DIR = PM.artifacts
```

## 1. Load Schema and Dataframe

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)

df, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_INPUT_FILE, 
                                   schema=schema)
```

## 2. Make Dataset Class

```python
dataset = ChosenDataset(pandas_df=df,
                        schema=schema,
                        kind=TaskKeys.REGRESSION,
                        feature_scaler="fit",
                        target_scaler="fit",
                        validation_size=0.1,
                        test_size=0,
                        random_state=42,
                        )
```

```python
dataset
```

## 3. Save Dataset Class

```python
dataset.save_dataset_bundle(directory=TRAIN_DATASET_OUTPUT_DIR)
```
