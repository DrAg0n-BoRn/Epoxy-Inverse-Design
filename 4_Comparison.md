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

# Comparison Range for the Training Data

```python
from ml_tools.utilities import load_dataframe, merge_dataframes, load_dataframe_greedy
from ml_tools.data_exploration import (filter_subset_continuous, 
                                       reconstruct_from_schema,
                                       plot_value_distributions_multi, 
                                       plot_numeric_overview_boxplot_macro, 
                                       summarize_dataframe)
from ml_tools.schema import FeatureSchema
from ml_tools.path_manager import list_csv_paths

from paths import PM
from helpers.constants import TARGET_RANGE, TARGET, TARGET_UNIT
```

```python
assert isinstance(TARGET_RANGE, (list, tuple)) and len(TARGET_RANGE) == 2, "TARGET_RANGE must be a list or tuple of length 2."
assert isinstance(TARGET, str) and TARGET, "TARGET must be a non-empty string."
assert isinstance(TARGET_UNIT, str) and TARGET_UNIT, "TARGET_UNIT must be a non-empty string."
```

## 1 Load Train Data and Feature Schema

```python
df, _ = load_dataframe(PM.imputed_file)
```

```python
schema = FeatureSchema.from_json(PM.start_data)
```

## Reconstruct categorical features from the schema

```python
df_reconstructed = reconstruct_from_schema(df=df, schema=schema, targets=[TARGET])
```

## Filter on the chosen range of the target variable

```python
df_range = filter_subset_continuous(df=df_reconstructed, 
                                    range_filters={TARGET: TARGET_RANGE},
                                    drop_filter_cols=True)
```

```python
summarize_dataframe(df_range)
```

## Plot Distributions

```python
plot_numeric_overview_boxplot_macro(df=df_range,
                                    save_dir=PM.comparison,
                                    plot_title=f"Train Data - {TARGET} {TARGET_RANGE[0]} to {TARGET_RANGE[1]}",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)
```

## 2 Load Diffusion Generated Data

```python
diffusion_csvs = list_csv_paths(PM.diffusion_generated)
```

```python
# merge all diffusion-generated CSVs into a single DataFrame
diffusion_dfs = []

for _, csv_path in diffusion_csvs.items():
    df_diffusion, _ = load_dataframe(csv_path)
    diffusion_dfs.append(df_diffusion)

df_diffusion = merge_dataframes(*diffusion_dfs, reset_index=True, direction="vertical")    
```

```python
summarize_dataframe(df_diffusion)
```

## Plot Distributions

```python
plot_numeric_overview_boxplot_macro(df=df_diffusion,
                                    save_dir=PM.comparison,
                                    plot_title=f"Diffusion Generated - {TARGET} {TARGET_RANGE[0]} to {TARGET_RANGE[1]}",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)
```

## 3 Load Inverse Design Generated Data

```python
df_inverse_raw = load_dataframe_greedy(PM.optimization)
```

```python
df_inverse = filter_subset_continuous(df=df_inverse_raw, 
                                    range_filters={TARGET: (TARGET_RANGE[0], None)},
                                    drop_filter_cols=True)
```

```python
summarize_dataframe(df_inverse)
```

## Plot Distributions

```python
plot_numeric_overview_boxplot_macro(df=df_inverse,
                                    save_dir=PM.comparison,
                                    plot_title=f"Inverse Design Generated - {TARGET} {TARGET_RANGE[0]}+",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)
```

```python
#TODO: Must have same column names and dtypes (cast int to float if necessary)
named_datasets = {f"Train Data ({TARGET_RANGE[0]}-{TARGET_RANGE[1]} {TARGET_UNIT})": df_range, 
                  f"Diffusion Generated ({TARGET_RANGE[0]}-{TARGET_RANGE[1]} {TARGET_UNIT})": df_diffusion,
                  f"Inverse Design Generated ({TARGET_RANGE[0]}+ {TARGET_UNIT})": df_inverse} 
```

## Plot Comparison Distributions

```python
plot_value_distributions_multi(named_dataframes=named_datasets,
                               save_dir=PM.comparison,
                               font_scaling=1.5,
                               mode="percentage")
```
