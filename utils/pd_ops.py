import os
from typing import List, Tuple, Callable, Union
import pandas as pd

def reset_copy(df:pd.DataFrame):
    result = df.copy()
    result.reset_index(drop=True, inplace=True)
    return result

def filter_dataframe(df:pd.DataFrame, pattern:List[Tuple]):
    filtered_df = df
    for col, val in pattern:
        filtered_df = filtered_df[filtered_df[col] == val]
    return filtered_df

def filter_common(patterns:List[Tuple]):
    common_values = set(patterns[0][0][patterns[0][1]])
    for df, col in patterns[1:]:
        common_values = common_values.intersection(df[col])
    filtered_dfs = [reset_copy(df[df[col].isin(common_values)]) for df, col in patterns]
    return filtered_dfs

def join_common(patterns:List[Tuple]):
    merged_df = patterns[0][0]
    for df, col in patterns[1:]:
        merged_df = merged_df.merge(df, on=col)
    return merged_df

def map_dict(df:pd, key:str, value:str, value_func:Callable=lambda v: v):
    keys = df[key]
    values = df[value]
    result_dict = dict(zip(keys, values))
    for k, v in result_dict.items():
        result_dict[k] = value_func(v)
    return result_dict

def convert_date_to_float(df:pd.DataFrame, col_name:str):
    df = df.copy()
    df[col_name] = pd.to_datetime(df[col_name])
    min_date = df[col_name].min()
    df[col_name] = (df[col_name] - min_date).dt.days.astype(float)
    return df    

def group_sample(group):
    return group.sample(n=1)