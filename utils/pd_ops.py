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

def pivot_labels_to_columns(df, date_col='Date', label_col='label', value_col='value', aggfunc='mean'):
    """
    将 DataFrame 中的标签列和值列转换为以日期为行的透视表，列名格式为 "label-value"。

    参数:
    df (pd.DataFrame): 输入的 DataFrame，其中包含日期列、标签列和值列。
    date_col (str): 表示日期的列名，默认为 'Date'。
    label_col (str): 表示标签的列名，默认为 'label'。
    value_col (str): 表示值的列名，默认为 'value'。
    aggfunc (str or function): 用于聚合重复项的函数，默认为 'mean'。

    返回:
    pd.DataFrame: 转换后的 DataFrame，列名格式为 "label-value"。
    """
    # 处理重复项，进行聚合
    df_agg = df.groupby([date_col, label_col], as_index=False).agg({value_col: aggfunc})
    
    # 通过 groupby 和 unstack 进行透视
    grouped_df = df_agg.set_index([date_col, label_col])[value_col].unstack().reset_index()
    
    # 修改列名，生成所需的格式
    grouped_df.columns = [f"{col}-{value_col}" if col != date_col else col for col in grouped_df.columns]
    
    return grouped_df
