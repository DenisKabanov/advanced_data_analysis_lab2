import src.config as cfg
import numpy as np
import pandas as pd

def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame: # в качестве индексов выставляем id
    if idx_col in df.columns:
        df.set_index(idx_col, inplace=True)
    return df

def work_with_NA(df: pd.DataFrame) -> pd.DataFrame:
    columns_with_NA = df.loc[:, df.isnull().any()].columns
    df[columns_with_NA].fillna( "-1", inplace=True)
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame: # каст типов
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    for col in cfg.INT_COLS:
        for row in df.index:
            if pd.isna(df[col][row]):
                df[col][row] = -1
    df[cfg.INT_COLS] = df[cfg.INT_COLS].astype(np.int32)
    df[cfg.FLOAT_COLS] = df[cfg.FLOAT_COLS].astype(np.float32)
    if cfg.TARGET_COLS in df.columns: # если таргет есть в переданном DataFrame, то и их тоже кастим
        df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int32)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = work_with_NA(df)
    df = cast_types(df)
    return df


def extract_target(df: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target