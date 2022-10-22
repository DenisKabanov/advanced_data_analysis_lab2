import src.config as cfg
import numpy as np
import pandas as pd

def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame: # в качестве индексов выставляем id
    if idx_col in df.columns:
        df.set_index(idx_col, inplace=True)
    return df

# def fill_other_missed_values(df: pd.DataFrame) -> pd.DataFrame: # помимо одного пропущенного значения 'Пол', в данных имеется 534 пропуска в 'Возраст курения', 546 в 'Сигарет в день', 732 в 'Частота пасс кур', 167 в 'Возраст алког'
#     for col in cfg.MANY_MISSING_DATA_COLS: # заполним пропуски медианой
#         med = df[col].median()
#         df[col].fillna(med, inplace=True)
#     return df
        

def cast_types(df: pd.DataFrame) -> pd.DataFrame: # каст типов
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    df[cfg.ONE_COLS] = df[cfg.ONE_COLS].astype(np.int32)
    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    if cfg.TARGET_COLS in df.columns: # если таргет есть в переданном DataFrame, то и их тоже кастим
        df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int32)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = cast_types(df)
    return df


def extract_target(df: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target