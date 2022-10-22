import pandas as pd
import numpy as np

def lot_size(df: pd.DataFrame) -> pd.DataFrame:
    df["House size"] = 0
    for row in df.index:
        if df["LotArea"][row] < 1000:
            df["House size"][row] = 1
        elif df["LotArea"][row] < 5000:
            df["House size"][row] = 2
        elif df["LotArea"][row] < 10000:
            df["House size"][row] = 3
        elif df["LotArea"][row] < 50000:
            df["House size"][row] = 4
        else:
            df["House size"][row] = 5
    df["House size"] = df["House size"].astype(np.int8)
    return df

def how_good_house(df: pd.DataFrame) -> pd.DataFrame:
    df["How good is the house"] = 0
    for row in df.index:
        df["How good is the house"][row] = (df["OverallQual"][row] + df["OverallCond"][row]) * df["House size"][row]
    df["How good is the house"] = df["How good is the house"].astype(np.int8)
    return df