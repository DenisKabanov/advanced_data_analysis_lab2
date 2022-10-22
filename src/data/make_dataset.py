# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from preprocess import preprocess_data, extract_target
from sklearn.model_selection import train_test_split
from src.utils import save_as_pickle
import src.config as cfg


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_data_dir', type=click.Path())
@click.argument('output_target_dir', type=click.Path())

def main(input_dir = "data/raw/", output_data_dir="data/interim/", output_target_dir="data/processed/"):
    logger = logging.getLogger(__name__)
    logger.info('initial preprocess data set from raw data')


    for filename in os.listdir(input_dir):
        if filename.find("csv") != -1 and filename.find("dvc") == -1:
            df = pd.read_csv(input_dir + filename)
            df = preprocess_data(df)
            if cfg.TARGET_COLS in df.columns: # проверка, что хотя таргет есть в столбцах рассматриваемого датасета
                train, val = train_test_split(df, test_size=0.2, random_state=cfg.RS)
                
                train_data, train_target =  extract_target(train)
                val_data, val_target =  extract_target(val)

                save_as_pickle(train_data, output_data_dir + "train_data.pkl")
                save_as_pickle(val_data, output_data_dir + "val_data.pkl")
                save_as_pickle(train_target, output_target_dir + "train_target.pkl")
                save_as_pickle(val_target, output_target_dir + "val_target.pkl")

            else: # случай для обработки test.csv
                save_as_pickle(df, output_data_dir + "test_data.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
