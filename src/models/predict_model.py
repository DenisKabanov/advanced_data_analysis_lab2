# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import GridSearchCV
from src.utils import save_as_pickle
import pandas as pd


@click.command()
@click.argument('input_data_dir', type=click.Path(exists=True))
@click.argument('input_model_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_data_dir = "data/processed/", input_model_dir = "models/", output_dir = "results/"):
    logger = logging.getLogger(__name__)
    logger.info('model inference...')

    Transformer = pd.read_pickle(input_model_dir + "transformer.pkl")
    model = pd.read_pickle(input_model_dir + "model_fit.pkl")

    val_data = pd.read_pickle(input_data_dir + "val_data.pkl")
    test_data = pd.read_pickle(input_data_dir + "test_data.pkl")
    
    val_pred = model.predict(Transformer.transform(val_data))
    save_as_pickle(val_pred, output_dir + "val_target_predicted.pkl")

    test_pred = model.predict(Transformer.transform(test_data))
    save_as_pickle(test_pred, output_dir + "test_target_predicted.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()