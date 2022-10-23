# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pandas as pd


@click.command()
@click.argument('input_pred_filepath', type=click.Path(exists=True))
@click.argument('input_true_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_pred_filepath = "results/val_target_predicted.pkl", input_true_filepath = "data/processed/val_target.pkl", output_dir = "results/"):
    logger = logging.getLogger(__name__)
    logger.info('model evaluation...')

    true = pd.read_pickle(input_true_filepath)
    pred = pd.read_pickle(input_pred_filepath)

    metrics = {'MAPE': mean_absolute_percentage_error(true, pred), 'MSE': mean_squared_error(true, pred)}
    with open(output_dir + "metrics.txt", "w") as f:
        f.write(f"MAPE: {metrics['MAPE']}\n")
        f.write(f"MSE: {metrics['MSE']}\n")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()