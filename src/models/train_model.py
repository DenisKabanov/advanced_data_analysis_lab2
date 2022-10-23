# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
from src.models.model import model, preprocess_pipe
import pandas as pd


@click.command()
@click.argument('input_train_data_filepath', type=click.Path(exists=True))
@click.argument('input_train_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_dir', type=click.Path())
def main(input_train_data_filepath = "data/processed/train_data.pkl", input_train_target_filepath = "data/processed/train_target.pkl", output_model_dir = "models/"):
    logger = logging.getLogger(__name__)
    logger.info('training model...')


    train = pd.read_pickle(input_train_data_filepath)
    target = pd.read_pickle(input_train_target_filepath)

    Transformer = preprocess_pipe.fit(train)

    save_as_pickle(Transformer, output_model_dir + "transformer.pkl")

    model.fit(Transformer.transform(train), target,
            metric_period=10, 
            plot=True,
    )

    save_as_pickle(model, output_model_dir + "model_fit.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()