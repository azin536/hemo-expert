from pathlib import Path

import click
import pandas as pd
from git import Repo

from src.data_ingestion import prepare_data
from src.utils import check_conditions
from src import DATA_FOLDER


@click.command()
@click.option('--meta-data', help='path to meta-data')
@click.option('--output-dir', help='where to write the prepared data')
@click.option('--debug', is_flag=True, default=False, help='if True, do not check for all-committed conditions or not.')
def main(meta_data: str, output_dir: str, debug: bool):
    md_df = pd.read_csv(meta_data)
    unique_datasources = md_df['DataSource'].unique().tolist()

    # Check if all data sources are added to the git
    if not debug:
        repo = Repo('.')
        for ds in unique_datasources:
            check_conditions(repo, DATA_FOLDER.joinpath(ds))

    prepare_data(meta_data_file=md_df,
                 output_dir=Path(output_dir))


if __name__ == '__main__':
    main()
