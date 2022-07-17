from pathlib import Path
from pydoc import locate

import click
from omegaconf import OmegaConf
from git import Repo

from src.model_building import DenseNet
from src.training import Trainer
from src.utils import setup_mlflow_active_run, check_conditions
from src import DATA_FOLDER
from src.data_pipeline import AugmentedImageSequence, StepCalculator


@click.command()
@click.option('--conf', type=str, default='config.yaml')
@click.option('--debug', is_flag=True, default=False, help='if True, do not check for all-committed conditions or not.')
def main(conf: str, debug: bool):
    config_file_path = Path(conf)
    config = OmegaConf.load(config_file_path)
    data_dir = DATA_FOLDER.joinpath(config.dataset_name)
    run_dir = Path('run')

    repo = Repo()

    if not debug:
        check_conditions(repo, data_dir=data_dir)

    print(f'instantiating {config.dataset} and config.model ..')
    dataset = locate(config.dataset)(config)
    model_builder = locate(config.model)(config)

    trainer = Trainer(config, run_dir)

    root = Path(repo.working_tree_dir).name
    experiment_name = root + '/' + repo.active_branch.name

    active_run = setup_mlflow_active_run(config_path=config_file_path,
                                         session_type='train',
                                         experiment_name=experiment_name, colab=True)

    tr_gen, val_gen, n_tr, n_val = dataset.create_data()

    trainer.train(model_builder=model_builder,
                  train_data_gen=tr_gen,
                  n_iter_train=n_tr,
                  val_data_gen=val_gen,
                  n_iter_val=n_val,
                  active_run=active_run)


if __name__ == '__main__':
    main()
