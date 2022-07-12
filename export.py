from pathlib import Path

import click
from git import Repo
from omegaconf import OmegaConf

from src.training import Exporter
from src.utils import setup_mlflow_active_run, check_conditions
from src.inference import MyHemoModel


DATA_DIR = Path('data')


@click.command()
@click.option('--conf', type=str, default='config.yaml')
@click.option('--debug', is_flag=True, default=False, help='if True, do not check for all-committed conditions.')
def main(conf: str, debug: bool):
    config_file_path = Path(conf)
    config = OmegaConf.load(config_file_path)
    data_dir = DATA_DIR.joinpath(config.dataset_name)
    run_dir = Path('run')

    repo = Repo('.')

    if not debug:
        check_conditions(repo, data_dir=data_dir)

    exporter = Exporter(config=config, run_dir=run_dir)

    root = Path(repo.working_tree_dir).name
    experiment_name = root + '/' + repo.active_branch.name

    active_run = setup_mlflow_active_run(config_path=config_file_path,
                                         session_type='export',
                                         experiment_name=experiment_name)

    # pyfunc_model_class = f'src.{config.exporter.pyfunc_model}'
    # pyfunc_model = locate(pyfunc_model_class)()
    # print(f'exporting the {pyfunc_model_class} to run {active_run.info.run_id} ..')

    pyfunc_model = MyHemoModel()
    exporter.log_model_to_mlflow(active_run=active_run,
                                 pyfunc_model=pyfunc_model,
                                 config_path=config_file_path)


if __name__ == '__main__':
    main()
