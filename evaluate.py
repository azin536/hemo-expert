from pathlib import Path
from pydoc import locate

import click
import tensorflow.keras as tfk
from omegaconf import OmegaConf
from git import Repo

from src.data_pipeline import DataLoader
from src.training import get_checkpoints_info
from src.evaluation import Evaluator
from src.utils import setup_mlflow_active_run, check_conditions
import src.eval_metrics
from src.base import Metric
from pydoc import locate
from src.eval_metrics import EvaluationMetrics


DATA_DIR = Path('data')


@click.command()
@click.option('--conf', type=str, default='config.yaml')
@click.option('--debug', is_flag=True, default=False, help='if True, do not check for all-committed conditions or not.')
def main(conf: str, debug: bool):
    config_file_path = Path(conf)
    config = OmegaConf.load(config_file_path)
    data_dir = DATA_DIR.joinpath(config.dataset_name)
    run_dir = Path('run')

    repo = Repo('hemo-expert')

    if not debug:
        check_conditions(repo, data_dir=data_dir)

    dataset = locate(config.dataset)(config)
    eval_gen, n_eval = dataset.create_eval_data()

    checkpoints = get_checkpoints_info(run_dir.joinpath('checkpoints'))
    selected_model = min(checkpoints, key=lambda x: x['value'])
    model = tfk.models.load_model(selected_model['path'])

    root = Path(repo.working_tree_dir).name
    experiment_name = root + '/' + repo.active_branch.name

    active_run = setup_mlflow_active_run(config_path=config_file_path,
                                         session_type='eval',
                                         experiment_name=experiment_name, colab = True)

    eval_metrics = [EvaluationMetrics]
    # for name, cls in inspect.getmembers(src.eval_metrics, lambda o: inspect.isclass(o) and issubclass(o, Metric)):
    #     if name != 'Metric':
    #         eval_metrics.append(cls())

    evaluator = Evaluator(eval_metrics=eval_metrics)
    evaluator.run(model, eval_gen, n_eval, active_run)


if __name__ == '__main__':
    main()
