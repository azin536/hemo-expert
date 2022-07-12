from pathlib import Path
import typing

import yaml
import mlflow
from mlflow.entities import ViewType
from git import Repo


MLFLOW_TRACKING_URI = "https://mlflow.aimedic.co"
MLFLOW_TRACKING_URI_COLAB = "http://185.110.190.127:7080/"


def setup_mlflow_active_run(session_type: str,
                            experiment_name: str,
                            config_path: typing.Optional[Path] = None,
                            colab: bool = False):
    """setup the MLFlow

    Args:
        config_path: path to the .yaml config file
        session_type: (train, eval, export)
        experiment_name: experiment name on the mlflow server
        colab: if True, use the proxy server for logging to MLFlow
    """

    tracking_uri = MLFLOW_TRACKING_URI
    if colab:
        tracking_uri = MLFLOW_TRACKING_URI_COLAB

    mlflow.end_run()
    active_run = _setup_mlflow(mlflow_experiment_name=experiment_name,
                               mlflow_tracking_uri=tracking_uri)

    mlflow.set_tag("session_type", session_type)  # ['hpo', 'evaluation', 'training']

    try:
        # config = load_config_as_dict(path=config_path)
        # _add_config_file_to_mlflow(config)
        mlflow.log_artifact(str(config_path))
    except Exception as e:
        print(f'exception when logging config file to mlflow: {e}')

    return active_run


def get_mlflow_run(parent_exp_id: str,
                   parent_run_id: str,
                   colab: bool = False):
    """set config_path if you want to log the config and set sessio_type"""

    tracking_uri = MLFLOW_TRACKING_URI
    if colab:
        tracking_uri = MLFLOW_TRACKING_URI_COLAB

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiments = client.list_experiments(view_type=ViewType.ALL)
    if parent_exp_id not in [i.experiment_id for i in experiments]:
        raise Exception(f'experiment {parent_exp_id} does not exist.')
    else:
        experiment = [i for i in experiments if i.experiment_id == parent_exp_id][0]
        if experiment.lifecycle_stage != 'active':
            print(f'experiment {experiment.name} exists but is not active, restoring ...')
            client.restore_experiment(experiment.experiment_id)
            print(f'restored {experiment.name}')

    mlflow.set_tracking_uri(tracking_uri)
    active_run = mlflow.start_run(run_id=parent_run_id,
                                  experiment_id=parent_exp_id)

    return active_run


def _setup_mlflow(mlflow_experiment_name: str,
                  mlflow_tracking_uri: str) -> mlflow.ActiveRun:
    """Sets up mlflow and returns an ``active_run`` object.

    tracking_uri/
        experiment_id/
            run1
            run2
            ...

    Args:
        mlflow_tracking_uri: ``tracking_uri`` for mlflow
        mlflow_experiment_name: ``experiment_name`` for mlflow, use the same ``experiment_name`` for all experiments
        related to the same task, i.e. the repository name.

    Returns:
        active_run: an ``active_run`` object to use for mlflow logging.

    """

    client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)
    experiments = client.list_experiments(view_type=ViewType.ALL)
    if mlflow_experiment_name not in [i.name for i in experiments]:
        print(f'creating a new experiment: {mlflow_experiment_name}')
        experiment_id = client.create_experiment(name=mlflow_experiment_name)
        experiment = client.get_experiment(experiment_id)
    else:
        experiment = [i for i in experiments if i.name == mlflow_experiment_name][0]
        if experiment.lifecycle_stage != 'active':
            print(f'experiment {mlflow_experiment_name} exists but is not active, restoring ...')
            client.restore_experiment(experiment.experiment_id)
            print(f'restored {mlflow_experiment_name}')

    print(f'Exp ID: {experiment.experiment_id}')
    print(f'Exp Name: {experiment.name}')
    print(f'Exp Artifact Location: {experiment.artifact_location}')
    print(f'Exp Tags: {experiment.tags}')
    print(f'Exp Lifecycle Stage: {experiment.lifecycle_stage}')
    # if experiment is not None:
    #     experiment_id = experiment.experiment_id
    # else:
    #     experiment_id = mlflow.create_experiment(mlflow_experiment_name)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return mlflow.start_run(experiment_id=experiment.experiment_id)
    # active_run = mlflow.start_run(experiment_id=experiment.experiment_id)


def _add_config_file_to_mlflow(config_dict: dict):
    """Adds parameters from config file to mlflow.

    Args:
        config_dict: config file as nested dictionary
    """

    def param_extractor(dictionary):

        """Returns a list of each item formatted like 'trainer.mlflow.tracking_uri: /tracking/uri' """

        values = []
        if dictionary is None:
            return values

        for key, value in dictionary.items():
            if isinstance(value, dict):
                items_list = param_extractor(value)
                for i in items_list:
                    values.append(f'{key}.{i}')
            else:
                values.append(f'{key}: {value}')
        return values

    fields_to_ignore = ['model_details', 'model_parameters', 'considerations']
    new_config = {k: v for k, v in config_dict.items() if k not in fields_to_ignore}
    str_params = param_extractor(new_config)
    params = {}
    for item in str_params:
        name = f"config_{item.split(':')[0]}"
        item_value = item.split(': ')[-1]

        params[name] = item_value

    mlflow.log_params(params)


def load_config_as_dict(path: Path) -> dict:
    """
    loads the ``yaml`` config file and returns a dictionary

    Args:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations, for example ``config.model.optimizer.lr``

    """

    with open(path) as f:
        data_map = yaml.safe_load(f)
    return data_map


def check_conditions(repo: Repo,
                     data_dir: Path):

    if repo.is_dirty():
        # print(f'there are uncommitted changes:')
        # print([item.a_path for item in repo.index.diff("HEAD")])
        raise Exception('there are uncommitted changes. commit all the changed and try again.')
    print('repo is clean: PASSED')

    data_dvc_file = data_dir.parent.joinpath(f'{data_dir.name}.dvc')

    if not data_dvc_file.exists():
        raise Exception(f'you are using {data_dir.name} dataset, and the .dvc can not be found in the {data_dir.parent}. make sure that your data is being version-controlled by dvc -> add data/{data_dir.name}')
    print(f'dataset {data_dir.name} is being version controlled by dvc: PASSED')

    if str(data_dvc_file) in repo.untracked_files:
        raise Exception(f'{data_dvc_file.name} is not being tracked, add and commit to continue.')
    print(f'{data_dvc_file} is being tracked.')
