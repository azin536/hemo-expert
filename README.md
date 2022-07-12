# samlple-exp-repository
Sample repository for using as a blue-print in ML experimentation projects, which is using DVC for versioning and MLFlow for tracking.


# Import Data
1. initialize DVC in the repository -> `dvc init`
2. install DVC `post-checkout`, `pre-commit`, `pre-push` -> `dvc install`
3. list the tracked data in the data registry -> `dvc list https://github.com/iamsoroush/dvc-minio-data-registry --rev main`
4. import the datasource(s) -> `dvc import https://github.com/iamsoroush/dvc-minio-data-registry.git "datasources/RSNA" -o data/`, or update the dataset to the latest version -> `dvc update datasources/pacs.dvc`
5. import the task meta-data -> `dvc import https://github.com/iamsoroush/dvc-minio-data-registry.git "tasks/hemo" -o data/hemo`
6. add and commit dvc files in the data folder -> `git add  data/.gitignore data/*.dvc; git commit data/.gitignore data/*.dvc -m "add RSNA datasource and hemo meta-data"`

If the `.dvc` is currently in the repository, you can just install DVC and pull the data related to this version of the repository:
```shell
dvc pull
```

If the data is available on the same system, use shared `cache`:
1. `dvc cache dir ~/samlple-exp-repository/.dvc/cache/`
2. `dvc config cache.shared group`
3. `dvc config cache.type symlink`
4. `dvc checkout`


# Run an experiment
1. prepare the data: `python prepare.py --meta-data data/hemo/meta-data.csv --output-dir data/hemo`
2. train: `python train.py --conf config.yaml`
3. evaluate: `python evaluate.py --conf config.yaml`
4. export: `python export.py --conf config.yaml`


# Run Pipeline
```shell
python pipeline.py --no-pylint run --meta-data "data/hemo/meta-data.csv" --output-dir "data/hemo" --prepare --train --evaluate --export
```

