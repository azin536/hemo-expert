dvc init
dvc install
dvc import https://github.com/iamsoroush/dvc-minio-data-registry.git "datasources/RSNA" -o data/
dvc import https://github.com/iamsoroush/dvc-minio-data-registry.git "tasks/hemo" -o data/hemo