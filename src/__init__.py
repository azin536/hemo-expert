from pathlib import Path


PACKAGE_PATH = Path(__file__).absolute().resolve()
DATA_FOLDER = PACKAGE_PATH.parent.parent.joinpath('data')
