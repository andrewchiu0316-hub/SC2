from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)
