import time
import random
import os
import json
import yaml


def get_dirname(base_dir):
    dirname = time.strftime(
        f"%y-%m-%d-%H%M%S-{random.randint(0, 1e6):06}", time.gmtime(time.time())
    )
    log_path = os.path.join(base_dir, dirname)

    return log_path


class ConfigParser:
    @staticmethod
    def parse_json_config(path):
        with open(path, "r") as f:
            config = yaml.load(f)

        return config
