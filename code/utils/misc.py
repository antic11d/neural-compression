import time
import random
import os


def get_dirname(base_dir):
    dirname = time.strftime(
        f"%y-%m-%d-%H%M%S-{random.randint(0, 1e6):06}", time.gmtime(time.time())
    )
    log_path = os.path.join(base_dir, dirname)

    return log_path