from argparse import ArgumentParser
from utils.log import Logger
from utils.misc import ConfigParser


def main(opts):
    config = ConfigParser.parse_json_config(opts.json_path)

    logger = Logger(config["log_dir"], config["verbose"])

    logger.log_config(config)
    # TODO: trainer, dataloader

    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json_path",
        help="Path to YAML file containing configuration. See config/test.yaml",
        default="../config/test.yaml",
    )
    opts = parser.parse_args()
    evaluation_metrics = main(opts)
