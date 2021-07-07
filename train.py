from argparse import ArgumentParser
import torch
from src.utils.log import Logger
from src.utils.misc import ConfigParser
from src.models.conv_vq_vae import ConvolutionalVQVAE
from src.trainers.conv_trainer import ConvolutionalTrainer
from src.dataset.vctk_stream import VCTKFeaturesLoader


def main(opts):
    config = ConfigParser.parse_yaml_config(opts.config_path)
    device = "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu"

    logger = Logger(config["log_dir"], config["verbose"])
    config["log_dir"] = str(logger.root_dir)

    logger.log_config(config)
    if config["decoder_type"] == "deconv":
        model = ConvolutionalVQVAE(config, device).to(device)
    else:
        raise NotImplementedError(
            "Decoder of type {} not implemented.".format(config["decoder_type"])
        )

    trainer = ConvolutionalTrainer(
        model=model, device=device, configuration=config, optimizer_name="adam"
    )

    datastream = VCTKFeaturesLoader(config["vctk_path"], config, False)

    training_info = trainer.train(datastream)

    # TODO: Evaluation
    evaluation_info = {}

    return {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="Path to YAML file containing configuration. See config/test.yaml",
        default="../config/test.yaml",
    )
    opts = parser.parse_args()
    evaluation_metrics = main(opts)
