import sys

sys.path.append("..")
from argparse import ArgumentParser
import torch
from src.utils.log import Logger
from src.utils.misc import ConfigParser
from src.models.conv_vq_vae import ConvolutionalVQVAE
from src.dataset.vctk_stream import VCTKFeaturesLoader
from src.eval.evaluator import Evaluator


def main(opts):
    config = ConfigParser.parse_yaml_config(opts.config_path)
    cuda_available = config["use_cuda"] and torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    model = ConvolutionalVQVAE(config, device).to(device)
    model.load_state_dict(
        torch.load(config["eval"]["pretrained_weights_path"], map_location=device)
    )
    datastream = VCTKFeaturesLoader(config["vctk_path"], config, cuda_available)
    evaluator = Evaluator(device, model, datastream, config)
    eval_dict = evaluator.evaluate()
    print(eval_dict["dtw_distance"])
    return eval_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="Path to YAML file containing configuration. See config/test.yaml",
        default="../config/test.yaml",
    )
    opts = parser.parse_args()
    evaluation_metrics = main(opts)
