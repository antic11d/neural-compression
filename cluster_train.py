from cluster import read_params_from_cmdline, save_metrics_params
import sys

sys.path.insert(0, ".")
sys.path.insert(1, "..")

from train import main

if __name__ == "__main__":
    params = read_params_from_cmdline(make_immutable=False)

    training_dict = main(params)

    save_metrics_params(
        {k: training_dict[k][-1] for k in training_dict.keys()},
        params,
    )
