from dtw import dtw
from numpy.linalg import norm
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

TYPE_VALID = 'Validation set'
TYPE_TRAIN = 'Training set'

class Evaluator(object):
    def __init__(self, device, model, data_stream, configuration):
        self._device = device
        self._model = model
        self._data_stream = data_stream
        self._configuration = configuration
        # self._vctk = VCTK(
        #     self._configuration["data_root"],
        #     ratio=self._configuration["train_val_split"],
        # )
        # self._results_path = results_path
        # self._experiment_name = experiment_name

    def evaluate(self):
        self._model.eval()
        evaluation_entry = self._evaluate_once()

        dtw_distance = self._dtw_distance(
            evaluation_entry["valid_originals"],
            evaluation_entry["valid_reconstructions"],
        )
        encoding_indices_dim = evaluation_entry['encoding_indices'].shape[0]
        embedding_dim = evaluation_entry['encodings'].shape[-1]
        categories = list(range(embedding_dim))

        train_empirical_probabilities = self.calculate_empirical_probs(
            categories=categories, type=TYPE_TRAIN, encoding_indices_dim=encoding_indices_dim,
            NUM_EVALS=1_000)

        valid_empirical_probabilities = self.calculate_empirical_probs(
            categories=categories, type=TYPE_VALID, encoding_indices_dim=encoding_indices_dim,
            NUM_EVALS=1_000)[0]

        baseline = encoding_indices_dim * np.log2(embedding_dim)
        entropy = self.calculate_entropy(train_empirical_probabilities)
        ic = self.calculate_ic(valid_empirical_probabilities)

        print("Baseline: ", baseline)
        print("Entropy: ", entropy)
        print("IC: ", ic)

        self.plot_train_probabilities(train_empirical_probabilities, categories)
        self.plot_valid_probabilities(valid_empirical_probabilities, categories)

        # TODO: Add comparison plotting
        evaluation_entry["dtw_distance"] = dtw_distance
        return evaluation_entry

    def calculate_entropy(self, train_empirical_probabilities):
        neg_logs = - np.log2(train_empirical_probabilities + 1e-6)
        return np.multiply(train_empirical_probabilities, neg_logs).sum(axis=-1).sum()

    def calculate_ic(self, valid_empirical_probabilities):
        return (-np.log2(valid_empirical_probabilities + 1e-6)).sum()

    def _dtw_distance(self, mfcc1, mfcc2):
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
        return dist

    def calculate_empirical_probs(self, categories, type, encoding_indices_dim=24, NUM_EVALS=1000):
        self._model.eval()
        buffer = torch.zeros((encoding_indices_dim, NUM_EVALS))

        if type == TYPE_TRAIN:
            iterator = iter(self._data_stream.training_loader)
        elif type == TYPE_VALID:
            iterator = iter(self._data_stream.validation_loader)
        else:
            raise Exception("Unsupported type")

        for i in range(NUM_EVALS):
            evaluation_entry = self._evaluate_once(iterator=iterator)
            buffer[:, i] = evaluation_entry["encoding_indices"].view(1, encoding_indices_dim)

        def hist_1d(arr, categories):
            return np.array([(arr == c).sum() for c in categories])

        counts = np.apply_along_axis(hist_1d, axis=1, arr=buffer.numpy(), categories=categories)
        if type == TYPE_VALID:
            counts = counts.sum(axis=0, keepdims=True)
        empirical_probabilities = counts / counts.sum(axis=-1).reshape(-1, 1)
        return empirical_probabilities

    def plot_train_probabilities(self, empirical_probabilities, categories):
        indices = random.sample(range(empirical_probabilities.shape[0]), 3)

        sample_1 = empirical_probabilities[indices[0]]
        sample_2 = empirical_probabilities[indices[1]]
        sample_3 = empirical_probabilities[indices[2]]

        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches(15, 10)

        axs[0].bar(categories, height=sample_1, alpha=0.5, label=f"Estimated probabilities for encoding {indices[0]}")
        axs[0].legend()
        axs[0].set_xticks(categories)

        axs[1].bar(categories, height=sample_2, alpha=0.5, label=f"Estimated probabilities for encoding {indices[1]}")
        axs[1].legend()
        axs[1].set_xticks(categories)

        axs[2].bar(categories, height=sample_3, alpha=0.5, label=f"Estimated probabilities for encoding {indices[2]}")
        axs[2].legend()
        axs[2].set_xticks(categories)

        fig.savefig(f'train.png', dpi=fig.dpi)
        # plt.show()

    def plot_valid_probabilities(self, empirical_probaiblities, categories):
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(10, 7)

        axs.bar(categories, empirical_probaiblities, alpha=0.5, color='r')
        axs.set_xticks(categories)

        fig.savefig(f'valid.png', dpi=fig.dpi)
        # plt.show()

    def _evaluate_once(self, iterator=None):
        self._model.eval()

        if iterator is None:
            iterator = iter(self._data_stream.validation_loader)

        data = next(iterator)

        preprocessed_audio = data["preprocessed_audio"].to(self._device)
        valid_originals = data["input_features"].to(self._device)
        speaker_ids = data["speaker_id"].to(self._device)
        target = data["output_features"].to(self._device)
        wav_filename = data["wav_filename"]
        shifting_time = data["shifting_time"].to(self._device)
        preprocessed_length = data["preprocessed_length"].to(self._device)

        valid_originals = valid_originals.permute(0, 2, 1).contiguous().float()
        batch_size = valid_originals.size(0)
        target = target.permute(0, 2, 1).contiguous().float()
        wav_filename = wav_filename[0][0]

        with torch.no_grad():
            z = self._model.encoder(valid_originals)
            z = self._model.pre_vq_conv(z)
            (
                _,
                quantized,
                _,
                encodings,
                distances,
                encoding_indices,
                _,
                encoding_distances,
                embedding_distances,
                frames_vs_embedding_distances,
                concatenated_quantized,
            ) = self._model.vq(z)
            valid_reconstructions = self._model.decoder(
                quantized, self._data_stream.speaker_dic, speaker_ids
            )[0]

        return {
            "preprocessed_audio": preprocessed_audio,
            "valid_originals": valid_originals,
            "speaker_ids": speaker_ids,
            "target": target,
            "wav_filename": wav_filename,
            "shifting_time": shifting_time,
            "preprocessed_length": preprocessed_length,
            "batch_size": batch_size,
            "quantized": quantized,
            "encodings": encodings,
            "distances": distances,
            "encoding_indices": encoding_indices,
            "encoding_distances": encoding_distances,
            "embedding_distances": embedding_distances,
            "frames_vs_embedding_distances": frames_vs_embedding_distances,
            "concatenated_quantized": concatenated_quantized,
            "valid_reconstructions": valid_reconstructions,
        }
