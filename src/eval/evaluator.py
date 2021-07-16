from dtw import dtw
from numpy.linalg import norm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.utils.log import Logger
from pathlib import Path
from ..utils.misc import (
    load_txts,
    parse_audio,
    plot_pcolormesh,
    compute_unified_time_scale,
)


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
        evaluation_entry["dtw_distance"] = dtw_distance
        self._compute_comparison_plot(evaluation_entry)
        return evaluation_entry

    def _dtw_distance(self, mfcc1, mfcc2):
        dist, _, _, _ = dtw(
            mfcc1.T, mfcc2.T, dist=lambda x, y: norm((x - y).cpu(), ord=1)
        )
        return dist

    def _evaluate_once(self):
        self._model.eval()

        data = next(iter(self._data_stream.validation_loader))

        preprocessed_audio = data["preprocessed_audio"].to(self._device)
        valid_originals = data["input_features"].to(self._device)
        speaker_ids = data["speaker_id"].to(self._device)
        target = data["output_features"].to(self._device)
        wav_filename = data["wav_filename"]
        shifting_time = data["shifting_time"].to(self._device)
        preprocessed_length = data["preprocessed_length"].to(self._device)

        valid_originals = valid_originals.contiguous().float()
        batch_size = valid_originals.size(0)
        target = target.permute(0, 2, 1).contiguous().float()
        wav_filename = wav_filename[0][0]

        with torch.no_grad():
            (
                reconstructed_x,
                _,
                _,
                _,
                quantized,
                encodings,
                distances,
                encoding_indices,
                encoding_distances,
                embedding_distances,
                frames_vs_embedding_distances,
                concatenated_quantized,
            ) = self._model(valid_originals, self._data_stream.speaker_dic, speaker_ids)

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
            "valid_reconstructions": reconstructed_x.permute(0, 2, 1),
        }

    def _compute_comparison_plot(self, evaluation_entry):

        logger = Logger.get_logger()

        tmp = evaluation_entry["wav_filename"].split("/")[-1].split("\\")[-1].split("_")
        utterence_key = tmp[0] + "_" + tmp[1]
        path = Path(self._configuration["vctk_path"]).joinpath("raw", "VCTK-Corpus")
        utterences = load_txts(path)
        print(utterences[utterence_key])

        preprocessed_audio = (
            evaluation_entry["preprocessed_audio"].detach().cpu()[0].numpy().squeeze()
        )
        spectrogram = parse_audio(preprocessed_audio).contiguous()

        spectrogram = spectrogram.detach().cpu().numpy()

        valid_originals = evaluation_entry["valid_originals"].detach().cpu()[0].numpy()

        probs = (
            F.softmax(-evaluation_entry["distances"][0], dim=1)
            .detach()
            .cpu()
            .transpose(0, 1)
            .contiguous()
        )

        valid_reconstructions = (
            evaluation_entry["valid_reconstructions"].detach().cpu().numpy()
        )

        fig, axs = plt.subplots(6, 1, figsize=(35, 30), sharex=True)

        # Waveform of the original speech signal
        axs[0].set_title("Waveform of the original speech signal")
        axs[0].plot(
            np.arange(len(preprocessed_audio))
            / float(self._configuration["sampling_rate"]),
            preprocessed_audio,
        )

        # Spectrogram of the original speech signal
        axs[1].set_title("Spectrogram of the original speech signal")
        plot_pcolormesh(
            spectrogram,
            fig,
            x=compute_unified_time_scale(spectrogram.shape[1]),
            axis=axs[1],
        )

        # MFCC + d + a of the original speech signal
        axs[2].set_title(
            "Augmented MFCC + d + a #filters=13+13+13 of the original speech signal"
        )
        plot_pcolormesh(
            valid_originals.T,
            fig,
            x=compute_unified_time_scale(valid_originals.shape[0]),
            axis=axs[2],
        )

        # Softmax of distances computed in VQ
        axs[3].set_title(
            "Softmax of distances computed in VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder prior to quantization)"
        )
        plot_pcolormesh(
            probs,
            fig,
            x=compute_unified_time_scale(probs.shape[1], downsampling_factor=2),
            axis=axs[3],
        )

        encodings = evaluation_entry["encodings"].detach().cpu().numpy()
        axs[4].set_title("Encodings")
        plot_pcolormesh(
            encodings[0].transpose(),
            fig,
            x=compute_unified_time_scale(
                encodings[0].transpose().shape[1], downsampling_factor=2
            ),
            axis=axs[4],
        )

        # Actual reconstruction
        axs[5].set_title("Actual reconstruction")
        plot_pcolormesh(
            valid_reconstructions.squeeze(0).T,
            fig,
            x=compute_unified_time_scale(valid_reconstructions.shape[1]),
            axis=axs[5],
        )

        plot_name = "evaluation-comparison-plot.png"
        logger.save_plot(plot_name)
