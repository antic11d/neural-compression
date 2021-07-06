from dtw import dtw
from numpy.linalg import norm
import torch


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
        # TODO: Add comparison plotting
        evaluation_entry["dtw_distance"] = dtw_distance
        return evaluation_entry

    def _dtw_distance(self, mfcc1, mfcc2):
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
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
