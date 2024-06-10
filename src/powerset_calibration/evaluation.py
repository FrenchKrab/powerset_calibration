
from pathlib import Path
from typing import TypedDict, Union

from powerset_calibration.utils.permutation import match_speaker_count_and_permutate
from pyannote.audio.utils.powerset import Powerset
from powerset_calibration.inference import load_inference_file, load_inference_metadata
import torch


class DerComponentsDict(TypedDict):
    total_false_alarm: float
    total_missed_detection: float
    total_speaker_confusion: float
    total_speech: float
    false_alarm: float
    missed_detection: float
    speaker_confusion: float
    der: float


def compute_der_inference_file(fid: Union[str, Path]) -> DerComponentsDict:
    """Compute the DER from the path to an inference file.
    Only support unaggregated inference files for now (eg Speaker Diarization).

    Parameters
    ----------
    id : Union[str, Path]
        Path/id of the inference result.

    Returns
    -------
    DerComponentsDict
        Components of the DER.
    """

    infdf = load_inference_file(fid)
    metadata = load_inference_metadata(fid)

    is_aggregated = len(metadata["last_inference_shape"]) == 2

    t_preds = torch.stack(
        [
            torch.from_numpy(infdf[col].values)
            for col in infdf.columns
            if col.startswith("out_")
        ],
        dim=-1,
    )
    t_refs = torch.stack(
        [
            torch.from_numpy(infdf[col].values)
            for col in infdf.columns
            if col.startswith("ref_")
        ],
        dim=-1,
    )
    t_uem = torch.from_numpy(infdf["uem"].values)

    # Reshape reference & hypothesis if output not aggregated/is sliding window
    if not is_aggregated:
        # reshape to (num_windows, num_frames_in_window, num_classes)
        t_preds = t_preds.reshape(
            (-1, metadata["model"]["num_frames"], t_preds.shape[-1])
        )
        t_refs = t_refs.reshape((-1, metadata["model"]["num_frames"], t_refs.shape[-1]))
        # reshape to (num_windows, num_frames_in_window)
        t_uem = t_uem.reshape((-1, metadata["model"]["num_frames"])).all(dim=-1)

        # Convert predictions to multilabel if needed
        if metadata["model"]["specifications"]["powerset"] is True:
            powerset = Powerset(
                num_classes=len(metadata["model"]["specifications"]["classes"]),
                max_set_size=metadata["model"]["specifications"][
                    "powerset_max_classes"
                ],
            )
            t_preds = powerset.to_multilabel(t_preds)
    else:
        raise NotImplementedError("Not implemented for non powerset.")

    total_false_alarm = 0.0
    total_missed_detection = 0.0
    total_speaker_confusion = 0.0
    total_speech_total = 0.0

    for chunk_preds, chunk_refs, chunk_uem in zip(t_preds, t_refs, t_uem):
        # Only compute DER on chunks where all UEM are true
        if not is_aggregated and chunk_uem.item() != True:
            continue

        chunk_preds, chunk_refs = match_speaker_count_and_permutate(
            chunk_preds[None], chunk_refs[None]
        )
        # print(chunk_preds.shape, chunk_refs.shape)

        # --- Compute DER like in pyannote.audio.torchmetrics.DiarizationErrorRate
        chunk_refs = torch.transpose(chunk_refs, 1, 2)
        chunk_preds = torch.transpose(chunk_preds, 1, 2)

        # turn continuous [0, 1] predictions into binary {0, 1} decisions
        hypothesis = (chunk_preds.unsqueeze(-1) > 0.5).float()
        # (batch_size, num_speakers, num_frames, num_thresholds)

        target = chunk_refs.unsqueeze(-1)
        # (batch_size, num_speakers, num_frames, 1)

        detection_error = torch.sum(hypothesis, 1) - torch.sum(target, 1)
        # (batch_size, num_frames, num_thresholds)

        false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
        # (batch_size, num_frames, num_thresholds)

        missed_detection = torch.maximum(
            -detection_error, torch.zeros_like(detection_error)
        )
        # (batch_size, num_frames, num_thresholds)

        speaker_confusion = (
            torch.sum((hypothesis != target) * hypothesis, 1) - false_alarm
        )
        # (batch_size, num_frames, num_thresholds)

        false_alarm = torch.sum(torch.sum(false_alarm, 1), 0)
        missed_detection = torch.sum(torch.sum(missed_detection, 1), 0)
        speaker_confusion = torch.sum(torch.sum(speaker_confusion, 1), 0)
        # (num_thresholds, )

        total_speech_total += (1.0 * torch.sum(target)).item()
        total_false_alarm += false_alarm.item()
        total_missed_detection += missed_detection.item()
        total_speaker_confusion += speaker_confusion.item()

    return {
        "total_false_alarm": total_false_alarm,
        "total_missed_detection": total_missed_detection,
        "total_speaker_confusion": total_speaker_confusion,
        "total_speech": total_speech_total,
        "false_alarm": total_false_alarm / total_speech_total,
        "missed_detection": total_missed_detection / total_speech_total,
        "speaker_confusion": total_speaker_confusion / total_speech_total,
        "der": (total_false_alarm + total_missed_detection + total_speaker_confusion)
        / total_speech_total,
    }