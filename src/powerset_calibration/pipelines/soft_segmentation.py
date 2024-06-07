import functools
from typing import Callable, Optional, Text, Union

import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

from powerset_calibration.utils.permutation import permutate_powerset


class SoftSpeakerSegmentationPowerset(Pipeline):
    """Aggregates overlapping sliding windows by making them match as much as possible.
    NOT usable for real diarization (speaker identities might swap after a few windows!).

    Copyright (c) 2024- CNRS
    AUTHORS
    Hervé BREDIN
    Alexis PLAQUET

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation-3.0".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to one third (i.e. 66% overlap between two consecutive windows).
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    normalize_output : bool, optional
        When `True`, output scores are normalized so that the sum of the probabilities
        still == 1.

    Usage
    -----
    # perform speaker segmentation
    >>> pipeline = SpeakerSegmentation()
    >>> segmentation: SlidingWindowFeature = pipeline("/path/to/audio.wav")

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation-3.0",
        segmentation_step: float = 1 / 3,
        segmentation_batch_size: int = 1,
        use_auth_token: Union[Text, None] = None,
        normalize_output: bool = True,
    ):
        super().__init__()

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, use_auth_token=use_auth_token)

        specifications = model.specifications
        if not specifications.powerset:
            raise ValueError("Only powerset segmentation models are supported.")

        self.segmentation_step = segmentation_step
        self.normalize_output = normalize_output

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            skip_conversion=True,
            batch_size=segmentation_batch_size,
        )
        frame_duration = model.specifications.duration / model.num_frames(
            16000 * model.specifications.duration
        )
        self._frames: SlidingWindow = SlidingWindow(frame_duration, frame_duration)

        self._powerset = Powerset(len(specifications.classes), specifications.powerset_max_classes)

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """

        if hook is None:
            inference_hook = None
        else:
            inference_hook = functools.partial(hook, "segmentation", None)

        if self.training:
            segmentations = file.setdefault("cache", dict()).setdefault("segmentation", None)

            if segmentations is None:
                segmentations = self._segmentation(file, hook=inference_hook)
                file["cache"]["segmentation"] = segmentations

            return segmentations

        return self._segmentation(file, hook=inference_hook)

    @staticmethod
    def align_chunks(
        ps_segmentations: SlidingWindowFeature,
        powerset: Powerset,
        frames: Optional[SlidingWindow] = None,
        normalize_output: bool = True,
        segmentations_in_logspace=True,
        ignore_index=None,
        hook: Optional[Callable] = None,
    ):
        # ps_segmentations shape: (num_chunks, num_frames, local_num_speakers)
        # MISSING:
        # - powerset
        # - segmentation step size
        if normalize_output and powerset is None:
            raise ValueError(
                "Normalizing multilabel output probably does not make sense. Please set normalize_output=False."
            )
        # if not segmentations_in_logspace and psdata[((psdata < 0) | (psdata > 1))].any():
        #     raise ValueError("Segmentations are not in logspace and contain values outside of [0,1].")

        num_chunks, num_frames, num_classes = ps_segmentations.data.shape

        permutated_segmentations = np.zeros_like(ps_segmentations.data)

        # number of frames in common between two consecutive chunks
        num_overlapping_frames = round(
            (1 - ps_segmentations.sliding_window.step / ps_segmentations.sliding_window.duration)
            * num_frames
        )

        # permutate each window to match previous one as much as possible
        for c, (_, segmentation) in enumerate(ps_segmentations):
            # segmentation/previous_segmentation are (num_frames, local_num_speakers)-shaped
            if hook is not None:
                hook("permutated_segmentation", None, completed=c, total=num_chunks)
            segmentation = (
                torch.from_numpy(segmentation)
                if isinstance(segmentation, np.ndarray)
                else segmentation
            )

            # remove unused classes
            if ignore_index is not None:
                num_missing_in_classes = (segmentation == ignore_index).int().sum(dim=0)
                if (num_missing_in_classes != num_frames & num_missing_in_classes != 0).any():
                    raise ValueError(
                        "Segmentation contains chunks where SOME frames have missing classes. The class should be missing for the whole chunk or not be missing at all."
                    )
                class_mask = ~(num_missing_in_classes == num_frames)
                segmentation = segmentation[..., class_mask]

            if c == 0:
                previous_segmentation = segmentation
            else:
                align_ref = previous_segmentation[-num_overlapping_frames:]
                align_pred = segmentation[:num_overlapping_frames]

                # align
                if powerset is not None:
                    _, best_perm = permutate_powerset(align_ref, align_pred, powerset)
                    previous_segmentation = segmentation[..., best_perm].clone()
                else:
                    # account that permutate works with batch, hence the [None,...] and [0]
                    _, best_perm = permutate(align_ref[None, ...], align_pred[None, ...])
                    previous_segmentation = segmentation[..., best_perm[0]].clone()
            permutated_segmentations[c] = previous_segmentation

        permutated_segmentations = SlidingWindowFeature(
            permutated_segmentations, ps_segmentations.sliding_window
        )

        result = Inference.aggregate(
            permutated_segmentations, frames=frames, hamming=True, skip_average=False
        )
        # set frames containing NaNs to [0,0,...] for multilabel (no label), [1,0,0,...] for multiclass (1st class)
        nan_containing_frames = np.isnan(result.data).any(axis=-1)
        if powerset is not None:
            result.data[nan_containing_frames] = 0.0
            result.data[nan_containing_frames, 0] = 1.0
        else:
            result.data[nan_containing_frames] = 0.0

        # normalize output when multiclass
        if normalize_output:
            if segmentations_in_logspace:
                result.data = np.exp(result.data)
            result.data = result.data / result.data.sum(axis=-1, keepdims=True)
            if segmentations_in_logspace:
                result.data = np.log(result.data)
        return result

    def apply(
        self,
        file: AudioFile,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        segmentation : SlidingWindowFeature
            Speaker segmentation
        """
        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model on a sliding window
        powerset_segmentations = self.get_segmentations(file, hook=hook)
        hook("powerset_segmentation", powerset_segmentations)
        # shape: (num_chunks, num_frames, local_num_speakers)

        permutated_segmentations = SoftSpeakerSegmentationPowerset.align_chunks(
            powerset_segmentations,
            self._powerset,
            frames=self._frames,
            normalize_output=self.normalize_output,
            hook=hook,
        )

        num_chunks, num_frames, _ = powerset_segmentations.data.shape
        hook(
            "permutated_segmentation",
            permutated_segmentations,
            completed=num_chunks,
            total=num_chunks,
        )

        return permutated_segmentations
