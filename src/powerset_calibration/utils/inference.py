"""Utilities for handling inference dataframes/tensors and for aggregating tensors."""

from typing import Dict, Iterable, Literal, Optional

import pandas as pd
from powerset_calibration.utils.permutation import lossy_match_speaker_count_and_permutate
import torch
import tqdm
from powerset_calibration.pipelines.soft_segmentation import SoftSpeakerSegmentationPowerset
from powerset_calibration.utils.pyannote_core import tensor_to_timeline
from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.powerset import Powerset
from pyannote.core import SlidingWindow, SlidingWindowFeature, Timeline

AggregationStrategyType = Literal["lazy", "soft_segmentation"]
""" 
- `lazy` = no aggregation, return the tensor as is (implying it will be aggregated later)
- `soft_segmentation` = rely on the soft segmentation pipeline"""


def infdf_to_timelines(
    df: pd.DataFrame, fps: float, colname: str, display_progress: bool = False
) -> Dict[str, Timeline]:
    """Convert an inference dataframe to timelines (one per URI) using the provided column.

    Example:
    ```
    df["new_uem"] = torch.ones_like(df["uem"])
    timelines = infdf_to_timelines(df, fps=fps, colname="new_uem")
    ```

    Parameters
    ----------
    df : pd.DataFrame
        Inference dataframe
    fps : float
        Frames per second
    colname : str
        Column that contain booleans, to be used as 'UEM'

    Returns
    -------
    Dict[str, Timeline]
        Maps each URI to its Timeline
    """
    result: Dict[str, Timeline] = {}
    for uri in tqdm.tqdm(df["uri"].unique(), disable=not display_progress):
        uem = torch.from_numpy(df[df["uri"] == uri][colname].values)
        result[uri] = tensor_to_timeline(uem, fps, uri=uri)
    return result


def get_output_columns(columns: Iterable[str]) -> list[str]:
    """Get the output columns from a list of columns

    Parameters
    ----------
    columns : list[str]
        List of columns

    Returns
    -------
    list[str]
        List of output columns
    """
    return [col for col in columns if col == "ov" or col.startswith("out_")]


def get_reference_columns(columns: Iterable[str]) -> list[str]:
    """Get the reference columns from a list of columns

    Parameters
    ----------
    columns : list[str]
        List of columns

    Returns
    -------
    list[str]
        List of reference columns
    """
    return [col for col in columns if col.startswith("ref_") or col.endswith("_ref")]


def get_correctshaped_filetensor(t: torch.Tensor, metadata) -> torch.Tensor:
    """Convert a tensor representing the data inside a single file (=URI) to a standard shape.
    - (num_chunks, num_frames, num_classes) if unaggregated
    - (num_frames, num_classes) if aggregated.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to reshape. Shape must be either
        - (num_frames, num_classes)
        - (num_frames)
    metadata : _type_
        InferenceMetadata
    """

    unaggregated = len(metadata["last_inference_shape"]) == 3
    # easy case, nothing to reshape
    if not unaggregated:
        if t.ndim == 1:
            return t[:, None]
        if t.ndim == 2:
            return t
        else:
            raise ValueError(f"Tensor has unexpected shape {t.shape}. ndim should be 1 or 2.")

    # the output is unaggregated, convert back to the (num_chunks, num_frames, num_classes) shape
    target_shape = (-1, metadata["model"]["num_frames"])
    if t.ndim == 2:
        target_shape = target_shape + t.shape[-1:]
    elif t.ndim == 1:
        target_shape = target_shape + (1,)
    else:
        raise ValueError(f"Tensor has unexpected shape {t.shape}. ndim should be 1 or 2.")
    return t.reshape(target_shape)


def aggregate_sliding_window_tensor(
    tensor: torch.Tensor,
    sliding_window: SlidingWindow,
    frames: Optional[SlidingWindow] = None,
    nan_value: float = 0.0,
) -> torch.Tensor:
    """Shorthand for applying Inference.aggregate, go from sliding window space to continuous space

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to aggregate (num_chunks, num_frames, num_classes)
    window_duration : float
        Sliding window duration in seconds
    window_step : float
        Sliding window step in seconds
    Returns
    -------
    torch.Tensor
        Aggregated (num_frames, num_classes) tensor
    """

    swf = SlidingWindowFeature(tensor.cpu().numpy(), sliding_window)
    swf_agg = Inference.aggregate(scores=swf, frames=frames)
    result = torch.from_numpy(swf_agg.data)
    if nan_value is not None:
        result = result.nan_to_num(nan=nan_value)
    return result


def aggregate_soft_segmentation(
    t: torch.Tensor,
    sliding_window: SlidingWindow,
    powerset: Optional[Powerset] = None,
    frames: Optional[SlidingWindow] = None,
) -> torch.Tensor:
    """Aggregate a speaker diarization tensor using the soft segmentation pipeline.

    Parameters
    ----------
    t : torch.Tensor
        Tensor (num_chunks, num_frames, num_classes)-shaped to aggregate, in probability space.
    sliding_window : SlidingWindow
        The sliding window that generated the tensor `t`.
    powerset : Optional[Powerset], optional
        If relevant, the powerset describing the classes of `t`, by default None
    frames : Optional[SlidingWindow], optional
        `model.example_output.frames` of the model that generated `t`, for more
        accurate aggregations. By default None

    Returns
    -------
    torch.Tensor
        The tensor aggregated in a (num_frames2, num_classes) shape.
    """
    segmentations = SlidingWindowFeature(t, sliding_window)

    r = SoftSpeakerSegmentationPowerset.align_chunks(
        segmentations,
        powerset=powerset,
        frames=frames,
        normalize_output=powerset is not None,
        segmentations_in_logspace=False,
    )
    return torch.from_numpy(r.data)


def apply_aggregation_strategy(
    strategy: AggregationStrategyType,
    t: torch.Tensor,
    sliding_window: SlidingWindow,
    powerset: Optional[Powerset] = None,
    frames: Optional[SlidingWindow] = None,
) -> torch.Tensor:
    """Apply one the existing aggregation strategies to a tensor.

    Parameters
    ----------
    strategy : AggregationStrategyType
        Type of aggregation.
    t : torch.Tensor
        Tensor (num_chunks, num_frames, num_classes)-shaped to aggregate, in probability space.
    sliding_window : SlidingWindow
        The sliding window that generated the tensor `t`.
    powerset : Powerset, optional
        If relevant, the powerset describing the classes of `t`, by default None
    frames : SlidingWindow, optional
        `model.example_output.frames` of the model that generated `t`, for more
        accurate aggregations. By default None

    Returns
    -------
    torch.Tensor
        The tensor aggregated in a (num_frames2, num_classes) shape.
    """
    if strategy == "lazy":
        return t
    elif strategy == "soft_segmentation":
        return aggregate_soft_segmentation(t, sliding_window, powerset=powerset, frames=frames)
    raise ValueError(f"Unknown strategy {strategy}")


def get_heuristic_tensor_segmentation(
    heuristic: Literal["confidence", "bce", "random", "ce", "entropy"],
    preds: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    sliding_window: Optional[SlidingWindow] = None,
    frames: Optional[SlidingWindow] = None,
    powerset: Optional[Powerset] = None,
) -> torch.Tensor:
    """Returns the heuristic tensor where low=bad and high=good (higher=best performance).
    Such that in active learning we usually want the low=bad values to be sampled.

    Parameters
    ----------
    heuristic : Literal['confidence', 'bce', 'random']
        The type of heuristic to compute/return.
        - `confidence` = max probability class
        - `bce` = -binary cross entropy (will happen in multilabel space even if powerset preds)
        - `ce` = -cross entropy (does not make sense if preds is multilabel)
        - `random` = random tensor
        - `entropy` = entropy in the feature dimension

    preds : torch.Tensor
        The prediction tensor in multilabel or powerset space.
    targets : torch.Tensor, optional
        The target tensor, in multilabel space.
        Only required for bce/ce, by default None
    sliding_window: Optional[SlidingWindow] = None
        Needed if the data is not aggregated.
        The sliding window use in the inference that generated the tensor `preds`.
    powerset: Optional[Powerset] = None,
        Needed if the preds and targets are not in the same space.
    Returns
    -------
    torch.Tensor
        Resulting heuristic tensor (n_frames) shaped.
    """
    # check aggregated and make sure everything is either
    # (num_frames, n_classes) for aggregated, or
    # (num_chunks, num_frames, n_classes) for unaggregated
    is_aggregated = preds.ndim == 2
    if is_aggregated:
        preds = preds.reshape((-1, preds.shape[-1]))
        targets = targets.reshape((-1, targets.shape[-1]))
    else:
        if preds.ndim != 3:
            raise ValueError(
                f"This function can only make sense of unaggregated tensors if they are 3D, got {preds.ndim} dims."
            )
        if sliding_window is None:
            raise ValueError(
                "You must pass a sliding window if you want to use an unaggregated tensor"
            )
    if preds[(preds < 0) | (preds > 1)].any():
        raise ValueError("Predictions must be between 0 and 1")

    # get heuristic
    if heuristic == "random":
        n_frames = preds.shape[0]
        if not is_aggregated:
            fps = preds.shape[1] / sliding_window.duration
            n_frames = int(sliding_window.step * preds.shape[0] * fps + preds.shape[1])
        return torch.rand((n_frames), device=preds.device)
    elif heuristic == "confidence":
        # for confidence we first compute the heuristic and THEN aggregated if needed
        conf = preds.max(dim=-1)[0][..., None]
        if not is_aggregated:
            conf = aggregate_sliding_window_tensor(
                conf,
                sliding_window=sliding_window,
                frames=frames,
            )
        return conf.flatten()
    elif heuristic == "entropy":
        entropy = torch.sum(preds * torch.log(preds + 1e-10), dim=-1)[..., None]
        if not is_aggregated:
            entropy = aggregate_sliding_window_tensor(
                entropy,
                sliding_window=sliding_window,
                frames=frames,
            )
        return entropy.flatten()
    elif heuristic in ["bce", "ce"]:
        if targets is None:
            raise ValueError("Target tensor must be provided for bce/ce")

        # Binary cross entropy loss
        if heuristic == "bce":
            if powerset is None:
                raise ValueError("If using bce, powerset must be provided.")
            if powerset is not None and targets.shape[-1] != powerset.num_classes:
                ml_preds = powerset.to_multilabel(preds.log(), True)
                targets, _ = lossy_match_speaker_count_and_permutate(
                    ml_preds[None, ...], targets[None, ...]
                )
                targets = targets[0]
                targets = powerset.to_powerset(targets).float()
            return -torch.nn.functional.binary_cross_entropy(
                preds, targets.float(), reduction="none"
            ).mean(dim=-1)
        # Cross entropy loss
        elif heuristic == "ce":
            if powerset is not None:
                # make targets align to the ml representation of preds, then convert targets to powerset space
                ml_preds = powerset.to_multilabel(preds.log(), soft=True)
                targets, _ = lossy_match_speaker_count_and_permutate(
                    ml_preds[None, ...], targets[None, ...]
                )
                targets = targets[0]
                targets = powerset.to_powerset(targets)
            else:
                raise ValueError("Cross entropy does not make sense for multilabel problems")
            # compute ce
            ce = -torch.nn.functional.nll_loss(
                preds.log(), targets.argmax(dim=-1), reduction="none"
            )
            return ce
    else:
        raise ValueError("Unsupported heuristic")
