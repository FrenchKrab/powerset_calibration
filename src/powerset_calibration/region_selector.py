"""
Select regions of data from tensors of values using convolutions.
e.g. select regions of low confidence in a file.
"""

import itertools
import math
import sys
from typing import Callable, Dict, Optional, Sequence

import torch
from powerset_calibration.utils.pyannote_core import tensor_to_timeline
from powerset_calibration.utils.torch import insert_at_indices, insert_at_indices_revert
from pyannote.core import Timeline


def generate_windows(
    values: torch.Tensor,
    fps: float,
    window_duration: float = 10.0,
    sliding_window_step: float = 0.5,
    annotated_ratio: Optional[float] = 0.05,
    annotated_duration: Optional[float] = None,
    conv=torch.nn.functional.avg_pool1d,
    argminmax=torch.argmin,
    penality: Optional[float] = None,
    uem: Optional[torch.Tensor] = None,
    selections_per_iteration: int = sys.maxsize,
) -> torch.Tensor:
    """Generate windows for active learning from a 'values' tensor.

    `penality` and `selections_per_iteration` go against each other in the selection
    process. `penality` helps to prevent selecting adjacent/overlapping windows
    (and needed so that the algorithm doesn't get stuck). `selections_per_iteration`
    helps to speed up the algorithm by selecting multiple windows at once, which
    allows overlapping selection.

    (If for some reason you want only regions of size of `window_duration`,
    set `penality=math.inf` and `selections_per_iterations=1`) (very slow).

    **Warning**: Currently, the last windows to be placed can be smaller than `window_duration`,
    and is **not** forced to be ajacent to the other windows to compensate for that!

    Parameters
    ----------
    values : torch.Tensor
        1D tensor guiding how to prioritize windows placement.
    fps : float
        Frames per second
    window_duration : float, optional
        Minimum duration of the annotation windows in seconds, by default 10.0
    sliding_window_step : float, optional
        Step size of the sliding window in seconds, by default 0.5
    annotated_ratio : float, optional
        Ratio of the 'values' to annotate, by default 0.05 (= 5%)
    annotated_duration : float, optional
        Duration of the 'values' to annotate. Give either this or annotated ratio, by default None
    conv : Callable, optional
        Convolutional fn to use, by default `torch.nn.functional.avg_pool1d`
    argminmax : Callable, optional
        Should be `torch.argmax` or `torch.argmin`, depending on whether kernels
        should contain the smallest or biggest value possible, by default `torch.argmin`
    penality : float, optional
        The penality to use for frames that have already be annotated.
        math.inf will prevent any overlap between windows. If left to None it will
        find the least punishing so that windows can overlap. By default None
    uem : torch.Tensor, optional
        UEM boolean tensor to limit the places where windows can be placed, by default None
    selections_per_iteration: int, optional
        How many windows should the algorithm try to place at each iteration.
        Higher values are faster but the results do not take penality into account
        as much. You probably either want to set this to `1` or `sys.maxsize`
        By default `sys.maxsize`

    Returns
    -------
    torch.Tensor
        Boolean tensor with same shape as 'value', True values for frames to annotate.
    """
    if uem is not None and uem.ndim != 1 and uem.dtype != torch.bool:
        raise ValueError(f"uem must be 1D bool tensor. Got shape {uem.shape} & dtype {uem.dtype}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D. Got shape {values.shape}")
    if (annotated_duration is None) == (annotated_ratio is None):
        raise ValueError(
            f"annotated_ratio and annotated_duration are mutually exclusive. (Only) one of them must be provided. You gave {annotated_ratio=} and {annotated_duration=}"
        )

    signs = torch.tensor([-1, +1], dtype=torch.int)
    penalities_sign: int = -int(signs[argminmax(signs)].item())
    higher_is_better: bool = penalities_sign == -1
    # If not given by user,
    # use something *slightly* worse than worse value as penality
    if penality is None:
        noninf_values = values[~values.isinf()]
        penality = noninf_values[argminmax(-noninf_values)].item()
        penality += 1e-6 * torch.sign(torch.Tensor([penality])).item()
    # or make sure the penality has the right sign if user inputed
    else:
        penality = penalities_sign * abs(penality)

    # compute # of frames per window, stride, init tensors, utils, etc
    window_nframes = max(1, math.ceil(window_duration * fps))
    stride = max(1, round(sliding_window_step * fps))
    to_annotate = torch.zeros_like(values, dtype=torch.bool)

    # compute # of frames to annotate
    if annotated_ratio is not None:
        n_annotatable_frames: int = values.shape[0] if uem is None else uem.sum().item()
        to_annotate_total: int = math.ceil(annotated_ratio * n_annotatable_frames)
    elif annotated_duration is not None:
        to_annotate_total: int = math.ceil(annotated_duration * fps)
    else:
        raise ValueError("Either annotated_ratio or annotated_duration must be provided")

    def left_to_annotate() -> int:
        return int(to_annotate_total - to_annotate.sum().item())

    # create a "penalized" version of 'values' where
    # non-uem frames & already annotated frames are penalized
    # so that they are not chosen at all / not chosen over and over
    penalized_vals = values.clone()
    if uem is not None:
        # infinite penality for non-uem belonging frames
        penalized_vals[~uem] = penalities_sign * math.inf

    left_previous = math.inf
    left = left_to_annotate()
    # while there is still frames to annotate and we aren't stuck
    while left > 0 and left_previous > left:
        # Compute the best window(s) using a convolution
        kernel_size = min(left_to_annotate(), window_nframes)
        penalized_vals[to_annotate] = penality
        convresult = conv(
            input=penalized_vals[None, :], kernel_size=kernel_size, stride=stride
        ).squeeze(0)

        # If selections_per_iteration is 1, we can use argminmax directly
        if selections_per_iteration <= 1:
            best_conv_idx = argminmax(convresult)
            best_start_idx = best_conv_idx * stride
            best_end_idx = best_conv_idx * stride + kernel_size
            to_annotate[best_start_idx:best_end_idx] = True
        # Else we take the n best windows
        else:
            best_conv_idx_sorted = torch.argsort(convresult, descending=higher_is_better)

            needed_windows = left_to_annotate() // kernel_size
            for i in range(min(needed_windows, selections_per_iteration)):
                ibest_idx = best_conv_idx_sorted[i].item()
                ibest_start = ibest_idx * stride
                ibest_end = ibest_idx * stride + kernel_size
                to_annotate[ibest_start:ibest_end] = True

        left_previous = left
        left = left_to_annotate()
    return to_annotate


def generate_windows_bins(
    values: torch.Tensor,
    fps: float,
    window_duration: float,
    sliding_window_step: float,
    bins_count: int,
    samples_per_quantile: int,
    conv=torch.nn.functional.avg_pool1d,
    uem: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate windows of a given duration that are spread out through the values quantiles.
    For example, generate 10 windows, one where the average value is in the 0-10% quantile,
    one in the 10-20% quantile, etc.
    Generates `bins_count * samples_per_quantile` windows.

    Warning: Windows can currently overlap, so there might be less data than desired.

    Parameters
    ----------
    values : torch.Tensor
        1D values tensor
    fps : float
        Frames per second
    window_duration : float
        Window duration in seconds
    sliding_window_step : float
        How precise the window placement should be in seconds.
    bins_count : int
        Number of bins/quantiles to use.
    samples_per_quantile : int
        Number of windows to generate per quantiles.
    conv : _type_, optional
        Convolutional function to summarize the windows values, by default torch.nn.functional.avg_pool1d
    uem : Optional[torch.Tensor], optional
        UEM indicating what regions can be used, by default None

    Returns
    -------
    torch.Tensor
        1D Boolean tensor, same shape as `values`, `True` values for frames to annotate.
    """

    # TODO: implement a way to ensure that the result has at least x seconds of annotations
    # currently, windows from multiple bins can overlap.

    if uem is not None and (uem.ndim != 1 and uem.dtype != torch.bool):
        raise ValueError(f"uem must be 1D bool tensor. Got shape {uem.shape} & dtype {uem.dtype}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D. Got shape {values.shape}")

    # compute # of frames per window, stride, init tensors, utils, etc
    window_nframes = max(1, math.ceil(window_duration * fps))
    stride = max(1, round(sliding_window_step * fps))
    to_annotate = torch.zeros_like(values, dtype=torch.bool)

    # Compute the best window(s) using a convolution
    kernel_size = window_nframes
    convresult: torch.Tensor = conv(
        input=values[None, :], kernel_size=kernel_size, stride=stride
    ).squeeze(0)
    # same resolution as convresult (= number of windows). true when windows is usable
    if uem is not None:
        # inverse of (windows that contain at least one non-uem frame)
        convresult_uem: torch.Tensor = ~(
            torch.nn.functional.max_pool1d(
                input=(~uem).float()[None, :], kernel_size=kernel_size, stride=stride
            ).squeeze(0)
            > 1e-5
        )
        # print(f"{convresult_uem=}")
    else:
        convresult_uem = torch.ones_like(convresult, dtype=torch.bool)

    quantiles: torch.Tensor = convresult[(~convresult.isnan()) & (~convresult.isinf())].quantile(
        torch.linspace(0, 1, bins_count + 1)
    )
    for quantile_start, quantile_end in zip(quantiles[:-1], quantiles[1:]):
        quantile_mask: torch.Tensor = (convresult >= quantile_start) & (convresult < quantile_end)
        quantile_mask = quantile_mask & convresult_uem

        quantile_mask_idxs: torch.Tensor = quantile_mask.nonzero()
        samples_idxs: torch.Tensor = quantile_mask_idxs[
            torch.randperm(quantile_mask_idxs.shape[0])[:samples_per_quantile]
        ]
        # print(
        #     f"quantile [{quantile_start.item():.2f}, {quantile_end.item():.2f}[ has {samples_idxs.shape[0]} samples"
        # )
        for sample_idx_t in samples_idxs:
            sample_idx: int = int(sample_idx_t.item())
            sample_start = sample_idx * stride
            sample_end = sample_idx * stride + kernel_size
            to_annotate[sample_start:sample_end] = True

    return to_annotate


def generate_windows_multiplefiles(
    t_htensor: torch.Tensor,
    t_uris: torch.Tensor,
    uris: Sequence[str],
    fps: float,
    window_duration: float,
    sliding_window_step: float,
    annotated_ratio: Optional[float] = None,
    annotated_duration: Optional[float] = None,
    selections_per_iteration: int = sys.maxsize,
    conv: Callable = torch.nn.functional.avg_pool1d,
    argminmax: Callable = torch.argmin,
    t_uem: Optional[torch.Tensor] = None,
) -> Dict[str, Timeline]:
    """Apply the region selection algorithm to a tensor of concatenated heuristics.
    This means the algorithm will be able to choose from all files (and thus have a 'global'
    selection budget).

    See `generate_windows` for other details

    Parameters
    ----------
    t_htensor : torch.Tensor
        1D "heuristic" tensor used to guide the region selection. Can contain multiple
        concatenated tensors.
    t_uris : torch.Tensor
        1D tensor containing uri IDs as to distinguish file borders.
        Same shape as `t_htensor`.
    uris : Sequence[str]
        URIs strings correspondig to the URI IDs in t_uris.
    fps : float
        Frames per second of the model
    window_duration : float
        Duration of selected regions
    sliding_window_step : float
        How precisely should regions be placed, in seconds
    annotated_ratio : Optional[float], optional
        Ratio of the data to annotate in [0; 1].
        Incompatible with `annotated_duration`, by default None
    annotated_duration : Optional[float], optional
        Total time to annotate in seconds.
        Incompatible with `annotated_ratio`, by default None
    selections_per_iteration : int, optional
        How many windows should the algorithm try to place at each iteration.
        Higher values are faster but the results do not take penality into account
        as much. You probably either want to set this to `1` or `sys.maxsize`
        By default `sys.maxsize`
    conv : Callable, optional
        Convolutional fn to use, by default `torch.nn.functional.avg_pool1d`
    argminmax : Callable, optional
        Should be `torch.argmax` or `torch.argmin`, depending on whether kernels
        should contain the smallest or biggest value possible, by default `torch.argmin`
    t_uem : Optional[torch.Tensor], optional
        1D UEM boolean tensor, same shape as `t_htensor`, by default None

    Returns
    -------
    Dict[str, Timeline]
        Maps each URI to the timeline of selected regions.
    """

    if (annotated_duration is None) == (annotated_ratio is None):
        raise ValueError(
            "annotated_ratio and annotated_duration are mutually exclusive. (Only) one of them must be provided"
        )

    if t_uem is not None and t_uem.dtype != torch.bool:
        raise ValueError("UEM must be a boolean tensor")

    if annotated_ratio is not None:
        if t_uem is not None:
            annotable_frame_count = t_uem.sum().item()
        else:
            annotable_frame_count = t_htensor.shape[0]
        annotated_duration = annotated_ratio * annotable_frame_count / fps

    result: Dict[str, Timeline] = {}

    # compute file uris indices to know where file start/end
    t_uris_count = torch.unique_consecutive(t_uris, return_counts=True)[1]
    t_uris_index = t_uris_count.cumsum(0)

    # create the split punished version of the htensor, where file boundaries are penalized
    # so that no selection can be made across file boundaries
    signs = torch.tensor([-1, +1])
    penalities_sign: int = -signs[argminmax(signs)].item()
    t_htensor_splitpunish = insert_at_indices(t_htensor, t_uris_index, penalities_sign * math.inf)
    if t_uem is None:
        t_uem = torch.ones_like(t_htensor)
    t_uem_splitpunish = insert_at_indices(t_uem, t_uris_index, False)

    t_selected = generate_windows(
        values=t_htensor_splitpunish,
        uem=t_uem_splitpunish,
        fps=fps,
        window_duration=window_duration,
        sliding_window_step=sliding_window_step,
        annotated_ratio=None,  # we computed it ourselves
        annotated_duration=annotated_duration,
        conv=conv,
        argminmax=argminmax,
        selections_per_iteration=selections_per_iteration,
    )
    # Remove the file boundary penalities
    t_selected = insert_at_indices_revert(t_selected, t_uris_index)

    for uri, idx_start, frame_count in zip(
        uris, itertools.chain([0], t_uris_index[:-1]), t_uris_count
    ):
        f_selected = t_selected[idx_start : idx_start + frame_count]

        result[uri] = tensor_to_timeline(f_selected, fps=fps, uri=uri)

    return result


def generate_windows_multiplefiles_bins(
    t_htensor: torch.Tensor,
    t_uris: torch.Tensor,
    uris: Sequence[str],
    fps: float,
    window_duration: float,
    sliding_window_step: float,
    bins_count: int,
    samples_per_bin: int,
    conv: Callable = torch.nn.functional.avg_pool1d,
    t_uem: Optional[torch.Tensor] = None,
) -> Dict[str, Timeline]:
    """Apply the BINNED region selection algorithm to a tensor of concatenated heuristics.
    This means the algorithm will be able to choose from all files (and thus have a 'global'
    selection budget).

    See `generate_windows_bins` for other details.

    Parameters
    ----------
    t_htensor : torch.Tensor
        1D "heuristic" tensor used to guide the region selection. Can contain multiple
        concatenated tensors.
    t_uris : torch.Tensor
        1D tensor containing uri IDs as to distinguish file borders.
        Same shape as `t_htensor`.
    uris : Sequence[str]
        URIs strings correspondig to the URI IDs in t_uris.
    fps : float
        Frames per second of the model
    window_duration : float
        Duration of selected regions
    sliding_window_step : float
        How precisely should regions be placed, in seconds
    bins_count: int,
        Number of bins/quantiles to use.
    samples_per_bin: int,
        Number of windows to sample per quantiles.
    conv : Callable, optional
        Convolutional fn to use, by default `torch.nn.functional.avg_pool1d`
    t_uem : Optional[torch.Tensor], optional
        1D UEM boolean tensor, same shape as `t_htensor`, by default None

    Returns
    -------
    Dict[str, Timeline]
        Maps each URI to the timeline of selected regions.
    """

    # TODO: factorize with get_windows_whole_inference
    if t_uem is not None and t_uem.dtype != torch.bool:
        raise ValueError("UEM must be a boolean tensor")

    result: Dict[str, Timeline] = {}

    # compute file uris indices to know where file start/end
    t_uris_count = torch.unique_consecutive(t_uris, return_counts=True)[1]
    t_uris_index = t_uris_count.cumsum(0)

    # create the split punished version of the htensor, where file boundaries are penalized
    # so that no selection can be made across file boundaries

    t_htensor_splitpunish = insert_at_indices(t_htensor, t_uris_index, float("nan"))
    if t_uem is not None:
        t_uem_splitpunish = insert_at_indices(t_uem, t_uris_index, False)
    else:
        t_uem_splitpunish = None

    t_selected = generate_windows_bins(
        values=t_htensor_splitpunish,
        uem=t_uem_splitpunish,
        fps=fps,
        window_duration=window_duration,
        bins_count=bins_count,
        samples_per_quantile=samples_per_bin,
        sliding_window_step=sliding_window_step,
        conv=conv,
    )
    # Remove the file boundary penalities
    t_selected = insert_at_indices_revert(t_selected, t_uris_index)

    for uri, idx_start, frame_count in zip(
        uris, itertools.chain([0], t_uris_index[:-1]), t_uris_count
    ):
        f_selected = t_selected[idx_start : idx_start + frame_count]

        result[uri] = tensor_to_timeline(f_selected, fps=fps, uri=uri)

    return result
