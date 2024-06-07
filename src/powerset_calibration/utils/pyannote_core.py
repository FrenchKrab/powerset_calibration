"""Utility method for pyannote.core objects."""

from typing import Generator, Optional, Tuple

import torch
from pyannote.core import Annotation, Segment, Timeline


def timeline_subtimeline(
    timeline: Timeline, start_ratio: float, end_ratio: float, min_duration: float = 0.0
) -> Timeline:
    """Extract a sub-timeline from a Timeline, given a start and end ratio (relative to the timeline duration).

    Example 1
    ---------
    >>> timeline = Timeline([Segment(0, 10), Segment(100, 110)])
    >>> subtimeline = timeline_subtimeline(timeline, 0.25, 0.75)
    >>> subtimeline
    Timeline([Segment(5.0, 10.0), Segment(100.0, 105.0)])

    Parameters
    ----------
    timeline : Timeline
        The timeline to work with
    start_ratio : float
        The beginning of the subtimeline, as a ratio of the original timeline's duration (in [0, 1])
    end_ratio : float
        The end of the subtimeline, as a ratio of the original timeline's duration (in [0, 1])
    min_duration : float, optional
        Minimum duration of the segments in the subtimeline, in seconds.
        Will disturb the expected duration (might be longer than expected).
        This is NOT the minimum total duration. Instead when a segment is present, and not bound
        by the original timeline, it will be at least 'min_duration' long.
        By default 0.0

    Returns
    -------
    Timeline
        Sub-timeline
    """
    new_timeline = Timeline()

    duration_before_start = timeline.duration() * start_ratio
    duration_left = (end_ratio - start_ratio) * timeline.duration()
    # iterate on gaps
    # maybe iterating on the support segments would be easier ?
    for segment in timeline.support_iter():
        offset = min(segment.duration, duration_before_start)
        duration_before_start -= offset
        if duration_before_start > 0:
            continue
        if duration_left <= 0:
            break
        seg = Segment(
            segment.start + offset,
            min(segment.start + offset + max(duration_left, min_duration), segment.end),
        )
        duration_left -= seg.duration
        new_timeline.add(seg)
    return new_timeline


def tensor_to_segments(t: torch.Tensor, fps: float) -> Generator[Tuple[Segment, bool], None, None]:
    """Iterate over the positive & negative continuous segments described by a boolean tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor
    fps : float
        Frames per second

    Yields
    ------
    Tuple[Segment, bool]
        A tuple of segment and its value
    """
    segments_values, segment_len = torch.unique_consecutive(t, return_counts=True)
    total_offset = 0
    for val_t, length_t in zip(segments_values, segment_len):
        val: bool = val_t.item()
        length: int = length_t.item()
        yield (Segment(total_offset / fps, (total_offset + length) / fps), val)
        total_offset += length


def tensor_to_timeline(t: torch.Tensor, fps: float, uri: Optional[str] = None) -> Timeline:
    """Convert a boolean tensor to a `pyannote.core.Timeline`.
    True values represent the segments, consecutive values make up one individual segment.

    Parameters
    ----------
    t : torch.Tensor
        Boolean tensor
    fps : float
        Frames per second
    uri : _type_, optional
        URI to give to the Timeline, by default None

    Returns
    -------
    Timeline
        Timeline corresponding to the provided tensor
    """
    tl = Timeline(uri=uri)
    for segment, seg_val in tensor_to_segments(t, fps):
        if seg_val:
            tl.add(segment)
    return tl


def tensor_to_annotation(t: torch.Tensor, fps: float, uri: Optional[str] = None) -> Annotation:
    """Convert a 2D tensor to a `pyannote.core.Annotation`.

    Parameters
    ----------
    t : torch.Tensor
        Input 2D tensor (time, tracks)
    fps : float
        Frames per second
    uri : str, optional
        The Annotation's URI, by default None

    Returns
    -------
    Annotation
        The Annotation corresponding to the given tensor.
    """
    ann = Annotation(uri=uri)
    for i in range(t.shape[1]):
        for segment, val in tensor_to_segments(t[:, i], fps):
            if val:
                ann[segment, i] = str(i)
    return ann
