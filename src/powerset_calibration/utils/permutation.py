"""Permutation utilities for powerset."""

import itertools
from typing import Callable, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset


def permutate_powerset(
    t1: torch.Tensor,
    t2: torch.Tensor,
    powerset: Powerset,
    loss_fn: Callable = F.mse_loss,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align t2 to t1.

    Parameters
    ----------
    t1 : torch.Tensor
        Reference tensor (..., num_powerset_classes)-shaped
    t2 : torch.Tensor
        Tensor to align (..., num_powerset_classes)-shaped
    powerset : Powerset
        Powerset information about the classes
    loss_fn : Callable, optional
        Loss function to compute the best alignement, by default torch.nn.functional.mse_loss

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (t2_aligned, best_permutation)
    """
    best_permutated: Optional[torch.Tensor] = None
    best_permutation: Optional[tuple[int]] = None
    best_score = torch.inf

    for permutation in itertools.permutations(range(powerset.num_classes), powerset.num_classes):
        permutation_ps = powerset.permutation_mapping[permutation]
        permutated_t2 = t2[..., permutation_ps]

        perm_loss = loss_fn(t1, permutated_t2).item()
        if perm_loss < best_score:
            best_permutated = permutated_t2
            best_score = perm_loss
            best_permutation = permutation_ps
    return best_permutated, torch.tensor(best_permutation, dtype=torch.int)


def lossy_match_speaker_count_and_permutate(
    t1: torch.Tensor,
    t2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    speaker_selection_mode: Literal["best_match", "activity"] = "best_match",
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Will align t2 to t1, but may discard the least relevant speakers if t2 has too many speakers.
    Not good for evaluation as speaker information may be lost.
    Good for putting t2 in the same space as t1 (multilabel <-> multiclass conversions), to compute heuristic losses
    or metrics that aren't supposed to be accurate.

    Parameters
    ----------
    t1 : torch.Tensor
        The reference tensor of size (batch_size, n_frames, num_speakers)
    t2 : torch.Tensor
        The tensor to align of size (batch_size, n_frames, num_speakers_wrong)
        where the tensor either has the same, too few or too many speakers.
    cost_func : Optional[Callable], optional
        Cost function to use for the alignment, by default None
    speaker_selection_mode : Literal['best_match', 'activity'], optional
        How to select the speakers to keep in t2 if it has too many speakers.
        'best_match' keeps the speakers that best match the speakers in t1.
        'activity' keeps the most active speakers in t2.

    Returns
    -------
    Tuple[torch.Tensor, List[torch.Tensor]]
        An aligned version of t2 (batch_size, n_frames, num_speakers)-shaped,
        where the least relevant speaker have been discarded.
        And a list of "permutation" (/w discard if needed) to reach it.
    """

    if speaker_selection_mode not in ["best_match", "activity"]:
        raise ValueError(
            f"speaker_selection_mode should be 'best_match' or 'activity'; got {speaker_selection_mode} instead."
        )
    if t1.ndim != 3 or t2.ndim != 3:
        raise ValueError("t1 and t2 should be (b,f,speaker_count) tensors")

    # first, remove all inactive speakers from t2
    t2 = t2[..., (t2.sum(dim=(0, 1)) > 0)]

    # if t2 has too few speakers, just pad with inactive speakers
    if t2.shape[-1] < t1.shape[-1]:
        missing_spk_cnt = t1.shape[-1] - t2.shape[-1]
        t2 = torch.nn.functional.pad(t2, (0, missing_spk_cnt), value=0.0)

    # if t2 has the right number of speakers, call the regular permutate
    if t1.shape[-1] == t2.shape[-1]:
        return permutate(t1, t2, cost_func=cost_func)

    # if t2 has too many speakers
    if t2.shape[-1] < t1.shape[-1]:
        raise Exception("Something went wrong in lossy_match_speaker_count_and_permutate.")

    # if we're there, t2 has too many speakers ! We need to discard the least relevant ones

    # list of (num_frames, num_speaker) tensors
    if speaker_selection_mode == "best_match":
        t2_aligned_batches = []
        best_perms = []
        for batch_idx in range(t1.shape[0]):
            u1: torch.Tensor = t1[batch_idx]
            u2 = t2[batch_idx]

            best_speaker_selection, best_permutation = None, None
            best_cost = torch.inf
            for selected_speakers in itertools.combinations(range(u2.shape[-1]), u1.shape[-1]):
                u2_select = u2[:, selected_speakers]
                u2_select_align, perms, costs = permutate(
                    u1[None, ...],
                    u2_select[None, ...],
                    cost_func=cost_func,
                    return_cost=True,
                )
                perm = perms[0]
                cost = costs[0].sum()
                if cost < best_cost:
                    best_cost = cost
                    best_permutation = perm
                    best_speaker_selection = selected_speakers
            t2_aligned_batches.append(
                u2[:, best_speaker_selection][..., best_permutation][None, ...]
            )
            best_perms.append(best_permutation)
            t2_aligned = torch.cat(t2_aligned_batches, dim=0)
    elif speaker_selection_mode == "activity":
        topk_active = t2.sum(dim=1).topk(dim=-1, k=t1.shape[-1]).indices
        t2_mostactive = t2.gather(dim=2, index=topk_active.unsqueeze(1).tile(1, t2.shape[1], 1))
        t2_aligned, best_perms = permutate(t1, t2_mostactive, cost_func=cost_func)
    return t2_aligned, best_perms


def match_speaker_count_and_permutate(
    t1: torch.Tensor,
    t2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    allow_empty_speaker_dim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Change both t1 and t2 so that they are "in the same space". This is good for evaluation, no speaker information is lost.
    This is not good for cross multilabel/multiclass conversion shenanigans: speaker channels may be dropped/padded.

    Parameters
    ----------
    t1 : torch.Tensor
        The reference tensor of size (batch_size, n_frames, num_speakers)
    t2 : torch.Tensor
        The tensor to align of size (batch_size, n_frames, num_speakers2)
    allow_empty_speaker_dim: bool, optional
        Prevent the output from collapsing to (batch_size, n_frames, 0)
        if no speakers are active in t1 and t2.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The tuple (t1_aligned, t2_aligned), where t1 and t2 have the same number of speakers.
        All inactive speaker channels have dropped if they do not correspond to any active channel in the other tensor.
        t2 has been aligned to t1.
    """
    if t1.ndim != 3 or t2.ndim != 3:
        raise ValueError(
            f"t1 and t2 should be (b,f,speaker_count) tensors; got {t1.shape} and {t2.shape} instead."
        )

    # first, remove all inactive speakers from t1 and t2
    t1 = t1[..., (t1.sum(dim=(0, 1)) > 0)]
    t2 = t2[..., (t2.sum(dim=(0, 1)) > 0)]

    # if t1/t2 have too few speakers, just pad with inactive speakers
    if t2.shape[-1] < t1.shape[-1]:
        t2_v2 = torch.zeros_like(t1)
        t2_v2[..., : t2.shape[-1]] = t2
        t2 = t2_v2
    elif t2.shape[-1] > t1.shape[-1]:
        t1_v2 = torch.zeros_like(t2)
        t1_v2[..., : t1.shape[-1]] = t1
        t1 = t1_v2

    # if no speaker in the end
    if t1.shape[-1] == 0 and t2.shape[-1] == 0:
        if not allow_empty_speaker_dim:
            t1 = torch.nn.functional.pad(t1, (1, 0), value=0)
            t2 = torch.nn.functional.pad(t1, (1, 0), value=0)
        return t1, t2

    # if t2 has the right number of speakers, call the regular permutate
    if t1.shape[-1] != t2.shape[-1]:
        raise Exception(
            "Something went wrong in match_speaker_count_and_permutate. This should not happen."
        )

    t2, _ = permutate(t1, t2, cost_func=cost_func)
    return t1, t2
