"""Holds miscellaneous PyTorch utility functions."""

from typing import Iterable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def get_device_in_objects(*args) -> torch.device:
    for obj in args:
        if obj.device.type != "cpu":
            return obj.device
    return torch.device("cpu")


def unique_consecutive_padded(
    x: torch.Tensor, pad_value: int = -1, return_indices: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Same as torch.unique_consecutive, but the resulting sequence is padded where it isn't long enough.
    For exemple with x=(1,1,2,2) -> result=(1,2,-1,-1)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    pad_value : int, optional
        Value to pad with, by default -1
    return_indices : bool, optional
        Should it return the indices, by default False

    Returns
    -------
    torch.Tensor
        The padded unique consecutive tensor.
    """
    unique_x, indices = torch.unique_consecutive(x, return_inverse=True)
    indices -= indices.min(dim=-1, keepdims=True)[0]
    result = pad_value * torch.ones_like(x)
    result = result.scatter_(-1, indices, x)
    if return_indices:
        return result, indices
    else:
        return result


def replace_value_with_last(tensor: torch.Tensor, to_replace=-1) -> torch.Tensor:
    """Replaces the specified values in the tensor with the value preceding it.
    For example,
        input  = [0,-1,5,-1,-1,6]
        output = [0,0,5,5,5,6]

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to operate on. Any dimension *should* work.
    to_replace, optional
        The value to be replaced/filled, by default -1.

    Returns
    -------
    torch.Tensor
        Resulting modified tensor.
    """
    uniques, inverse, counts = torch.unique_consecutive(
        tensor, return_inverse=True, return_counts=True
    )
    uniques_indexes = (counts).cumsum(dim=0)
    result = tensor.clone().flatten()
    for i in range(1, len(uniques)):
        if uniques[i] == to_replace:
            start_idx = uniques_indexes[i - 1]
            end_idx = uniques_indexes[i - 1] + counts[i]
            # the tensor is flattened: make sure it doesnt overflow to the next dim
            end_idx = min(end_idx, ((start_idx // tensor.shape[-1]) + 1) * tensor.shape[-1])
            result[start_idx:end_idx] = uniques[i - 1]
    return result.reshape(tensor.shape)


def create_slice_mask(
    shape,
    indices_start: Optional[torch.Tensor] = None,
    indices_stop: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Given a shape and start/stop indices, creates a 'slice' mask that's true between these indices.
    If shape=(a,b,c,d), the indices should be of shape (a,b,c).

    Parameters
    ----------
    shape : _type_
        Mask shape
    indices_start : torch.Tensor, optional
        Starting indices for the 'slice' mask, by default None
    indices_stop : torch.Tensor, optional
        Stopping indices (**excluded**) for the 'slice' mask, by default None

    Returns
    -------
    torch.Tensor
        Boolean mask of shape 'shape'.

    Raises
    ------
    ValueError
        If a start index is greater than its corresponding stop index.
    """
    device = get_device_in_objects(indices_start, indices_stop)
    # set default values if None is provided
    if indices_start is None:
        indices_start = torch.zeros(shape[:-1], device=device)
    if indices_stop is None:
        indices_stop = torch.ones(shape[:-1], device=device) * shape[-1]

    if torch.any(indices_start > indices_stop):
        raise ValueError("indices_start must be <= indices_stop")
    arange = torch.arange(shape[-1])

    while arange.ndim < len(shape):
        arange = arange[None, ...]

    mask = (arange >= indices_start[..., None]) & (arange < indices_stop[..., None])  # type: ignore
    return mask


def insert_at_indices(
    x: Tensor,
    indexes: Tensor,
    value: Union[bool, float, int, Tensor],
) -> Tensor:
    """Insert value in vector at specified indexes.

    Example 1
    ----------
    >>> x = torch.as_tensor([1, 1, 2, 2, 2, 3])
    >>> indexes = torch.as_tensor([2, 5])
    >>> values = 4
    >>> insert_values(x, indexes, values)
    tensor([1, 1, 4, 2, 2, 2, 4, 3])

    Parameters
    ----------
    x : Tensor
        Input tensor
    indexes : Tensor
        Tensor of indices
    value : Union[bool, float, int, Tensor]
        Value to insert. Or tensor of values with same shape as indexes.

    Returns
    -------
    Tensor
        Tensor with inserted values.
    """
    out = torch.empty((x.shape[0] + indexes.shape[0]), dtype=x.dtype, device=x.device)
    indexes = indexes + torch.arange(indexes.shape[0], device=indexes.device)
    out[indexes] = value
    mask = torch.full((out.shape[0],), True, dtype=torch.bool)
    mask[indexes] = False
    out[mask] = x
    return out


def insert_at_indices_revert(x: Tensor, indices: Tensor) -> Tensor:
    """Remove values such that it cancels an 'insert_at_indices' call.
    NOTE: be careful, this will not remove elements at the exact indices
    provided, because insert_at_indices will shift the indices to the right.

    Example 1
    ----------
    >>> x = torch.as_tensor([1, 1, -1, 2, 2, 2, -1, 3])
    >>> indices = torch.as_tensor([2, 5])
    >>> remove_at_indices(x, indices)
    tensor([1, 1, 2, 2, 2, 3])

    Parameters
    ----------
    x : Tensor
        Input tensor
    indexes : Tensor
        Tensor of indices previously passed to insert_at_indices.

    Returns
    -------
    Tensor
        Tensor where previously inserted values have been removed.
    """
    indices_offset = indices + torch.arange(indices.shape[0], device=indices.device)
    return remove_at_indices(x, indices_offset)


def remove_at_indices(x: Tensor, indices: Tensor) -> Tensor:
    """Remove values at indices. Basically util to apply a mask.

    Example 1
    ----------
    >>> x = torch.as_tensor([1, 1, 2, 2, 2, 3])
    >>> indices = torch.as_tensor([0, 2])
    >>> remove_at_indices(x, indices)
    tensor([1, 2, 2, 3])

    Parameters
    ----------
    x : Tensor
        Input tensor
    indices : Tensor
        Indices where values should be removed.

    Returns
    -------
    Tensor
        Tensor where values at indices have been removed.
    """
    mask = torch.full((x.shape[0],), True, dtype=torch.bool)
    indices = indices
    mask[indices] = False
    return x[mask]


def pad_and_stack(
    tensors: Iterable[Tensor],
    dim_pad: int,
    dim_stack: int,
    fill_value: float,
) -> Tensor:
    """Pad tensors to the same length and stack them.
    NOTE: Only tested for list of 1D tensor, building a 2D tensor.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        List of tensors to pad under the same final tensor.
    dim_pad : int
        Padding dimension.
    fill_value : float
        Value to use for padding.

    Returns
    -------
    Tensor
        Padded and stacked tensor.
    """
    target_len = max(tensor.shape[dim_pad] for tensor in tensors)
    tensors = [pad_dim(tensor, target_len, "left", fill_value, dim_pad) for tensor in tensors]
    tensors = torch.stack(tensors, dim=dim_stack)
    return tensors


def pad_dim(
    x: Tensor,
    target_length: int,
    align: Literal["left", "right", "center", "random"] = "left",
    fill_value: float = 0.0,
    dim: int = -1,
    mode: str = "constant",
) -> Tensor:
    """
    Generic function for pad a specific tensor dimension.
    Credit: Etienne LABBE
    """
    missing = max(target_length - x.shape[dim], 0)

    if missing == 0:
        return x

    if align == "left":
        missing_left = 0
        missing_right = missing
    elif align == "right":
        missing_left = missing
        missing_right = 0
    elif align == "center":
        missing_left = missing // 2 + missing % 2
        missing_right = missing // 2
    elif align == "random":
        missing_left = int(torch.randint(low=0, high=missing + 1, size=()).item())
        missing_right = missing - missing_left
    else:
        raise ValueError(
            f"Invalid argument {align=}. (expected one of {'left', 'right', 'center', 'random'})"
        )

    # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
    idx = len(x.shape) - (dim % len(x.shape)) - 1
    pad_seq = [0 for _ in range(len(x.shape) * 2)]
    pad_seq[idx * 2] = missing_left
    pad_seq[idx * 2 + 1] = missing_right
    x = F.pad(x, pad_seq, mode=mode, value=fill_value)
    return x
