import numpy as np
import torch

# --------------------------------------------------------------------------------------------------------------------


def unique(a, dim=-1):

    _, perm_idx = np.unique(a.numpy(), return_index=True, axis=dim)
    perm_idx = np.sort(perm_idx)

    perm_values = a[perm_idx]

    return perm_values

# --------------------------------------------------------------------------------------------------------------------


def get_permute_for_gather_axes(tensor, perm_axes, last=False):

    rank = tensor.ndim
    axes = torch.arange(rank)
    perm_axes = torch.as_tensor(perm_axes, dtype=torch.int32)

    if last:

        perm_axes = torch.flip(perm_axes, dims=(0, ))

    perm_axes = torch.concat([perm_axes, axes], dim=-1)
    perm_axes = unique(perm_axes.detach()).tolist()

    if last:

        perm_axes = perm_axes[::-1]

    return perm_axes

# --------------------------------------------------------------------------------------------------------------------


def permute_for_gather(tensor, axis, last=False):

    axes = get_permute_for_gather_axes(tensor, axis, last)

    return tensor.permute(axes)

# --------------------------------------------------------------------------------------------------------------------


def undo_permute_for_gather(tensor, axis, last=False):

    axes = get_permute_for_gather_axes(tensor, axis, last)

    return tensor.permute(torch.argsort(axes))

# --------------------------------------------------------------------------------------------------------------------


def flatten_for_gather(tensor, axis, last=False, keepdims=False):

    n = len(axis)
    shape = torch.as_tensor(tensor.shape, dtype=torch.int32)

    if last:

        k = torch.prod(shape[-n:], dim=-1, keepdim=keepdims)

        if keepdims:

            out_shape = torch.concat([shape[:-n], k], dim=-1)

        else:

            out_shape = [-1, k]

    else:

        k = torch.prod(shape[:n], dim=-1, keepdim=True)

        if keepdims:

            out_shape = torch.concat([k, shape[n:]], dim=-1)

        else:

            out_shape = [k, -1]

    a = torch.reshape(tensor, out_shape)

    return a

# --------------------------------------------------------------------------------------------------------------------
