import torch
from torch.nn import functional

# -------------------------------------------------------------------------------------------------------------------


def sample_unit_l1_weights(k):

    weights = torch.rand(size=(1, k))
    weights = weights / weights.sum(dim=1)

    return weights

# -------------------------------------------------------------------------------------------------------------------


def micro_cluster_loss(groups):

    total_loss = 0
    total_aux_loss = 0

    for yi in groups:

        errors = groups[yi]['distances']

        n, k = errors.shape[0:2]

        weights = sample_unit_l1_weights(k)

        logits = groups[yi]['logits']
        y = torch.as_tensor([yi] * n, dtype=torch.long)

        p = torch.softmax(-weights * errors, dim=1)

        loss = functional.cross_entropy(logits, y, reduction='none')
        loss = p * loss.unsqueeze(1)

        total_loss += loss.sum(1).mean(0)
        total_aux_loss += p.sum(1).mean(0)

    return total_loss, total_aux_loss

# -------------------------------------------------------------------------------------------------------------------
