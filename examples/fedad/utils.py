import numpy as np
import torch
import torch.nn.functional as F

from typing import Iterable, cast, Optional


def log_softmax_mod(
    input: torch.Tensor,
    targets: torch.Tensor,
    dim: Optional[int] = None,
):

    batch_size, _ = input.shape

    # Clone logits to keep gradients
    masked_logits = input.clone()
    masked_logits[torch.arange(batch_size), targets] = float("-inf")

    log = F.log_softmax(masked_logits, dim)
    log[torch.arange(batch_size), targets] = 0.0
    return log

def softmax_mod(
    input: torch.Tensor,
    majority: torch.Tensor,
    dim: Optional[int] = None,
):

    # Clone logits to keep gradients
    masked_logits = input.clone()
    masked_logits[:, majority] = float("-inf")

    return F.softmax(masked_logits, dim)


def l1_distillation(
    server_log: torch.Tensor,
    ensemble_log: torch.Tensor,
    T: int = 3,
) -> torch.Tensor:

    return F.l1_loss(
        server_log,
        ensemble_log,
        reduction="mean",
    )


def l2_distillation(
    server_log: torch.Tensor,
    ensemble_log: torch.Tensor,
    T: int = 3,
) -> torch.Tensor:

    return F.mse_loss(
        torch.sigmoid(server_log),
        torch.sigmoid(ensemble_log),
        reduction="mean",
    )


def l2_distillation_new(
    server_log: torch.Tensor,
    ensemble_log: torch.Tensor,
    T: int = 3,
) -> torch.Tensor:

    return (
        torch.linalg.vector_norm(
            server_log - ensemble_log,
            ord=2,
        )
        / 10
    )


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    T: int = 3,
) -> torch.Tensor:

    return (
        F.kl_div(
            F.log_softmax(p / T, dim=1),
            F.softmax(q / T, dim=1),
            reduction="batchmean",
        )
        * T
        * T
    )


def sample_with_top(
    weights: torch.Tensor,
    num_out_samples: int,
    top: int = 2,
) -> list[list[int]]:
    """
    Args:
        weights (torch.Tensor):
            3D tensor of shape (num_nodes, num_classes, 1),
            containing the node weights for each class.

    Return:
    """

    num_nodes, num_classes, _ = weights.size()

    assert isinstance(top, int)
    assert isinstance(num_out_samples, int)
    assert isinstance(weights, torch.Tensor)
    assert weights.size()[-1] == 1
    assert top < num_nodes
    assert num_out_samples + top <= num_nodes

    sampled: list[list[int]] = []
    for c in range(num_classes):
        # add top places caused sampled with prob 1
        top_places = weights.sort(dim=0, descending=True).indices[:top, c, :].squeeze(1)
        sampled.append(top_places.tolist())

        if num_out_samples == 0:
            continue

        # create candidates
        pool = [cand for cand in range(weights.size(0)) if cand not in top_places]

        # filter out top places
        pool_probs_raw = [
            prob
            for ind, prob in enumerate(weights[:, c, :].squeeze().tolist())
            if ind not in top_places
        ]

        # adjust to sum up to 1 + all zero case
        probs_sum = sum(pool_probs_raw)
        if probs_sum == 0.0:
            pool_probs_raw = [1] * len(pool_probs_raw)
            probs_sum = len(pool_probs_raw)

        p = [prob / probs_sum for prob in pool_probs_raw]

        # single 1 + all zero case
        if 1.0 in p and len(p) > 1:
            rest = 0.05 / (len(p) - 1)
            p = [0.95 if val == 1 else rest for val in p]

        # sample missing values
        sampled[c].extend(
            cast(
                list[int],
                np.random.choice(
                    pool,
                    num_out_samples,
                    replace=False,
                    p=p,
                ).tolist(),
            )
        )

    return sampled


# TODO -> change this name
def alternative_avg(
    raw_logits: Iterable[torch.Tensor],
    raw_statistics: Iterable[torch.Tensor],
    num_classes: int,
    num_nodes: int,
) -> torch.Tensor:

    assert isinstance(raw_logits, (list, tuple))
    assert isinstance(raw_statistics, (list, tuple))
    assert len(raw_logits) == len(raw_statistics)

    logits = torch.stack(raw_logits)
    node_statistics = torch.stack(raw_statistics)
    weights = (
        node_weights(
            node_statistics,
            num_classes,
            len(logits),
        )
        .squeeze(2)
        .unsqueeze(1)
    )

    return torch.sum((logits * weights), dim=0)


# TODO -> remove?
def logits_ensemble_eq_3(
    raw_logits: Iterable[torch.Tensor],
    raw_statistics: Iterable[torch.Tensor],
    num_classes: int,
    num_nodes: int,
) -> torch.Tensor:

    assert isinstance(raw_logits, (list, tuple))
    assert isinstance(raw_statistics, (list, tuple))
    assert len(raw_logits) == len(raw_statistics)

    node_logits = torch.stack(raw_logits)
    node_statistics = torch.stack(raw_statistics)
    weights = node_weights(node_statistics, num_classes, len(node_logits))
    return logits_ensemble(node_logits, weights, num_classes, num_nodes)


# fedad - node weights for equation 3
def node_weights(
    node_stats: torch.Tensor, num_classes: int, num_nodes: int
) -> torch.Tensor:
    """
    Computes the node weights for each class.

    Args:
        node_stats (torch.Tensor):
            3D tensor of shape (num_nodes, num_classes, 1),
            containting the number of samples for each class

    Returns:
        torch.Tensor:
            3D tensor of shape (num_nodes, num_classes, 1),
            containing the node weights for each class.
    """
    assert node_stats.shape == (num_nodes, num_classes, 1)

    num_per_class = node_stats.sum(dim=0)
    non_zero = torch.where(
        num_per_class == 0, torch.ones_like(num_per_class), num_per_class
    )
    node_based_weight = node_stats / non_zero
    return node_based_weight


# fedad - equation 3
def logits_ensemble(
    logits: torch.Tensor,
    node_weights: torch.Tensor,
    num_classes: int,
    num_nodes: int,
) -> torch.Tensor:
    r"""
    (\hat{z}^c) weighted average of nodes per class

    Args:
        logits (torch.Tensor):
            - avg logit per node for a certain class
            - shape (len(logit), number of classes)

        node_weights (torch.Tensor):
            - weights per node
            - shape (1, number of nodes)

    Returns:
        torch.Tensor:
            - weighted logit per class
            - shape (number of classes,1)

    """

    assert logits.shape == (num_nodes, num_classes, num_classes)
    assert node_weights.shape == (num_nodes, num_classes, 1)

    return (logits * node_weights).sum(dim=0)


def average_logits_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    r"""
    Computes the average logit for samples for the given label. (\hat{z}^c_k)

    Args:
        logits (torch.Tensor): A 2D tensor of shape (batch_size, num_classes),
                               containing the logits for each class.
        targets (torch.Tensor): A 1D tensor of shape (batch_size,),
                                containing the gold class labels (0-indexed).
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: A 2D tensor of shape (targets, num_classes), where each element
                      is the average logit across all samples for the corresponding class.
    """
    assert logits.dim() == 2, "must be 2D with shape (batch_size, num_classes)"
    assert (
        logits.size(1) == num_classes
    ), "must be 2D with shape (batch_size, num_classes)"
    assert targets.dim() == 1, "targets must be 1D tensor"
    assert targets.size(0) == logits.size(
        0
    ), "targets have the same batch size as logits."

    # Create tensors to store cumulative logits and sample counts
    logit_sums = torch.zeros(num_classes, num_classes, device=logits.device)
    class_counts = torch.zeros(num_classes, device=logits.device)

    # Iterate over each class
    for i in range(num_classes):
        # Mask for samples belonging to class i
        mask = targets == i
        # Sum all logits for samples of class i
        logit_sums[i] = logits[mask].sum(dim=0)
        # Count samples of class i
        class_counts[i] = mask.sum()

    # Avoid division by zero by replacing zero counts with ones
    counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
    counts = torch.unsqueeze(counts, dim=1)

    # also unsqueeze the real count
    class_counts = torch.unsqueeze(class_counts, dim=1)

    # Compute the average logit per class
    avg_logits = logit_sums / counts

    return avg_logits


# fedad - equation 7
def masking(attenion_map, rho, b):
    """
    masking = T(.) = soft maxing operation

    b = threshold value
    rho = scaling value
    """

    return torch.sigmoid(rho * (attenion_map - b))


# fedad - equation 6
def intersection(maps: torch.Tensor):
    """
    calculates the intersection of the attention maps.
    The intersection equals to minimum value of the attention maps.
    """
    assert isinstance(maps, torch.Tensor)
    return maps.amin(dim=(0, 1))


# fedad - equation 6
def union(maps: torch.Tensor):
    """
    calculates the union of the attention maps.
    The union equals to maximum value of the attention maps.
    """

    assert isinstance(maps, torch.Tensor)
    return maps.amax(dim=(0, 1))


# fedad - equation 8
def loss_intersection(
    intersections: torch.Tensor,
    attentions: torch.Tensor,
    num_classes=10,
):
    # intesections include all classes
    # attentions include also all classes
    """
    intersections: torch.Size[num_classes, height, width]
    attentions: torch.Size[num_classes, height, width]

    """

    assert num_classes > 0
    assert intersections.shape == attentions.shape

    # weight the intersection with the attention map mask
    # Sum up the weighted intersection map
    weighted = intersections * masking(attentions, rho=10, b=0.6)
    weighted_sum = weighted.sum(dim=(1, 2, 3))

    # Sum of pixels in the non-weighted intersection
    # epsilon is added to avoid division by zero
    class_sums = intersections.sum(dim=(1, 2, 3))
    non_zero_class_sum = torch.where(
        class_sums == 0.0,
        torch.ones_like(class_sums) * torch.finfo(torch.float64).eps,
        class_sums,
    )

    result = (-1 / num_classes) * (weighted_sum / non_zero_class_sum).sum()
    # print(result)
    return result


# fedad - equation 9
def loss_union(unions: torch.Tensor, attentions: torch.Tensor, num_classes=10):
    """
    unions: torch.Size[num_classes, height, width]
    attentions: torch.Size[num_classes, height, width]
    """
    assert unions.shape == attentions.shape

    # weight the intersection with the attention map mask
    # Sum up the weighted intersection map
    weighted = attentions * masking(unions, rho=10, b=0.3)
    weighted_sum = weighted.sum(dim=(1, 2, 3))

    # Sum of pixels in the non-weighted intersection
    # epsilon is added to avoid division by zero
    class_sums = attentions.sum(dim=(1, 2, 3))
    non_zero_class_sum = torch.where(
        class_sums == 0.0,
        torch.ones_like(class_sums) * torch.finfo(torch.float64).eps,
        class_sums,
    )

    result = (-1 / num_classes) * (weighted_sum / non_zero_class_sum).sum()
    # print("uinion", result)
    return result
