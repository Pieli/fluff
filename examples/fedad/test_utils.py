import torch
import utils


def test_masking_values():
    example_attention = torch.ones(3, 3)
    expected = torch.ones(3, 3) * 0.5

    rho = 1
    b = 1
    result = utils.masking(example_attention, rho, b)

    print(result)
    print(expected)
    assert torch.allclose(result, expected)


def test_masking_values_batch():
    example_attention = torch.ones(1, 3, 3)
    expected = torch.ones(1, 3, 3) * 0.5

    rho = 1
    b = 1
    result = utils.masking(example_attention, rho, b)

    print("results", result)
    print("expected", expected)
    assert torch.allclose(result, expected)


def test_union_simple():
    maps = torch.stack(
        [
            torch.tensor(
                [
                    [1, 2, 3],
                    [2, 2, 1],
                    [3, 2, 1],
                ]
            ),
            torch.tensor(
                [
                    [1, 2, 1],
                    [3, 2, 3],
                    [2, 2, 1],
                ]
            ),
            torch.tensor(
                [
                    [3, 2, 2],
                    [2, 2, 2],
                    [1, 2, 3],
                ]
            ),
        ]
    )

    results = utils.union(maps)

    expected = torch.tensor([[3, 2, 3]] * 3)

    assert torch.allclose(results, expected)


def test_intersection_simple():
    maps = torch.stack(
        [
            torch.tensor(
                [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            ),
            torch.tensor(
                [
                    [3, 2, 1],
                    [3, 2, 1],
                    [3, 2, 1],
                ]
            ),
            torch.tensor(
                [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ]
            ),
        ]
    )

    results = utils.intersection(maps)
    print(results)

    expected = torch.tensor([[1, 2, 1]] * 3)

    assert torch.allclose(results, expected)


def test_intersection_loss_no_outliers():
    inter = torch.ones(1, 3, 3)
    att = torch.ones(1, 3, 3) * 6

    result = utils.loss_intersection(inter, att, num_classes=1)

    # result would be -1 (error)
    assert torch.isclose(result, torch.tensor([-1.0]), atol=5e-2).item()


def test_intersection_loss_half_inliers_simple():
    inter = torch.zeros(1, 2, 2)
    inter[:, 1, :] = 1
    print(inter)

    att = torch.zeros(1, 2, 2)
    att[:, 1, 1] = 6
    print(att)

    result = utils.loss_intersection(inter, att, num_classes=1)

    # result would be -0.5 (error)
    assert torch.isclose(result, torch.Tensor([-0.5]), atol=2e-1).item()


def test_intersection_zero_division():
    inter = torch.zeros(1, 2, 2)
    print(inter)

    att = torch.zeros(1, 2, 2)
    print(att)

    result = utils.loss_intersection(inter, att, num_classes=1)

    # result would be -0.0 (error)
    assert torch.isclose(result, torch.Tensor([-0.0]), atol=2e-1).item()


# result would be -0.5 (error)
def test_intersection_loss_half_inliers():
    num_class = 2
    inter = torch.zeros(num_class, 2, 2)
    inter[:, 1, :] = 0.6
    print(inter)

    att = torch.zeros(num_class, 2, 2)
    att[:, 1, 1] = 6
    print(att)

    result = utils.loss_intersection(inter, att, num_classes=num_class)

    assert torch.isclose(result, torch.Tensor([-0.5]), atol=2e-1).item()


# result would be -0 (error)
def test_intersection_loss_all_outliers():
    num_class = 1
    inter = torch.zeros(num_class, 2, 2)
    inter[:, 0, :] = 0.5

    att = torch.zeros(num_class, 2, 2)
    att[:, 1, :] = 0.9

    result = utils.loss_intersection(inter, att, num_classes=num_class)

    assert torch.isclose(result, torch.Tensor([-0.0]), atol=2e-1).item()


def test_union_loss_no_outliers():
    num_class = 1
    union = torch.ones(num_class, 3, 3) * 6
    att = torch.ones(num_class, 3, 3)
    result = utils.loss_union(union, att, num_classes=num_class)

    assert torch.isclose(result, torch.tensor([-1.0]), atol=5e-2).item()


def test_union_loss_half_inliers():
    num_class = 1
    union = torch.zeros(num_class, 2, 2)
    union[:, :, 1] = 6

    att = torch.zeros(num_class, 2, 2)
    att[:, 1, :] = 0.9

    result = utils.loss_union(union, att, num_classes=num_class)

    assert torch.isclose(result, torch.Tensor([-0.5]), atol=2e-1).item()


def test_union_loss_all_outliers():
    num_class = 1
    union = torch.zeros(num_class, 2, 2)
    union[:, 0, :] = 0.5

    att = torch.zeros(num_class, 2, 2)
    att[:, 1, :] = 0.9

    result = utils.loss_union(union, att, num_classes=num_class)

    assert torch.isclose(result, torch.Tensor([-0.0]), atol=2e-1).item()


def test_union_loss_all_outliers_part_2():
    num_class = 1
    union = torch.zeros(num_class, 2, 2)

    att = torch.zeros(num_class, 2, 2)
    att[:, 1, :] = 0.9

    result = utils.loss_union(union, att, num_classes=num_class)

    assert torch.isclose(result, torch.Tensor([-0.0]), atol=2e-1).item()


def test_node_weights_zero_present():
    nodes = 2
    classes = 4

    node_statistics = torch.tensor([[[1], [2], [0], [0]], [[4], [2], [0], [0]]])

    result = utils.node_weights(node_statistics, classes, nodes)

    expected = torch.tensor(
        [
            [[0.2000], [0.5000], [0.0000], [0.0000]],
            [[0.8000], [0.5000], [0.0000], [0.0000]],
        ]
    )

    assert torch.isnan(result).sum() == 0
    assert torch.allclose(result, expected)


def test_node_weights_all_equal():
    nodes = 4
    classes = 10

    node_statistics = torch.ones(nodes, classes, 1)

    result = utils.node_weights(node_statistics, classes, nodes)

    expected = torch.ones(nodes, classes, 1) / nodes

    assert torch.allclose(result, expected)


def test_sample_with_top_simple():
    nodes = 3
    classes = 3
    num_out = 1
    top = 1

    node_statistics = torch.ones(nodes, classes) + torch.eye(nodes)
    node_statistics = node_statistics.unsqueeze(2)
    print(node_statistics)
    print(node_statistics.shape)

    weights = utils.node_weights(node_statistics, classes, nodes)
    print(weights)
    samples = utils.sample_with_top(weights, num_out_samples=num_out, top=top)

    print(samples)
    for ind, sam in enumerate(samples):
        assert sam[0] == ind
        assert len(sam) == num_out + top


def test_sample_with_top_simple_2():
    nodes = 6
    classes = 6
    num_out = 3
    top = 1

    node_statistics = torch.ones(nodes, classes) + torch.eye(nodes)
    node_statistics = node_statistics.unsqueeze(2)
    print(node_statistics.shape)

    weights = utils.node_weights(node_statistics, classes, nodes)
    samples = utils.sample_with_top(weights, num_out_samples=num_out, top=top)

    print(samples)
    for ind, sam in enumerate(samples):
        assert sam[0] == ind
        assert len(sam) == num_out + top


def test_sample_with_top_without_num():
    nodes = 6
    classes = 6
    num_out = 0
    top = 1

    node_statistics = torch.ones(nodes, classes) + torch.eye(nodes)
    node_statistics = node_statistics.unsqueeze(2)
    print(node_statistics.shape)

    weights = utils.node_weights(node_statistics, classes, nodes)
    samples = utils.sample_with_top(weights, num_out_samples=num_out, top=top)

    print(samples)
    for ind, sam in enumerate(samples):
        assert sam[0] == ind
        assert len(sam) == num_out + top


def test_sample_with_top_simple_3():
    nodes = 6
    classes = 6
    num_out = 3

    node_statistics = torch.ones(nodes, classes, 1)
    for num in range(nodes):
        node_statistics[num, :, :] = num

    weights = utils.node_weights(node_statistics, classes, nodes)
    samples = utils.sample_with_top(weights, num_out_samples=num_out, top=2)

    for ind, sam in enumerate(samples):
        assert len(sam) == num_out + 2
        assert 0 not in sam


def test_node_weights_all_unequal():
    nodes = 2
    classes = 4

    node_statistics = torch.tensor([[[1], [2], [3], [4]], [[4], [3], [2], [1]]])

    result = utils.node_weights(node_statistics, classes, nodes)

    expected = torch.tensor(
        [
            [[0.2000], [0.4000], [0.6000], [0.8000]],
            [[0.8000], [0.6000], [0.4000], [0.2000]],
        ]
    )
    print(result)

    assert torch.allclose(result, expected)


def test_logits_ensemble_simple():
    classes = 2
    nodes = 2

    node_weights = torch.tensor([[[1.0], [0.5]], [[0.0], [0.5]]])

    logits = torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])

    result = utils.logits_ensemble(logits, node_weights, classes, nodes)

    expected = torch.tensor([[1.0, 1.0], [3.0, 3.0]])

    assert torch.allclose(result, expected)


def test_average_logits_simple():

    logits = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

    targets = torch.tensor([0, 1, 2, 2])

    num_classes = 3

    avg_l = utils.average_logits_per_class(logits, targets, num_classes)

    assert torch.allclose(avg_l, torch.ones(num_classes, num_classes))


def test_average_logits_simple_2():

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0],
            [0.5, 1.5, 2.5, 3.5],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    # Gold labels
    targets = torch.tensor([0, 1, 2, 1, 3])

    # Compute average logits per class
    num_classes = 4
    avg_l = utils.average_logits_per_class(logits, targets, num_classes)

    answer = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert torch.allclose(avg_l, answer)


def test_alternative_simple():
    classes = 2
    nodes = 2
    batch = 2

    # nodes x (batch x num_classes)
    logits = list(torch.tensor([[[1, 1], [1, 1]], [[3, 3], [4, 4]]]))

    # nodes x num_classes
    statistics = list(torch.tensor([[[0], [1]], [[1], [0]]]))
    print(len(statistics))

    result = utils.alternative_avg(logits, statistics, classes, nodes)

    # batch x num_classes
    expected = torch.tensor([[3.0, 1.0], [4.0, 1.0]])

    assert torch.allclose(result, expected)


def test_alternative_simple_v2():
    classes = 2
    nodes = 3
    batch = 4

    # nodes x (batch x num_classes)
    logits = list(
        torch.tensor(
            [
                [[2, 2], [2, 2], [2, 2], [2, 2]],
                [[4, 4], [4, 4], [4, 4], [4, 4]],
                [[4, 4], [4, 4], [4, 4], [4, 4]],
            ]
        )
    )

    # nodes x num_classes
    statistics = list(torch.tensor([[[1], [1]], [[0], [0]], [[1], [1]]]))
    print(len(statistics))

    result = utils.alternative_avg(logits, statistics, classes, nodes)

    # batch x num_classes
    expected = torch.tensor([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])

    assert torch.allclose(result, expected)
