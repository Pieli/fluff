import torch
import utils


def test_logits_ensemble():
    nodes = 3
    classes = 10

    node_statistics = []
    resulting_node_logits = []
    for k in range(nodes):
        logits_of_batch = torch.sigmoid(torch.randn(32, 10))
        targets = torch.randint(0, 10, (32,))

        (result, count) = utils.average_logits_per_class(logits_of_batch, targets, classes)
        assert result.shape == (classes, classes)
        assert count.shape == (classes, 1)

        node_statistics.append(count)
        resulting_node_logits.append(result)

    counter = torch.stack(node_statistics)
    assert counter.shape == (nodes, classes, 1)

    node_weights = utils.node_weights(counter, classes, nodes)
    assert node_weights.shape == (nodes, classes, 1)

    node_logits = torch.stack(resulting_node_logits)
    assert node_logits.shape == (nodes, classes, classes)

    logits = utils.logits_ensemble(node_logits, node_weights, classes, nodes)
    assert logits.shape == (classes, classes)


def test_runnning_average():
    pass
