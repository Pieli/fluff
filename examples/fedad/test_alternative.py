import torch
import utils
from typing import Iterable


def test_alternative_simple():
    classes = 2
    nodes = 2
    batch = 2

    # nodes x (batch x num_classes)
    logits = list(torch.tensor([[[1, 1],
                                 [1, 1]],
                                [[3, 3],
                                 [4, 4]]]))

    # nodes x num_classes
    statistics = list(torch.tensor([[[0],
                                     [1]],

                                    [[1],
                                     [0]]]))
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
    logits = list(torch.tensor([[[2, 2],
                                 [2, 2],
                                 [2, 2],
                                 [2, 2]],

                                [[4, 4],
                                 [4, 4],
                                 [4, 4],
                                 [4, 4]],

                                [[4, 4],
                                 [4, 4],
                                 [4, 4],
                                 [4, 4]]
                                ]))

    # nodes x num_classes
    statistics = list(torch.tensor([[[1],
                                     [1]],

                                    [[0],
                                     [0]],

                                    [[1],
                                     [1]]
                                    ]))
    print(len(statistics))

    result = utils.alternative_avg(logits, statistics, classes, nodes)

    # batch x num_classes
    expected = torch.tensor([[3.0, 3.0],
                             [3.0, 3.0],
                             [3.0, 3.0],
                             [3.0, 3.0]])

    assert torch.allclose(result, expected)
