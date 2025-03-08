import utils
import torch


def test_log_sofmax_mod_1():
    a = torch.ones(3, 3)

    targets = torch.tensor([1, 1, 1])

    p = utils.log_softmax_mod(a, targets, dim=1)

    print(p)
    assert torch.allclose(
        p,
        torch.tensor(
            [
                [-0.6931, 0.0000, -0.6931],
                [-0.6931, 0.0000, -0.6931],
                [-0.6931, 0.0000, -0.6931],
            ]
        ),
        rtol=1e-4,
    )


def test_log_sofmax_mod_2():
    a = torch.ones(3, 3)

    targets = torch.tensor([0, 1, 2])

    p = utils.log_softmax_mod(a, targets, dim=1)

    print(p)
    assert torch.allclose(
        p,
        torch.tensor(
            [
                [0.0000, -0.6931, -0.6931],
                [-0.6931, 0.0000, -0.6931],
                [-0.6931, -0.6931, 0.0000],
            ]
        ),
        rtol=1e-4,
    )


def test_sofmax_mod_1():
    a = torch.ones(3, 3)

    targets = torch.tensor([1, 2])

    p = utils.softmax_mod(a, targets, dim=1)

    print(p)
    assert torch.allclose(
        p,
        torch.tensor(
            [
                [1.0000, 0.0000, 0.0000],
                [1.0000, 0.0000, 0.0000],
                [1.0000, 0.0000, 0.0000],
            ]
        ),
        rtol=1e-4,
    )


def test_sofmax_mod_2():
    a = torch.ones(3, 3)

    targets = torch.tensor([0])

    p = utils.softmax_mod(a, targets, dim=1)

    print(p)
    assert torch.allclose(
        p,
        torch.tensor(
            [
                [0.0000, 0.5000, 0.5000],
                [0.0000, 0.5000, 0.5000],
                [0.0000, 0.5000, 0.5000],
            ]
        ),
        rtol=1e-4,
    )


def test_sofmax_mod_3():
    a = torch.ones(3, 6)

    targets = torch.tensor([0])

    p = utils.softmax_mod(a, targets, dim=1)

    print(p)
    assert torch.allclose(
        p,
        torch.tensor(
            [
                [0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            ]
        ),
        rtol=1e-4,
    )
